"""Single-variable ablation: strict baseline + ONE knob. Trains, then probes (bar-pointer
read-out + dependency sensitivities) and writes result.json. faithful/ stays pure."""
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
import numpy as np, torch
import torch.nn.functional as F
from faithful.data import FPS, N_MELS, LogMel, build_train_loader, iter_val_songs
from faithful.model import BarPointerVAE
from faithful.distributions import (TWO_PI, gumbel_softmax, sample_von_mises,
                                    kl_categorical, kl_von_mises, kl_log_normal)
from faithful.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure


def rollout(model, h, b, temp, beta, fb, tclamp, pos_w, overshoot=1, os_fn=0.0, os_w=1.0):
    """strict_elbo + knobs: beta (KL weight), fb=(m,p,t) per-step KL floors, tclamp=(lo,hi)
    log-tempo clamp, pos_w (BCE pos_weight). All-off (beta=1,fb=0,tclamp=None,pos_w=1)=strict.

    LATENT OVERSHOOTING (PlaNet, Hafner 2019), continuous latents (phase+tempo):
    for each target t and distance d in 2..overshoot, roll the PRIOR forward d steps from the
    (stop-gradient) posterior sample at t-d and add KL(stop_grad q(z_t) || p^{(d)}(z_t)), with
    free-nats os_fn per term. The d=1 term is the ordinary one-step ELBO KL (posterior-grad KEPT)
    and is left untouched -> overshoot=1 reproduces strict_elbo EXACTLY (known-answer test). The
    prior mean rollout uses constant log-tempo mean (random-walk mean = previous) and advances
    phase by d*exp(lt_seed); kappa/sigma are read at the target frame pc[:,t]. Meter is NOT
    overshot (its prior transition is wrap-gated and not a clean random walk) -- documented scope:
    overshooting targets the divergent continuous random-walk states (phase backward / tempo blow-up)."""
    B, T, _ = h.shape
    pc = model.encode_prior(h); qc = model.encode_posterior(h, b)
    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B); kl_t = h.new_zeros(B)
    zf = []
    s_phi, s_lt = [], []                 # detached posterior samples (overshoot seeds)
    q_pm, q_pk, q_tm, q_ts = [], [], [], []   # posterior params (overshoot targets, detached at use)
    def clamp_lt(x):
        return x.clamp(tclamp[0], tclamp[1]) if tclamp else x
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init_head(pc.mean(1)))
    meter = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI
    lt = clamp_lt(qtm + qts * torch.randn_like(qtm))
    kl_m = kl_m + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm, -1)).clamp(min=fb[0])
    kl_p = kl_p + kl_von_mises(qpm, qpk, ppm, ppk).clamp(min=fb[1])
    kl_t = kl_t + kl_log_normal(qtm, qts, ptm, pts).clamp(min=fb[2])
    zf.append(model.z_features(meter, phi, lt)); mp, pp, lp = meter, phi, lt
    s_phi.append(phi.detach()); s_lt.append(lt.detach())
    q_pm.append(qpm); q_pk.append(qpk); q_tm.append(qtm); q_ts.append(qts)
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, t], model.z_features(mp, pp, lp)], -1)))
        tprev = torch.exp(lp)
        ppm = (pp + tprev) % TWO_PI
        ppk = F.softplus(model.prior_phase_kappa(pc[:, t]).squeeze(-1)) + 0.01
        ptm = lp
        pts = F.softplus(model.prior_tempo_sigma(pc[:, t]).squeeze(-1)) + 1e-3
        meter = gumbel_softmax(qm, temp); phi = sample_von_mises(qpm, qpk) % TWO_PI
        lt = clamp_lt(qtm + qts * torch.randn_like(qtm))
        logpi = model.meter_prior_logp(mp, phi, pp, pc[:, t])
        kl_m = kl_m + kl_categorical(torch.log_softmax(qm, -1), logpi).clamp(min=fb[0])
        kl_p = kl_p + kl_von_mises(qpm, qpk, ppm, ppk).clamp(min=fb[1])
        kl_t = kl_t + kl_log_normal(qtm, qts, ptm, pts).clamp(min=fb[2])
        zf.append(model.z_features(meter, phi, lt)); mp, pp, lp = meter, phi, lt
        s_phi.append(phi.detach()); s_lt.append(lt.detach())
        q_pm.append(qpm); q_pk.append(qpk); q_tm.append(qtm); q_ts.append(qts)
    logits = torch.stack([model.decode(zf[t], pc[:, t]) for t in range(T)], 1)
    pw = torch.tensor(pos_w, device=h.device) if pos_w != 1.0 else None
    recon = F.binary_cross_entropy_with_logits(logits, b, reduction="none", pos_weight=pw).sum(1)
    loss = (recon + beta * (kl_m + kl_p + kl_t)).mean()
    info = {"recon": float(recon.mean()), "kl_m": float(kl_m.mean()),
            "kl_p": float(kl_p.mean()), "kl_t": float(kl_t.mean()), "os": 0.0}
    if overshoot >= 2:
        # vectorized over distance d: for each d, all valid targets t in [d,T) at once.
        Sphi = torch.stack(s_phi, 1); Slt = torch.stack(s_lt, 1)            # [B,T] detached seeds
        Qpm = torch.stack(q_pm, 1).detach(); Qpk = torch.stack(q_pk, 1).detach()
        Qtm = torch.stack(q_tm, 1).detach(); Qts = torch.stack(q_ts, 1).detach()
        os_sum = h.new_zeros(())
        for d in range(2, min(overshoot, T - 1) + 1):
            ts = torch.arange(d, T, device=h.device); src = ts - d
            ppm_d = (Sphi[:, src] + d * torch.exp(Slt[:, src])) % TWO_PI    # prior mean rolled d steps
            ptm_d = Slt[:, src]                                             # tempo random-walk mean = seed
            ppk_d = F.softplus(model.prior_phase_kappa(pc[:, ts]).squeeze(-1)) + 0.01
            pts_d = F.softplus(model.prior_tempo_sigma(pc[:, ts]).squeeze(-1)) + 1e-3
            klp = kl_von_mises(Qpm[:, ts], Qpk[:, ts], ppm_d, ppk_d).mean(0).clamp(min=os_fn)
            klt = kl_log_normal(Qtm[:, ts], Qts[:, ts], ptm_d, pts_d).mean(0).clamp(min=os_fn)
            os_sum = os_sum + (klp + klt).sum()
        os_term = os_sum / (overshoot - 1)            # avg over distances ~ one-step scale/frame
        loss = loss + os_w * os_term
        info["os"] = float(os_term)
    return loss, info


@torch.no_grad()
def free_run_min(model, h, temp=0.3, clamp=None):
    from faithful.elbo import free_run
    if clamp is None:
        return free_run(model, h, temperature=temp)
    # bar-phase clamped free-run: bound log-tempo at init, stochastic chain, AND the mean chain
    lo, hi = clamp
    B, T, _ = h.shape
    pc = model.encode_prior(h)
    pm_, pphmu, pphk, ptmu, pts = model.unpack(model.prior_init_head(pc.mean(1)))
    meter = gumbel_softmax(pm_, temp); phi = sample_von_mises(pphmu, pphk) % TWO_PI
    lt = (ptmu + pts * torch.randn_like(ptmu)).clamp(lo, hi)
    phi_mu = pphmu % TWO_PI; lt_mu = ptmu.clamp(lo, hi)
    zf = [model.z_features(meter, phi, lt)]; ph = [phi]; phmu = [phi_mu]; lts = [lt]; met = [meter.argmax(-1)]
    mp, pp, lp = meter, phi, lt
    for t in range(1, T):
        pphmu = (pp + torch.exp(lp)) % TWO_PI
        pphk = F.softplus(model.prior_phase_kappa(pc[:, t]).squeeze(-1)) + 0.01
        ptmu = lp; pts = F.softplus(model.prior_tempo_sigma(pc[:, t]).squeeze(-1)) + 1e-3
        phi = sample_von_mises(pphmu, pphk) % TWO_PI
        lt = (ptmu + pts * torch.randn_like(ptmu)).clamp(lo, hi)
        meter = gumbel_softmax(model.meter_prior_logp(mp, phi, pp, pc[:, t]), temp)
        phi_mu = (phi_mu + torch.exp(lt_mu)) % TWO_PI
        zf.append(model.z_features(meter, phi, lt)); ph.append(phi); phmu.append(phi_mu); lts.append(lt); met.append(meter.argmax(-1))
        mp, pp, lp = meter, phi, lt
    logits = torch.stack([model.decode(zf[t], pc[:, t]) for t in range(T)], 1)
    return {"phase": torch.stack(ph, 1), "phase_mu": torch.stack(phmu, 1),
            "log_tempo": torch.stack(lts, 1), "meter": torch.stack(met, 1),
            "decoder_prob": torch.sigmoid(logits)}


def sens(out, inp, r):
    s = (out.reshape(-1) * r[:out.numel()]).sum()
    g, = torch.autograd.grad(s, inp, retain_graph=True, allow_unused=True)
    if g is None:            # input not in graph (e.g. latent-only decoder ignores h) -> 0
        return 0.0
    return g.norm().item() / math.sqrt(inp.numel())


def probe(model, dev, root, keys, clamp=None):
    songs = list(iter_val_songs(root, keys, max_per_dataset=4))
    logmel = LogMel().to(dev)
    db, bo, bpms, tbpms = [], [], [], []
    for key, audio, beats, downs, meta in songs:
        T = min(len(beats), 1000)
        ref = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
        dt = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) < 4:
            continue
        m = 4
        if len(dt) >= 2:
            bpb = np.median([np.sum((ref >= dt[i]) & (ref < dt[i + 1])) for i in range(len(dt) - 1)])
            m = max(2, min(int(round(bpb)) if bpb > 0 else 4, 4))
        h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
        pm = free_run_min(model, h, clamp=clamp)["phase_mu"][0, :T].cpu().numpy()
        db.append(f_measure(dt, downbeats_from_barphase(pm, FPS)))
        bo.append(f_measure(ref, beats_from_barphase(pm, m, FPS)))
        d = np.diff(pm); d = d[d > 0]
        br = float(np.median(d)) if len(d) else 0.0
        bpms.append(60.0 * FPS * m * br / TWO_PI); tbpms.append(60.0 / np.median(np.diff(ref)))
    bpms = np.array(bpms); tbpms = np.array(tbpms)
    acc1 = float(np.mean(np.abs(bpms - tbpms) / tbpms < 0.04)) if len(bpms) else 0.0
    # dependency sensitivities (one song, few timesteps)
    torch.manual_seed(0)
    K = model.K; rdec = torch.randn(1, device=dev); rpar = torch.randn(model.param_dim, device=dev)
    key, audio, beats, downs, meta = songs[0]
    T = min(len(beats), 800); h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
    b = beats[:T].to(dev).unsqueeze(0).float()
    pc = model.encode_prior(h); qc = model.encode_posterior(h, b)
    dz, dh, pz, pctx = [], [], [], []
    for t in (T // 4, T // 2, 3 * T // 4):
        cos = torch.cos(torch.tensor([[1.3]], device=dev)).requires_grad_(True)
        sin = torch.sin(torch.tensor([[1.3]], device=dev)).requires_grad_(True)
        ltv = torch.tensor([[-2.0]], device=dev).requires_grad_(True)
        me = torch.zeros(1, K, device=dev); me[0, min(3, K - 1)] = 1.0; me = me.requires_grad_(True)
        zfeat = torch.cat([cos, sin, ltv, me], -1)
        pcc = pc[:, t].detach().clone().requires_grad_(True)
        bh = model.decode(zfeat, pcc).reshape(-1)
        dz.append(sens(bh, zfeat, rdec)); dh.append(sens(bh, pcc, rdec))
        ctxp = qc[:, t].detach().clone().requires_grad_(True)
        zprev = zfeat.detach().clone().requires_grad_(True)
        qp = model.post_head(torch.cat([ctxp, zprev], -1))
        pz.append(sens(qp, zprev, rpar)); pctx.append(sens(qp, ctxp, rpar))
    return {"downbeat_F": float(np.nanmean(db)), "beat_F_oracle": float(np.nanmean(bo)),
            "tempo_Acc1": acc1, "bpm_mean": float(bpms.mean()), "bpm_std": float(bpms.std()),
            "dec_dz": float(np.mean(dz)), "dec_dh": float(np.mean(dh)),
            "post_dzprev": float(np.mean(pz)), "post_dctx": float(np.mean(pctx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--datasets", default="ballroom,beatles,hains,rwc_popular")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent_only", action="store_true")
    ap.add_argument("--kl_anneal_frac", type=float, default=0.0)   # ramp beta 0->1 over this frac of steps
    ap.add_argument("--fb_m", type=float, default=0.0)
    ap.add_argument("--fb_p", type=float, default=0.0)
    ap.add_argument("--fb_t", type=float, default=0.0)
    ap.add_argument("--tempo_clamp", action="store_true")          # clamp log-tempo to [-5.5,-0.5]
    ap.add_argument("--bce_pos_weight", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0, help="fixed KL weight (1.0=strict ELBO)")
    ap.add_argument("--overshoot", type=int, default=1, help="latent-overshoot horizon D (1=strict)")
    ap.add_argument("--os_free_nats", type=float, default=0.0, help="free-nats floor per overshoot term")
    ap.add_argument("--os_weight", type=float, default=1.0, help="weight on the overshoot loss")
    ap.add_argument("--selftest", action="store_true", help="run D=1==strict known-answer test and exit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(42)
    dev = "cuda"
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tclamp = (-4.7, -1.7) if args.tempo_clamp else None   # bar-rate range: ~0.4-8 s/bar (clamp now in train AND inference)
    fb = (args.fb_m, args.fb_p, args.fb_t)
    print(f"[{args.cell}] start latent_only={args.latent_only} kl_anneal={args.kl_anneal_frac} "
          f"fb={fb} tempo_clamp={bool(tclamp)} pos_w={args.bce_pos_weight}", flush=True)
    logmel = LogMel().to(dev)
    model = BarPointerVAE(h_dim=N_MELS, hidden=64, num_meters=4, latent_only=args.latent_only).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = build_train_loader(args.data_root, keys, args.frames, args.batch_size,
                                examples_per_epoch=1000, num_workers=4, seed=42)
    if args.selftest:
        # KNOWN-ANSWER TEST: overshoot=1 must reproduce strict_elbo EXACTLY; D>=2 adds a >0 term.
        di = iter(loader); audio, beats, _ = next(di)
        h = logmel(audio.to(dev))[:, :args.frames]; bt = beats[:, :args.frames].to(dev)
        Tm = min(h.shape[1], bt.shape[1]); h, bt = h[:, :Tm], bt[:, :Tm]
        from faithful.elbo import strict_elbo
        torch.manual_seed(123); ls, _ = strict_elbo(model, h, bt, temperature=1.0)
        torch.manual_seed(123); l1, i1 = rollout(model, h, bt, 1.0, 1.0, (0., 0., 0.), None, 1.0, overshoot=1)
        torch.manual_seed(123); ld, idd = rollout(model, h, bt, 1.0, 1.0, (0., 0., 0.), None, 1.0,
                                                  overshoot=4, os_fn=0.0, os_w=1.0)
        t1 = abs(float(ls) - float(l1)) < 1e-3
        t2 = idd["os"] > 0.0 and abs(float(ld) - float(l1) - idd["os"]) < 1e-2
        print(f"[selftest] strict={float(ls):.4f} D1={float(l1):.4f} D4={float(ld):.4f} os={idd['os']:.4f}")
        print(f"[selftest] T1 D1==strict: {t1}  T2 D4==D1+os & os>0: {t2}  -> {'PASS' if t1 and t2 else 'FAIL'}")
        return
    di = iter(loader); step = 0; t0 = time.time()
    while step < args.steps:
        try:
            audio, beats, _ = next(di)
        except StopIteration:
            di = iter(loader); audio, beats, _ = next(di)
        step += 1
        temp = 1.0 + (0.3 - 1.0) * min(step / args.steps, 1.0)
        beta = args.beta if args.kl_anneal_frac <= 0 else min(step / (args.kl_anneal_frac * args.steps), 1.0)
        h = logmel(audio.to(dev))[:, :args.frames]; bt = beats[:, :args.frames].to(dev)
        Tm = min(h.shape[1], bt.shape[1]); h, bt = h[:, :Tm], bt[:, :Tm]
        opt.zero_grad()
        loss, info = rollout(model, h, bt, temp, beta, fb, tclamp, args.bce_pos_weight,
                             overshoot=args.overshoot, os_fn=args.os_free_nats, os_w=args.os_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 50 == 0 or step == 1:
            print(f"[{args.cell}] step {step} L={info['recon']+info['kl_m']+info['kl_p']+info['kl_t']:.1f} "
                  f"recon={info['recon']:.1f} kl(m/p/t)={info['kl_m']:.3f}/{info['kl_p']:.3f}/{info['kl_t']:.3f} "
                  f"os={info['os']:.3f} beta={beta:.2f} {step/(time.time()-t0):.2f}it/s", flush=True)
    torch.save({"model": model.state_dict(), "cell": args.cell, "args": vars(args)}, out / "final.pt")
    model.eval()
    res = {"cell": args.cell, "final_train": info,
           **{f"flag_{k}": v for k, v in dict(latent_only=args.latent_only, kl_anneal=args.kl_anneal_frac,
              fb=fb, tempo_clamp=bool(tclamp), pos_w=args.bce_pos_weight).items()}}
    try:
        res.update(probe(model, dev, args.data_root, keys, clamp=tclamp))
    except Exception as e:
        res["probe_error"] = repr(e)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"[{args.cell}] DONE -> {out/'result.json'}", flush=True)


if __name__ == "__main__":
    main()
