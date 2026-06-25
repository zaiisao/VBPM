"""He-2019 aggressive-encoder training, faithfully ported from third_party/vae-lagging-encoder
(text.py:351-464). FAITHFUL core: two optimizers (enc=inference post_*, dec=generator rest),
inner loop updates ONLY the encoder on fresh minibatches until the ELBO (burn loss) stops
decreasing (checked every `plateau_every`, max `inner_cap`), then ONE generator step (+encoder
if not aggressive). ADAPTED: their MI-based aggressive-exit uses a Gaussian calc_mi that does
NOT apply to our von Mises/categorical latents, so the exit uses a rate(total-KL)-plateau proxy
(documented deviation). Includes --test: known-answer integration tests, run before training."""
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
import numpy as np, torch
import torch.nn.functional as F
from faithful.model import BarPointerVAE
from faithful.elbo import strict_elbo, free_run
from faithful.data import FPS, N_MELS, LogMel, build_train_loader, iter_val_songs
from faithful.distributions import TWO_PI
from faithful.evaluate import beats_from_barphase, f_measure

ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = "ballroom,beatles,hains,rwc_popular"


def split_params(model):
    enc, dec = [], []
    for n, p in model.named_parameters():
        (enc if n.startswith(("post_gru", "post_ctx", "post_head")) else dec).append((n, p))
    return enc, dec


def integration_tests(model, h, b):
    print("==== KNOWN-ANSWER INTEGRATION TESTS ====")
    enc, dec = split_params(model)
    encn = {n for n, _ in enc}; decn = {n for n, _ in dec}; alln = {n for n, _ in model.named_parameters()}
    # T1: partition disjoint + complete
    t1 = (encn & decn == set()) and (encn | decn == alln) and len(encn) > 0 and len(decn) > 0
    print(f"[T1] enc/dec partition disjoint+complete (enc={len(encn)},dec={len(decn)}): {t1}  expected=True")
    print(f"     enc params: {sorted(encn)}")
    encp = [p for _, p in enc]; decp = [p for _, p in dec]
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
    def deltas():
        de = max((p - snap[n]).abs().max().item() for n, p in model.named_parameters() if n in encn)
        dd = max((p - snap[n]).abs().max().item() for n, p in model.named_parameters() if n in decn)
        return de, dd
    def restore():
        with torch.no_grad():
            for n, p in model.named_parameters(): p.copy_(snap[n])
    # T2: encoder-only step freezes the generator
    oe = torch.optim.SGD(encp, lr=0.1); od = torch.optim.SGD(decp, lr=0.1)
    oe.zero_grad(); od.zero_grad()
    torch.manual_seed(0); loss, _ = strict_elbo(model, h, b); loss.backward(); oe.step()
    de, dd = deltas(); t2 = de > 0 and dd == 0.0
    print(f"[T2] enc-only step: Δenc={de:.2e}(>0) Δdec={dd:.2e}(==0): {t2}  expected=True")
    restore()
    # T3: decoder-only step freezes the encoder
    oe = torch.optim.SGD(encp, lr=0.1); od = torch.optim.SGD(decp, lr=0.1)
    oe.zero_grad(); od.zero_grad()
    torch.manual_seed(0); loss, _ = strict_elbo(model, h, b); loss.backward(); od.step()
    de, dd = deltas(); t3 = dd > 0 and de == 0.0
    print(f"[T3] dec-only step: Δenc={de:.2e}(==0) Δdec={dd:.2e}(>0): {t3}  expected=True")
    restore()
    # T4: ELBO seed-reproducible AND loss == recon + sum(KL)
    torch.manual_seed(7); l1, i1 = strict_elbo(model, h, b)
    torch.manual_seed(7); l2, _ = strict_elbo(model, h, b)
    t4a = abs(float(l1) - float(l2)) < 1e-4
    t4b = abs(float(l1) - (i1["recon"] + i1["kl_meter"] + i1["kl_phase"] + i1["kl_tempo"])) < 1e-2
    print(f"[T4] ELBO seed-reproducible:{t4a} | loss==recon+ΣKL:{t4b}  expected=True,True")
    restore()
    ok = all([t1, t2, t3, t4a, t4b])
    print(f"==== INTEGRATION TESTS {'PASS' if ok else 'FAIL'} ====")
    return ok


@torch.no_grad()
def posterior_mean_phase(model, h, b):
    B, T, _ = h.shape; qc = model.encode_posterior(h, b)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1); traj = [phi]
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, t], model.z_features(meter, phi, lt)], -1)))
        phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1); traj.append(phi)
    return torch.stack(traj, 1)[0].cpu().numpy()


def probe(model, dev):
    model.eval(); logmel = LogMel().to(dev)
    pf, ff, sg, kp = [], [], [], []
    for key, audio, beats, downs, meta in iter_val_songs(ROOT, DS.split(","), max_per_dataset=3):
        T = min(len(beats), 1000); ref = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
        dt = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) < 4: continue
        m = 4
        if len(dt) >= 2:
            x = np.median([np.sum((ref >= dt[i]) & (ref < dt[i+1])) for i in range(len(dt)-1)]); m = max(2, min(int(round(x)) if x>0 else 4, 4))
        h = logmel(audio.to(dev).unsqueeze(0))[:, :T]; b = beats[:T].to(dev).unsqueeze(0).float()
        pf.append(f_measure(ref, beats_from_barphase(posterior_mean_phase(model, h, b), m, FPS)))
        ff.append(f_measure(ref, beats_from_barphase(free_run(model, h, temperature=0.3)["phase_mu"][0, :T].cpu().numpy(), m, FPS)))
        with torch.no_grad():
            pc = model.encode_prior(h)
            sg.append((F.softplus(model.prior_tempo_sigma(pc).squeeze(-1)) + 1e-3).mean().item())
            kp.append((F.softplus(model.prior_phase_kappa(pc).squeeze(-1)) + 0.01).mean().item())
    model.train()
    return {"posterior_beatF": float(np.nanmean(pf)), "freerun_beatF": float(np.nanmean(ff)),
            "prior_sigma": float(np.nanmean(sg)), "prior_kappa": float(np.nanmean(kp))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_only", action="store_true")
    ap.add_argument("--frames", type=int, default=96); ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--inner_cap", type=int, default=8); ap.add_argument("--plateau_every", type=int, default=4)
    ap.add_argument("--aggressive", type=int, default=1)
    ap.add_argument("--test", action="store_true"); ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"; torch.manual_seed(42)
    model = BarPointerVAE(h_dim=N_MELS, hidden=64, num_meters=4, latent_only=args.latent_only).to(dev)
    loader = build_train_loader(ROOT, DS.split(","), args.frames, args.batch_size, examples_per_epoch=1000, num_workers=4, seed=42)
    logmel = LogMel().to(dev)
    di = iter(loader)
    def nb():
        nonlocal di
        try: a, b, _ = next(di)
        except StopIteration: di = iter(loader); a, b, _ = next(di)
        h = logmel(a.to(dev))[:, :args.frames]; bb = b[:, :args.frames].to(dev)
        T = min(h.shape[1], bb.shape[1]); return h[:, :T], bb[:, :T]

    # ---- integration tests first ----
    h0, b0 = nb()
    ok = integration_tests(model, h0, b0)
    if args.test:
        return
    if not ok:
        print("ABORT: integration tests failed"); return

    enc = [p for _, p in split_params(model)[0]]; dec = [p for _, p in split_params(model)[1]]
    oe = torch.optim.AdamW(enc, lr=args.lr); od = torch.optim.AdamW(dec, lr=args.lr)
    aggressive = bool(args.aggressive); pre_rate = -1.0; t0 = time.time()
    print(f"[he] latent_only={args.latent_only} aggressive={aggressive} (faithful inner loop; rate-plateau exit proxy)")
    for step in range(1, args.steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / args.steps, 1.0)
        if aggressive:                                    # faithful inner loop (encoder only)
            burn_pre, burn_cur, cnt = 1e9, 0.0, 0
            for sub in range(args.inner_cap):
                he, be = nb(); oe.zero_grad(); od.zero_grad()
                loss, info = strict_elbo(model, he, be, temperature=temp)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); oe.step()
                burn_cur += info["loss"]; cnt += 1
                if (sub + 1) % args.plateau_every == 0:
                    avg = burn_cur / cnt
                    if burn_pre - avg < 0: break
                    burn_pre, burn_cur, cnt = avg, 0.0, 0
        h, b = nb(); oe.zero_grad(); od.zero_grad()
        loss, info = strict_elbo(model, h, b, temperature=temp)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not aggressive: oe.step()
        od.step()
        if step % 20 == 0 or step == 1:
            print(f"[he] step {step} L={info['loss']:.1f} recon={info['recon']:.1f} "
                  f"kl(m/p/t)={info['kl_meter']:.3f}/{info['kl_phase']:.3f}/{info['kl_tempo']:.3f} "
                  f"aggr={aggressive} {step/(time.time()-t0):.2f}it/s", flush=True)
        # rate-plateau aggressive-exit proxy (replaces Gaussian MI), checked every 30 steps
        if aggressive and step % 30 == 0:
            rate = info["kl_meter"] + info["kl_phase"] + info["kl_tempo"]
            if rate - pre_rate < 0: aggressive = False; print(f"[he] STOP BURNING (rate plateau) @ {step}", flush=True)
            pre_rate = rate
    ft = {k: float(info[k]) for k in ("loss", "recon", "kl_meter", "kl_phase", "kl_tempo")}
    res = {"latent_only": args.latent_only, "final_train": ft, **probe(model, dev)}
    if args.out:                          # SAVE FIRST (so a print bug can't lose the run)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, Path(args.out) / "final.pt")
        (Path(args.out) / "result.json").write_text(json.dumps(res, indent=1))
    print("[he] RESULT:", json.dumps(res), flush=True)
    print("[he] DONE", flush=True)


if __name__ == "__main__":
    main()
