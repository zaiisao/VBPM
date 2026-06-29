"""IDEA #2: LOCAL SELF-SUPERVISED TEMPO. phi = integral of phi-dot (soft-bounded), geometric emission,
PLUS a direct local target for phi-dot: the autocorrelation tempo of the frontend's own beat activation
(act2[:,0]) -- a local, well-conditioned, label-free signal that bypasses the ill-conditioned integral
gradient. The autocorr target is TRAIN-ONLY (deploy infers phi-dot from h). Grounds the RATE directly;
the emission fixes the phase OFFSET. Reports phi-revs/tempo/leak.
"""
import sys, glob, math, importlib.util, argparse, random
import numpy as np, torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
fv = importlib.util.spec_from_file_location("fv", f"{ROOT}/experiments/kvae_barpointer/faithful_v2.py")
v2 = importlib.util.module_from_spec(fv); fv.loader.exec_module(v2)
da = v2.da; BPVAE = da.BPVAE; soft_lt, LT_MIN, LT_MAX = v2.soft_lt, v2.LT_MIN, v2.LT_MAX
fmeas, phase_beats, phase_downbeats = da.fmeas, da.phase_beats, da.phase_downbeats
gumbel_softmax, kl_von_mises, kl_log_normal, kl_categorical = da.gumbel_softmax, da.kl_von_mises, da.kl_log_normal, da.kl_categorical
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4; KAPPA = 8.0


def autocorr_logadv(beat_act, min_bpm=55, max_bpm=200):
    """log bar-phase advance/frame from the autocorr peak of the frontend beat activation (self-supervised)."""
    a = np.asarray(beat_act, float); a = a - a.mean()
    if a.std() < 1e-6: return math.log(TWO_PI * 120 / 60 / M / FPS)
    ac = np.correlate(a, a, mode="full")[len(a) - 1:]
    lo = int(60 * FPS / max_bpm); hi = int(60 * FPS / min_bpm)
    seg = ac[lo:hi + 1]
    if len(seg) < 2: return math.log(TWO_PI * 120 / 60 / M / FPS)
    lag = lo + int(np.argmax(seg))                            # frames per beat
    return float(np.clip(math.log(TWO_PI / (M * lag)), LT_MIN, LT_MAX))


def load6(cd, n, seed):
    fs = sorted(glob.glob(f"{cd}/*.pt")); random.Random(seed).shuffle(fs); out = []
    for f in fs[:n]:
        d = torch.load(f, map_location="cpu"); hh = d["activations"].float()
        if hh.shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        out.append((hh, d["beat_targets"].float(), d["downbeat_targets"].float(), d["act2"].float()))
    return out


def batch6(songs, frames, bs):
    H, B_, D_, tgt = [], [], [], []
    while len(H) < bs:
        hh, b, db, a2 = random.choice(songs)
        if hh.shape[0] <= frames: continue
        s0 = random.randint(0, hh.shape[0] - frames); bb = b[s0:s0 + frames]
        if bb.sum() < 2: continue
        H.append(hh[s0:s0 + frames]); B_.append(bb); D_.append(db[s0:s0 + frames])
        tgt.append(autocorr_logadv(a2[s0:s0 + frames, 0].numpy()))
    Ht = torch.stack(H).to(DEV); Bt = torch.stack(B_).to(DEV)
    Dt = torch.stack(D_).to(DEV); Tt = torch.tensor(tgt, device=DEV)
    return Ht, Bt, Dt, Tt


def rollout_lt(model, h, b_in, db_in, temp=0.5, sample=True, compute_kl=True):
    B, T, _ = h.shape
    pc = model.enc_post(h, b_in, db_in); pr = model.enc_prior(h) if compute_kl else None
    klm = klp = klt = (h.new_zeros(B) if compute_kl else None)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, 0], z0], -1)))
    m = gumbel_softmax(qm, temp) if sample else F.softmax(qm, -1); lt = soft_lt(qtm)
    phi = (da.sample_von_mises(qpm % TWO_PI, qpk) % TWO_PI) if sample else (qpm % TWO_PI)
    if compute_kl:
        pm, ppm, ppk, ptm, pts = model.unpack(model.prior_init(pr.mean(1)))
        klm = klm + kl_categorical(torch.log_softmax(qm, -1), torch.log_softmax(pm, -1))
        klp = klp + kl_von_mises(phi, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
    phis = [phi]; lts = [lt]; mprev, phiprev, ltprev = m, phi, lt
    for t in range(1, T):
        zp = model.zfeat(mprev, phiprev, ltprev)
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([pc[:, t], zp], -1)))
        m = gumbel_softmax(qm, temp) if sample else F.softmax(qm, -1); lt = soft_lt(qtm)
        phi_mean = (phiprev + torch.exp(lt)) % TWO_PI
        phi = (da.sample_von_mises(phi_mean, qpk) % TWO_PI) if sample else phi_mean
        if compute_kl:
            ppm = (phiprev + torch.exp(ltprev)) % TWO_PI
            ppk = F.softplus(model.prior_pk(pr[:, t]).squeeze(-1)) + 0.01
            ptm = ltprev; pts = F.softplus(model.prior_ts(pr[:, t]).squeeze(-1)) + 1e-3
            klm = klm + kl_categorical(torch.log_softmax(qm, -1), model.meter_logp(mprev, phi, phiprev, pr[:, t]))
            klp = klp + kl_von_mises(phi_mean, qpk, ppm, ppk); klt = klt + kl_log_normal(qtm, qts, ptm, pts)
        phis.append(phi); lts.append(lt); mprev, phiprev, ltprev = m, phi_mean, lt
    return ((klm, klp, klt) if compute_kl else None), torch.stack(phis, 1), torch.stack(lts, 1)  # lts [B,T]


def geom_logits(phi):
    return torch.stack([KAPPA * torch.cos(M * phi), KAPPA * torch.cos(phi)], -1)


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, revs, bpm = [], [], [], []; n = len(val)
    for i, (hh, b, db, a2) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        _, phis, _ = rollout_lt(model, h_in, z, z, sample=False, compute_kl=False)
        phi = phis[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); adv = np.where(dphi < -math.pi, dphi + TWO_PI, dphi)
        revs.append(float(np.sum(adv) / TWO_PI)); a3 = adv[adv > 1e-4]
        bpm.append(M * float(np.median(a3)) / TWO_PI * FPS * 60 if len(a3) else 0.0)
    model.train(); mn = lambda x: float(np.nanmean(x)) if x else float("nan")
    return mn(gb), mn(gd), mn(revs), mn(bpm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lam", type=float, default=5.0); ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load6("cache/acts/bt_train_rich", a.n_train, 1); val = load6("cache/acts/bt_val_rich", a.n_val, 2)
    nb = int(np.mean([(d > 0.5).sum() for _, _, d, _ in val]))
    # sanity: how good is the autocorr target itself vs GT?
    err = []
    for hh, b, db, a2 in val[:20]:
        bf = np.where(b.numpy() > 0.5)[0]
        if len(bf) > 2:
            gt = 60 * FPS / np.median(np.diff(bf)); pred = M * math.exp(autocorr_logadv(a2[:, 0].numpy())) / TWO_PI * FPS * 60
            err.append(pred / gt)
    print(f"FAITHFUL v6 (LOCAL SELF-SUP TEMPO, lam={a.lam}) | train={len(train)} val={len(val)} | GT #bars~{nb}", flush=True)
    print(f"  autocorr-target sanity: pred/GT ratio median {np.median(err):.2f} (1.0=perfect, 0.5/2.0=octave)", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = torch.tensor([8.0, 20.0], device=DEV)
    for step in range(1, a.steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        H, Bt, Dt, Tt = batch6(train, 256, 16)
        keep = (torch.rand(16, 1, device=DEV) >= 0.5).float()
        (klm, klp, klt), phis, lts = rollout_lt(model, H, Bt * keep, Dt * keep, temp, sample=True, compute_kl=True)
        recon = F.binary_cross_entropy_with_logits(geom_logits(phis), torch.stack([Bt, Dt], -1), pos_weight=pw, reduction="none").sum((1, 2))
        tempo_mse = ((lts - Tt.unsqueeze(1)) ** 2).mean(1)            # [B] local tempo grounding
        loss = (recon + klm.clamp(min=25.6) + klp.clamp(min=25.6) + klt.clamp(min=25.6) + a.lam * tempo_mse).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            gb, gd, rv, bp = evaluate(model, val, "real")
            print(f"  step {step:4d} | recon {float(recon.mean()):.1f} tmse {float(tempo_mse.mean()):.3f} | GEOM beat {gb:.3f} db {gd:.3f} | phi-revs {rv:.1f}/{nb} | tempo {bp:.0f}BPM", flush=True)
    gb, gd, rv, bp = evaluate(model, val, "real"); gbs, _, _, _ = evaluate(model, val, "shuffle"); gbz, _, _, _ = evaluate(model, val, "zero")
    print(f"\n--- FINAL (local self-sup tempo) ---")
    print(f"  real {gb:.3f} db {gd:.3f} revs {rv:.1f}/{nb} tempo {bp:.0f}BPM | shuf {gbs:.3f} zero {gbz:.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
