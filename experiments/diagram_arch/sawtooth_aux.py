"""SAWTOOTH-PHASE AUXILIARY LOSS (keep the generative ELBO). Tests the user's idea + ref [42]
(Oyama 2021) / Chen & Su 2022: ground the posterior BAR-PHASE phi to a sawtooth target (0 at
downbeat -> 2pi at next downbeat) via a circular (1-cos) auxiliary loss, ON TOP of the full
bar-pointer ELBO. The sawtooth is DENSE + rate-informative (its slope is the tempo), attacking the
rate-blind-Bernoulli failure (link 1) and forbidding the oscillation cheat (link 4) WITHOUT
positional encoding or an explicit integrator. If it works, the prior coupling
ppm = phiprev + exp(ltprev) back-fills tempo lt=phidot from the grounded phase.
Deploy = posterior phi read-out (the CVAE). Reports geometric beat/db F, phi-revs, tempo, LEAK test.
"""
import sys, math, importlib.util, argparse, random
import numpy as np
import torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, rollout, load_pool, sample_batch = da.BPVAE, da.rollout, da.load_pool, da.sample_batch
phase_beats, phase_downbeats, fmeas, peaks = da.phase_beats, da.phase_downbeats, da.fmeas, da.peaks
DEV = da.DEV; FPS = 86.1328125; M = 4; TWO_PI = 2 * math.pi


def gt_barphase(dbvec, T):
    phi = np.zeros(T, dtype=np.float32); mask = np.zeros(T, dtype=np.float32)
    dbf = np.where(dbvec > 0.5)[0]
    for k in range(len(dbf) - 1):
        a, c = dbf[k], dbf[k + 1]
        phi[a:c] = np.linspace(0.0, TWO_PI, c - a, endpoint=False); mask[a:c] = 1.0
    return phi, mask


def gt_batch(db):
    B, T = db.shape; P = np.zeros((B, T), np.float32); Mk = np.zeros((B, T), np.float32)
    dbn = db.detach().cpu().numpy()
    for j in range(B):
        P[j], Mk[j] = gt_barphase(dbn[j], T)
    return torch.from_numpy(P).to(db.device), torch.from_numpy(Mk).to(db.device)


def elbo_aux(model, h, b, db, temp, lam, pw_b=8.0, pw_db=20.0, fb=0.1, b_drop=0.5):
    B, T, _ = h.shape
    keep = (torch.rand(B, 1, device=h.device) >= b_drop).float()
    (klm, klp, klt), phase_mu, logits = rollout(model, h, b * keep, db * keep, temp, sample=True, compute_kl=True)
    pw = torch.tensor([pw_b, pw_db], device=h.device)
    recon = F.binary_cross_entropy_with_logits(logits, torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
    klm = klm.clamp(min=fb * T); klp = klp.clamp(min=fb * T); klt = klt.clamp(min=fb * T)
    phi_gt, mask = gt_batch(db)
    perframe = (1.0 - torch.cos(phase_mu - phi_gt)) * mask
    L_phase = perframe.sum(1) / mask.sum(1).clamp(min=1.0)
    loss = (recon + klm + klp + klt + lam * T * L_phase).mean()
    return loss, {"recon": float(recon.mean()), "klt": float(klt.mean()), "Lphase": float(L_phase.mean())}


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, revs, bpm = [], [], [], []; n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        _, phase_mu, _ = rollout(model, h_in, z, z, sample=False, compute_kl=False)
        phi = phase_mu[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        d = np.diff(phi); adv = np.where(d < -math.pi, d + TWO_PI, d)
        revs.append(float(adv.sum() / TWO_PI)); a = adv[adv > 1e-4]
        bpm.append(M * float(np.median(a)) / TWO_PI * FPS * 60 if len(a) else 0.0)
    model.train(); f = lambda x: float(np.nanmean(x)) if x else float("nan")
    return f(gb), f(gd), f(revs), f(bpm)


def run(tag, lam, train, val, steps, nb):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = BPVAE(h_dim=512, hidden=64).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / steps, 1.0)
        h, b, db = sample_batch(train, 256, 16)
        loss, info = elbo_aux(model, h, b, db, temp, lam)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 200 == 0 or step == steps:
            gb, gd, rv, bp = evaluate(model, val, "real")
            print(f"  [{tag}] step {step:4d} | recon {info['recon']:.0f} Lph {info['Lphase']:.3f} klt {info['klt']:.1f} "
                  f"| GEOM beat {gb:.3f} db {gd:.3f} | revs {rv:.1f}/{nb} tempo {bp:.0f}", flush=True)
    gb, gd, rv, bp = evaluate(model, val, "real")
    gbs, gds, rvs, _ = evaluate(model, val, "shuffle"); gbz, gdz, rvz, _ = evaluate(model, val, "zero")
    print(f"  [{tag}] FINAL real beat {gb:.3f} db {gd:.3f} revs {rv:.1f}/{nb} tempo {bp:.0f} | "
          f"shuf beat {gbs:.3f} db {gds:.3f} | zero beat {gbz:.3f} db {gdz:.3f}", flush=True)
    return (gb, gbs, gbz), (gd, gds, gdz), rv, bp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    ap.add_argument("--lams", default="0,0.5,2.0")
    a = ap.parse_args()
    train = load_pool("cache/acts/bt_train_rich", a.n_train, seed=1)
    val = load_pool("cache/acts/bt_val_rich", a.n_val, seed=2)
    nb = int(np.mean([(d > 0.5).sum() for _, _, d in val]))
    print(f"SAWTOOTH-AUX | train={len(train)} val={len(val)} | GT #bars~{nb} | deploy=posterior phi read-out\n", flush=True)
    res = {}
    for lam in [float(x) for x in a.lams.split(",")]:
        print(f"=== lambda={lam} (0 = plain ELBO baseline) ===", flush=True)
        res[lam] = run(f"lam{lam}", lam, train, val, a.steps, nb)
    print("\n==== VERDICT ====")
    for lam, ((gb, gbs, gbz), (gd, gds, gdz), rv, bp) in res.items():
        audio = "AUDIO-LOCKED" if (gb > 0.5 and gbs < gb - 0.2 and gbz < gb - 0.2) else "weak/leak"
        rot = "ROTATES" if abs(rv - nb) < 0.4 * nb else "no-rotate"
        print(f"  lam={lam}: beat {gb:.3f}(shuf {gbs:.3f} zero {gbz:.3f}) db {gd:.3f} | revs {rv:.1f}/{nb} tempo {bp:.0f} -> {audio}, {rot}")
    print("  WIN = high beat/db + leak collapse + revs~#bars + tempo~real -> sawtooth grounds phi AND phidot")
    print("DONE")


if __name__ == "__main__":
    main()
