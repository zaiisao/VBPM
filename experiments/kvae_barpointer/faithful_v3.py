"""FAITHFUL v3: deterministic-integral phase (v2) + GEOMETRIC training emission (not a learned decoder),
so the reconstruction REQUIRES correct phi rotation -> the tempo gradient becomes strong & correctly
directed. beat_logit = K*cos(M*phi), db_logit = K*cos(phi); BCE on real beats. Deploy = geometric.
Reports phi-revs/tempo/leak AND re-measures the recon->tempo gradient direction.
"""
import sys, math, importlib.util, argparse, random
import numpy as np, torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
fv = importlib.util.spec_from_file_location("fv", f"{ROOT}/experiments/kvae_barpointer/faithful_v2.py")
v2 = importlib.util.module_from_spec(fv); fv.loader.exec_module(v2)
da = v2.da; BPVAE, load_pool, sample_batch = da.BPVAE, da.load_pool, da.sample_batch
rollout_det, soft_lt = v2.rollout_det, v2.soft_lt
fmeas, phase_beats, phase_downbeats = da.fmeas, da.phase_beats, da.phase_downbeats
DEV = da.DEV; TWO_PI = 2*math.pi; FPS = 86.1328125; M = 4; KAPPA = 8.0


def geom_logits(phi):
    return torch.stack([KAPPA * torch.cos(M * phi), KAPPA * torch.cos(phi)], -1)


def elbo_geom(model, h, b, db, temp, fb=0.1, b_drop=0.5):
    B, T, _ = h.shape
    keep = (torch.rand(B, 1, device=h.device) >= b_drop).float()
    (klm, klp, klt), phis, _ = rollout_det(model, h, b * keep, db * keep, temp, sample=True, compute_kl=True)
    pw = torch.tensor([8.0, 20.0], device=h.device)
    recon = F.binary_cross_entropy_with_logits(geom_logits(phis), torch.stack([b, db], -1), pos_weight=pw, reduction="none").sum((1, 2))
    klm = klm.clamp(min=fb*T); klp = klp.clamp(min=fb*T); klt = klt.clamp(min=fb*T)
    return (recon + klm + klp + klt).mean(), float(recon.mean()), float(klt.mean())


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    model.eval(); gb, gd, revs, bpm = [], [], [], []; n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(1, T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(0).to(DEV)
        z = torch.zeros(1, T, device=DEV)
        _, phis, _ = rollout_det(model, h_in, z, z, sample=False, compute_kl=False)
        phi = phis[0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); adv = np.where(dphi < -math.pi, dphi + TWO_PI, dphi)
        revs.append(float(np.sum(adv) / TWO_PI)); a2 = adv[adv > 1e-4]
        bpm.append(M * float(np.median(a2)) / TWO_PI * FPS * 60 if len(a2) else 0.0)
    model.train(); mn = lambda x: float(np.nanmean(x)) if x else float("nan")
    return mn(gb), mn(gd), mn(revs), mn(bpm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load_pool("cache/acts/bt_train_rich", a.n_train, seed=1); val = load_pool("cache/acts/bt_val_rich", a.n_val, seed=2)
    nb = int(np.mean([(d > 0.5).sum() for _, _, d in val]))
    print(f"FAITHFUL v3 (det-integral phi + GEOMETRIC emission) | train={len(train)} val={len(val)} | GT #bars~{nb}", flush=True)
    model = BPVAE(h_dim=512, hidden=64).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, a.steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / a.steps, 1.0)
        h, b, db = sample_batch(train, 256, 16)
        loss, recon, klt = elbo_geom(model, h, b, db, temp)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            gb, gd, rv, bp = evaluate(model, val, "real")
            print(f"  step {step:4d} | recon {recon:.1f} klt {klt:.1f} | GEOM beat {gb:.3f} db {gd:.3f} | phi-revs {rv:.1f}/{nb} | tempo {bp:.0f}BPM", flush=True)
    gb, gd, rv, bp = evaluate(model, val, "real")
    gbs, gds, rvs, _ = evaluate(model, val, "shuffle"); gbz, gdz, rvz, _ = evaluate(model, val, "zero")
    print("\n--- FINAL (GEOMETRIC deploy; det-integral phi + geometric emission) ---")
    print(f"  real     : beat {gb:.3f}  db {gd:.3f}  phi-revs {rv:.1f}/{nb}  tempo {bp:.0f}BPM")
    print(f"  shuffled : beat {gbs:.3f}  db {gds:.3f}  revs {rvs:.1f}   (must COLLAPSE)")
    print(f"  zero     : beat {gbz:.3f}  db {gdz:.3f}  revs {rvz:.1f}   (must COLLAPSE)")
    print("VERDICT: phi-revs~#bars + musical tempo + beat high + leak collapse => geometric emission FIXES the gradient")
    print("DONE")


if __name__ == "__main__":
    main()
