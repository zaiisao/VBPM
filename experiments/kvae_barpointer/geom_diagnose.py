"""DIAGNOSTIC 1 (decisive): WHY does the geometric read-out fail? Three controlled tests:

  A. CLAMP (read-out ceiling): build the GT bar-phase ramp from GT beats, read beats geometrically.
     If high (~0.95+) the read-out path (phase_beats/phase_downbeats) is SOUND -> failure is upstream.
  B. GT-PHI-SUPERVISED: train the integrate-tempo KVAE with a STRONG loss pulling phi toward the GT
     ramp. If geometric beat-F then jumps and leak collapses -> the problem is INFERENCE/insufficient
     signal (fixable), NOT a fundamental degeneracy. If phi STILL won't track GT even when directly
     supervised -> the latent/filter structurally can't carry a clean rotating phi (deeper problem).
  C. (reference) the same model with NO supervision (the unsupervised geometric loss) -> reproduces the
     constant-grid failure for contrast.

Verdict logic:
  clamp high + sup works  -> read-out sound, failure was the (unsupervised) inference/loss -> fixable
  clamp high + sup fails  -> filter/latent can't represent rotating phi -> structural
  clamp low               -> the read-out itself is broken (our earlier diagnosis was wrong)
"""
import sys, math, random, argparse, importlib.util
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
kvae_run = importlib.util.module_from_spec(kr); kr.loader.exec_module(kvae_run)
KVAEBarPointer, kvae_elbo, load, batch = kvae_run.KVAEBarPointer, kvae_run.kvae_elbo, kvae_run.load, kvae_run.batch
da = kvae_run.da; phase_beats, phase_downbeats, fmeas = da.phase_beats, da.phase_downbeats, da.fmeas
from kvae.sample_control import SampleControl
DEV = kvae_run.DEV; TWO_PI = 2 * math.pi; FPS = 86.1328125; M = 4; KAPPA = 8.0
LT_MIN = math.log(TWO_PI * 40 / 60 / M / FPS); LT_MAX = math.log(TWO_PI * 250 / 60 / M / FPS)


def gt_barphase(beat_fr, m, T):
    phi = np.zeros(T)
    if len(beat_fr) < 2: return phi
    vals = np.arange(len(beat_fr)) * (TWO_PI / m)
    for k in range(len(beat_fr) - 1):
        a, b = beat_fr[k], beat_fr[k + 1]; phi[a:b] = np.linspace(vals[k], vals[k + 1], b - a, endpoint=False)
    phi[beat_fr[-1]:] = vals[-1]
    return phi % TWO_PI


def integrate_phi(z, tempo_head):
    lt = tempo_head(z).squeeze(-1).clamp(LT_MIN, LT_MAX)
    omega = torch.exp(lt)
    return torch.cumsum(omega, dim=0) % TWO_PI, omega


def geom_logits(phi):
    return torch.stack([KAPPA * torch.cos(M * phi), KAPPA * torch.cos(phi)], -1)


# ---------------------------------------------------------------- A: clamp (read-out ceiling)
def clamp_test(val, frames=1600):
    gb, gd = [], []
    for hh, b, db in val:
        T = min(hh.shape[0], b.shape[0], frames)
        bf = np.where(b.numpy()[:T] > 0.5)[0]
        phi = gt_barphase(bf, M, T)
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(gb), m(gd)


@torch.no_grad()
def evaluate(model, tempo_head, val, h_mode="real", frames=1600):
    model.eval(); tempo_head.eval()
    sc = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    gb, gd, revs = [], [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(T, 1, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(1).to(DEV)
        a = model.encoder(h_in.reshape(-1, hh.shape[1])).mean.view(T, 1, model.a_dim)
        fm, *_ = model.ssm.kalman_filter(a, sample_control=sc)
        phi, _ = integrate_phi(fm, tempo_head); phi = phi[:, 0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); revs.append(float(np.sum(np.where(dphi < -math.pi, dphi + TWO_PI, dphi)) / TWO_PI))
    model.train(); tempo_head.train(); m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(gb), m(gd), m(revs)


def train_run(train, val, sup_w, steps, tag):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = KVAEBarPointer(h_dim=512, a_dim=8, z_dim=8, K=5).to(DEV)
    tempo_head = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.Adam(list(model.parameters()) + list(tempo_head.parameters()), lr=1e-3)
    sc = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    pw = torch.tensor([8.0, 20.0], device=DEV)
    for step in range(1, steps + 1):
        H, Bt, Dt = batch(train, 256, 16)                      # (T,B,*)
        elbo, z, _ = kvae_elbo(model, H, sc, recon_w=0.3)
        phi, _ = integrate_phi(z, tempo_head)
        gbce = F.binary_cross_entropy_with_logits(geom_logits(phi), torch.stack([Bt, Dt], -1), pos_weight=pw)
        loss = -elbo + 8.0 * gbce
        if sup_w > 0:                                          # GT-phi supervision
            GP = torch.zeros_like(phi)
            for j in range(Bt.shape[1]):
                bf = torch.where(Bt[:, j] > 0.5)[0].cpu().numpy()
                GP[:, j] = torch.tensor(gt_barphase(bf, M, Bt.shape[0]), device=DEV, dtype=phi.dtype)
            loss = loss + sup_w * (1.0 - torch.cos(phi - GP)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(tempo_head.parameters()), 5.0); opt.step()
        if step % 250 == 0 or step == steps:
            gb, gd, rv = evaluate(model, tempo_head, val, "real")
            print(f"  [{tag}] step {step} | GEOM real beat {gb:.3f} db {gd:.3f} | phi-revs {rv:.1f}", flush=True)
    gb, gd, rv = evaluate(model, tempo_head, val, "real")
    gbs, gds, _ = evaluate(model, tempo_head, val, "shuffle"); gbz, gdz, _ = evaluate(model, tempo_head, val, "zero")
    print(f"  [{tag}] FINAL real beat {gb:.3f} db {gd:.3f} revs {rv:.1f} | shuf {gbs:.3f} | zero {gbz:.3f}", flush=True)
    return gb, gbs, gbz, rv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--ntrain", type=int, default=300); ap.add_argument("--nval", type=int, default=30)
    a = ap.parse_args()
    train = load("cache/acts/bt_train_rich", a.ntrain, 1); val = load("cache/acts/bt_val_rich", a.nval, 2)
    print(f"GEOM DIAGNOSE | train={len(train)} val={len(val)}", flush=True)

    cb, cd = clamp_test(val)
    print(f"\nA. CLAMP (read-out ceiling, GT phi): beat {cb:.3f}  downbeat {cd:.3f}", flush=True)

    print("\nB. GT-PHI-SUPERVISED (sup_w=20):", flush=True)
    sb, sbs, sbz, srev = train_run(train, val, sup_w=20.0, steps=a.steps, tag="sup")

    print("\nC. UNSUPERVISED reference (sup_w=0):", flush=True)
    ub, ubs, ubz, urev = train_run(train, val, sup_w=0.0, steps=a.steps, tag="unsup")

    print("\n==== VERDICT ====")
    print(f"  clamp ceiling      : beat {cb:.3f}")
    print(f"  GT-supervised      : beat {sb:.3f} (shuf {sbs:.3f}/zero {sbz:.3f}) revs {srev:.1f}")
    print(f"  unsupervised       : beat {ub:.3f} (shuf {ubs:.3f}/zero {ubz:.3f}) revs {urev:.1f}")
    print("  IF clamp high & sup high & sup-leak collapses => read-out SOUND, failure was inference/loss (FIXABLE)")
    print("  IF clamp high & sup FAILS                     => filter/latent can't carry rotating phi (STRUCTURAL)")
    print("  IF clamp LOW                                  => the read-out itself is broken (earlier diagnosis WRONG)")


if __name__ == "__main__":
    main()
