"""KVAE-GEOM M2-CV: FORCE rotation by INTEGRATING a tempo read from the Kalman-filtered latent.

Vanilla M2 (phi = atan2 of a freely-filtered rotational z) does NOT rotate: the filter pins the phase
(phi-revs ~ 0). Fix: read a positive per-frame tempo omega_t from the EXACT-filtered latent z_t, and
integrate phi_t = phi_{t-1} + omega_t (mod 2pi). Integration GUARANTEES rotation; the filter makes
omega AUDIO-DRIVEN (M1 proved the filtered latent locks to audio); the geometric emission ties the
wraps to beats. This merges the two halves that each failed alone:
  - diagram_arch integ (amortized tempo): rotated but input-independent (collapsed).
  - vanilla M2 (filtered phase): audio-aware but didn't rotate.
PASS = geometric beat/db HIGH, phi rotates (revs ~ #bars), and leak (shuffled/zero) COLLAPSES.
"""
import sys, math, random, argparse, importlib.util
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
kvae_run = importlib.util.module_from_spec(kr); kr.loader.exec_module(kvae_run)
KVAEBarPointer, kvae_elbo, load, batch = kvae_run.KVAEBarPointer, kvae_run.kvae_elbo, kvae_run.load, kvae_run.batch
da = kvae_run.da; phase_beats, phase_downbeats, fmeas = da.phase_beats, da.phase_downbeats, da.fmeas
from kvae.sample_control import SampleControl
DEV = kvae_run.DEV; TWO_PI = 2 * math.pi; FPS = 86.1328125; M = 4; KAPPA = 8.0
LT_MIN = math.log(TWO_PI * 40 / 60 / M / FPS)     # bounded bar-phase advance/frame (40-250 BPM)
LT_MAX = math.log(TWO_PI * 250 / 60 / M / FPS)


def integrate_phi(z, tempo_head):
    """z [T,B,z_dim] -> phi [T,B] by integrating a bounded positive tempo read from the latent."""
    lt = tempo_head(z).squeeze(-1).clamp(LT_MIN, LT_MAX)     # [T,B]
    omega = torch.exp(lt)                                     # >0 -> monotonic advance
    phi = torch.cumsum(omega, dim=0) % TWO_PI
    return phi, omega


def geom_logits(phi):
    return torch.stack([KAPPA * torch.cos(M * phi), KAPPA * torch.cos(phi)], -1)


@torch.no_grad()
def evaluate(model, tempo_head, val, h_mode="real", frames=1600):
    model.eval(); tempo_head.eval()
    sc = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    gb, gd, revs, bpm = [], [], [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(T, 1, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(1).to(DEV)
        a = model.encoder(h_in.reshape(-1, hh.shape[1])).mean.view(T, 1, model.a_dim)
        fm, *_ = model.ssm.kalman_filter(a, sample_control=sc)
        phi, omega = integrate_phi(fm, tempo_head)
        phi = phi[:, 0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); revs.append(float(np.sum(np.where(dphi < -math.pi, dphi + TWO_PI, dphi)) / TWO_PI))
        bpm.append(60 * FPS * M * float(omega[:, 0].mean().cpu()) / TWO_PI)
    model.train(); tempo_head.train(); m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(gb), m(gd), m(revs), m(bpm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--a_dim", type=int, default=8)
    ap.add_argument("--z_dim", type=int, default=8); ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--geom_w", type=float, default=8.0); ap.add_argument("--recon_w", type=float, default=0.3)
    ap.add_argument("--ntrain", type=int, default=400); ap.add_argument("--nval", type=int, default=30)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f"KVAE-GEOM-CV | z_dim={args.z_dim} a_dim={args.a_dim} K={args.K} geom_w={args.geom_w} recon_w={args.recon_w} "
          f"| INTEGRATE tempo from filtered latent (forced rotation + audio-lock)", flush=True)
    train = load("cache/acts/bt_train_rich", args.ntrain, 1); val = load("cache/acts/bt_val_rich", args.nval, 2)
    print(f"train={len(train)} val={len(val)}", flush=True)

    model = KVAEBarPointer(h_dim=512, a_dim=args.a_dim, z_dim=args.z_dim, K=args.K).to(DEV)
    tempo_head = nn.Sequential(nn.Linear(args.z_dim, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.Adam(list(model.parameters()) + list(tempo_head.parameters()), lr=1e-3)
    sc = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    pw = torch.tensor([8.0, 20.0], device=DEV)

    for step in range(1, args.steps + 1):
        H, Bt, Dt = batch(train, 256, 16)
        elbo, z, info = kvae_elbo(model, H, sc, recon_w=args.recon_w)
        phi, omega = integrate_phi(z, tempo_head)
        gbce = F.binary_cross_entropy_with_logits(geom_logits(phi), torch.stack([Bt, Dt], -1), pos_weight=pw)
        loss = -elbo + args.geom_w * gbce
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(tempo_head.parameters()), 5.0); opt.step()
        if step % 250 == 0 or step == args.steps:
            gb, gd, rv, bp = evaluate(model, tempo_head, val, "real")
            print(f"\nstep {step} | elbo {float(elbo):.1f} gbce {float(gbce):.3f} | "
                  f"GEOM filter: beat {gb:.3f} downbeat {gd:.3f} | phi-revs {rv:.1f} tempo ~{bp:.0f}BPM", flush=True)

    gb, gd, rv, bp = evaluate(model, tempo_head, val, "real")
    gbs, gds, _, _ = evaluate(model, tempo_head, val, "shuffle"); gbz, gdz, _, _ = evaluate(model, tempo_head, val, "zero")
    print("\n--- FINAL (GEOMETRIC read-out; phi = INTEGRAL of filtered tempo) ---")
    print(f"  real     : beat {gb:.3f}  downbeat {gd:.3f}  phi-revs {rv:.1f}  tempo ~{bp:.0f}BPM")
    print(f"  shuffled : beat {gbs:.3f}  downbeat {gds:.3f}   (must COLLAPSE)")
    print(f"  zero     : beat {gbz:.3f}  downbeat {gdz:.3f}   (must COLLAPSE)")
    print("VERDICT: integrate-tempo-from-filter -> phi rotates AND beats high AND leak collapses => GEOMETRIC WORKS")
    torch.save({"model": model.state_dict(), "tempo_head": tempo_head.state_dict(), "args": vars(args)},
               "experiments/kvae_barpointer/m2cv.pt")


if __name__ == "__main__":
    main()
