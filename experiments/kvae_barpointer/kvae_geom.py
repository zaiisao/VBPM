"""KVAE-bar-pointer M2: the GEOMETRIC read-out (the actual contribution).

M1 proved an exact differentiable filter escapes the amortized wall, read out by a learned head.
M2 imposes the bar-pointer GEOMETRY so beats are read straight off the phase, with the filter (not an
amortized encoder) driving rotation:

  z = [cos phi, sin phi, ...]   (rotational latent; A_k init as K rotation matrices at K tempi 40-250 BPM)
  Kalman filter tracks z from a=enc(h)        -> EXACT posterior, cannot collapse to prior
  phi = atan2(z[...,1], z[...,0])             -> bar phase
  GEOMETRIC read-out: beats = m*phi wraps, downbeats = phi wraps   (your deployment)
  training: KVAE ELBO (recon of activations + kalman terms) + FIXED geometric emission BCE on phi
            (kappa*cos(m*phi) for beats, kappa*cos(phi) for downbeats) -> ties phi to the audio's beats

PASS = geometric beat/db HIGH and leak controls (shuffled/zero) COLLAPSE and phi actually rotates.
"""
import sys, math, random, argparse, importlib.util
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
kvae_run = importlib.util.module_from_spec(kr); kr.loader.exec_module(kvae_run)
KVAEBarPointer, kvae_elbo, load, batch = kvae_run.KVAEBarPointer, kvae_run.kvae_elbo, kvae_run.load, kvae_run.batch
da = kvae_run.da; phase_beats, phase_downbeats, fmeas = da.phase_beats, da.phase_downbeats, da.fmeas
from kvae.sample_control import SampleControl
DEV = kvae_run.DEV; FPS = 86.1328125; TWO_PI = 2 * math.pi; M = 4; KAPPA = 8.0


def rotation_init(z_dim, K):
    """A_k = block-diag[R(theta_k), I]; theta_k = bar-phase advance/frame at K log-spaced tempi."""
    bpms = torch.exp(torch.linspace(math.log(40), math.log(250), K))
    A = torch.zeros(K, z_dim, z_dim)
    for k, bpm in enumerate(bpms):
        th = TWO_PI * (float(bpm) / 60 / M) / FPS
        c, sn = math.cos(th), math.sin(th)
        A[k, 0, 0] = c; A[k, 0, 1] = -sn; A[k, 1, 0] = sn; A[k, 1, 1] = c
        for d in range(2, z_dim):
            A[k, d, d] = 1.0
    return A


def phi_of(z):  # z [...,z_dim] -> phi [...]
    return torch.atan2(z[..., 1], z[..., 0]) % TWO_PI


def geom_logits(phi):
    return torch.stack([KAPPA * torch.cos(M * phi), KAPPA * torch.cos(phi)], -1)


@torch.no_grad()
def evaluate(model, val, h_mode="real", mode="filter", frames=1600):
    """Geometric read-out from the FILTERED (causal) or SMOOTHED latent. + phi revolutions."""
    model.eval()
    sc = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    gb, gd, revs = [], [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(T, 1, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].unsqueeze(1).to(DEV)
        a = model.encoder(h_in.reshape(-1, hh.shape[1])).mean.view(T, 1, model.a_dim)
        fm, fc, fnm, fnc, matA, matC, _, _ = model.ssm.kalman_filter(a, sample_control=sc)
        if mode == "smooth":
            sm, *_ = model.ssm.kalman_smooth(a, fm, fc, fnm, fnc, matA, matC, sample_control=sc)
            z = sm
        else:
            z = fm
        phi = phi_of(z.view(T, model.z_dim)).cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, phase_beats(phi, M)))
        if len(dref) >= 2: gd.append(fmeas(dref, phase_downbeats(phi)))
        dphi = np.diff(phi); revs.append(float(np.sum(np.where(dphi < -math.pi, dphi + TWO_PI, dphi)) / TWO_PI))
    model.train(); m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(gb), m(gd), m(revs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--a_dim", type=int, default=4)
    ap.add_argument("--z_dim", type=int, default=4); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--geom_w", type=float, default=8.0); ap.add_argument("--recon_w", type=float, default=0.3)
    ap.add_argument("--ntrain", type=int, default=400); ap.add_argument("--nval", type=int, default=40)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f"KVAE-GEOM M2 | z_dim={args.z_dim} a_dim={args.a_dim} K={args.K} geom_w={args.geom_w} recon_w={args.recon_w} "
          f"| rotational z, geometric phi-wrap read-out, EXACT filter", flush=True)
    train = load("cache/acts/bt_train_rich", args.ntrain, 1); val = load("cache/acts/bt_val_rich", args.nval, 2)
    print(f"train={len(train)} val={len(val)}", flush=True)

    model = KVAEBarPointer(h_dim=512, a_dim=args.a_dim, z_dim=args.z_dim, K=args.K).to(DEV)
    model.ssm.mat_A_K = rotation_init(args.z_dim, args.K).to(DEV)   # rotation bank init (setter -> Parameter)
    init_mean = torch.zeros(args.z_dim); init_mean[0] = 1.0          # start ON the circle (rotating 0 stays 0)
    model.ssm.initial_state_mean = init_mean.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sc = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    pw = torch.tensor([8.0, 20.0], device=DEV)

    for step in range(1, args.steps + 1):
        H, Bt, Dt = batch(train, 256, 16)
        elbo, z, info = kvae_elbo(model, H, sc, recon_w=args.recon_w)
        phi = phi_of(z)                                  # [T,B] from smoothed z
        gbce = F.binary_cross_entropy_with_logits(geom_logits(phi), torch.stack([Bt, Dt], -1), pos_weight=pw)
        loss = -elbo + args.geom_w * gbce
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 200 == 0 or step == args.steps:
            gb, gd, rv = evaluate(model, val, "real", "filter")
            print(f"\nstep {step} | elbo {float(elbo):.1f} gbce {float(gbce):.3f} | "
                  f"GEOM filter: beat {gb:.3f} downbeat {gd:.3f} | phi-revs {rv:.1f}", flush=True)

    gbf, gdf, rvf = evaluate(model, val, "real", "filter")
    gbs_, gds_, rvs_ = evaluate(model, val, "real", "smooth")
    gbsh, gdsh, _ = evaluate(model, val, "shuffle", "filter"); gbz, gdz, _ = evaluate(model, val, "zero", "filter")
    print("\n--- FINAL (GEOMETRIC bar-pointer read-out = phi wraps) ---")
    print(f"  filter real  : beat {gbf:.3f}  downbeat {gdf:.3f}  phi-revs {rvf:.1f}   <- causal deploy")
    print(f"  smooth real  : beat {gbs_:.3f}  downbeat {gds_:.3f}  phi-revs {rvs_:.1f}   (offline)")
    print(f"  filter shuf  : beat {gbsh:.3f}  downbeat {gdsh:.3f}   (must COLLAPSE)")
    print(f"  filter zero  : beat {gbz:.3f}  downbeat {gdz:.3f}   (must COLLAPSE)")
    print("VERDICT: geometric beat/db HIGH + leak COLLAPSE + phi rotates => GEOMETRIC bar-pointer WORKS")
    torch.save({"model": model.state_dict(), "args": vars(args)}, "experiments/kvae_barpointer/m2_geom.pt")


if __name__ == "__main__":
    main()
