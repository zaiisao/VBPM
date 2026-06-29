"""DIAGNOSTIC 3: re-test the inherited 'free-run collapses' claim in the CURRENT model. Train M1, then
deploy three ways and compare beat-F:
  (a) FILTER  -- the M1 deploy (Kalman filter uses audio every frame),
  (b) FREE-RUN -- warmup-filter K frames, then roll the PRIOR open-loop (predict_future feeds its own
      predictions back; no audio after warmup) for the rest,
  (c) leak controls on the filter deploy.
If free-run << filter, the prior is not a standalone generator (free-run problem real). If free-run ~
filter, the inherited 'free-run collapses' claim does NOT hold for this model.
"""
import sys, random, importlib.util
import numpy as np
import torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
kvae_run = importlib.util.module_from_spec(kr); kr.loader.exec_module(kvae_run)
KVAEBarPointer, kvae_elbo, load, batch = kvae_run.KVAEBarPointer, kvae_run.kvae_elbo, kvae_run.load, kvae_run.batch
da = kvae_run.da; peaks, fmeas = da.peaks, da.fmeas
from kvae.sample_control import SampleControl
DEV = kvae_run.DEV; FPS = 86.1328125


@torch.no_grad()
def eval_modes(model, val, warmup=100, frames=1600):
    model.eval()
    sc = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    filt, freerun = [], []
    for hh, b, db in val:
        T = min(hh.shape[0], b.shape[0], frames)
        a = model.encoder(hh[:T].to(DEV)).mean.view(T, 1, model.a_dim)
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) < 2: continue
        # (a) filter all
        fm, *_ = model.ssm.kalman_filter(a, sample_control=sc)
        pf = torch.sigmoid(model.head(fm.view(T, model.z_dim)))[:, 0].cpu().numpy()
        filt.append(fmeas(ref, peaks(pf)))
        # (b) warmup filter then free-run the prior
        K = min(warmup, T - 2)
        fm2, fc2, fnm2, fnc2, mA2, mC2, _, _ = model.ssm.kalman_filter(a[:K], sample_control=sc)
        out = model.ssm.predict_future(a[:K], fm2, fc2, fnm2, fnc2, mA2, mC2, num_steps=T - K, sample_control=sc)
        means = out[1]                                    # list of [B,z_dim], length T
        z_seq = torch.stack(means[:T]).view(T, model.z_dim)
        pr = torch.sigmoid(model.head(z_seq))[:, 0].cpu().numpy()
        freerun.append(fmeas(ref, peaks(pr)))
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(filt), m(freerun)


@torch.no_grad()
def leak(model, val, h_mode, frames=1600):
    model.eval(); sc = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    out = []; n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(T, hh.shape[1], device=DEV) if h_mode == "zero" else h_use[:T].to(DEV)
        a = model.encoder(h_in).mean.view(T, 1, model.a_dim)
        fm, *_ = model.ssm.kalman_filter(a, sample_control=sc)
        pf = torch.sigmoid(model.head(fm.view(T, model.z_dim)))[:, 0].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: out.append(fmeas(ref, peaks(pf)))
    return float(np.nanmean(out)) if out else float("nan")


def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load("cache/acts/bt_train_rich", 200, 1); val = load("cache/acts/bt_val_rich", 30, 2)
    print(f"FREE-RUN TEST | train={len(train)} val={len(val)}", flush=True)
    model = KVAEBarPointer(h_dim=512, a_dim=8, z_dim=8, K=5).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sc = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    pw = torch.tensor([8.0, 20.0], device=DEV)
    for step in range(1, 501):
        H, Bt, Dt = batch(train, 256, 16)
        elbo, z, _ = kvae_elbo(model, H, sc)
        bl = model.head(z.reshape(-1, model.z_dim)).view(*z.shape[:2], 2)
        bce = F.binary_cross_entropy_with_logits(bl, torch.stack([Bt, Dt], -1), pos_weight=pw)
        loss = -elbo + 5.0 * bce
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    filt, fr = eval_modes(model, val)
    shuf = leak(model, val, "shuffle"); zero = leak(model, val, "zero")
    print(f"\n  (a) FILTER deploy   : beat {filt:.3f}")
    print(f"  (b) FREE-RUN prior  : beat {fr:.3f}   (warmup 100 frames, then open-loop)")
    print(f"  (c) filter leak     : shuf {shuf:.3f}  zero {zero:.3f}")
    print(f"\n  free-run / filter ratio = {fr/filt:.2f}")
    print("  IF free-run ~ filter -> 'free-run collapses' does NOT hold here; IF free-run << filter -> it does")
    print("DONE")


if __name__ == "__main__":
    main()
