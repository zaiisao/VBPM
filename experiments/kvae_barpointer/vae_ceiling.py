"""DIAGNOSTIC 2: is the VAE (filter + learned head) fundamentally capped below the frontend (0.929),
or just under-scaled? Two configs vs the frontend peak-pick ceiling:
  A. scaled M1 (a/z=16, K=8, 1500 steps, 800 songs) on the 512-dim penultimate -- does scale help?
  B. + act2 (concat the frontend's OWN beat/downbeat probs to the encoder input) -- given the frontend's
     output, can the VAE represent/refine it toward 0.929? If yes, the VAE is capacity/info-limited, not
     fundamentally capped; if even B << 0.929, something deeper limits the read-out.
"""
import sys, glob, random, argparse, importlib.util
import numpy as np
import torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
kvae_run = importlib.util.module_from_spec(kr); kr.loader.exec_module(kvae_run)
KVAEBarPointer, kvae_elbo, batch, evaluate = kvae_run.KVAEBarPointer, kvae_run.kvae_elbo, kvae_run.batch, kvae_run.evaluate
da = kvae_run.da; peaks, fmeas = da.peaks, da.fmeas
from kvae.sample_control import SampleControl
DEV = kvae_run.DEV; FPS = 86.1328125


def load_aug(cd, n, seed, use_act2):
    fs = sorted(glob.glob(f"{cd}/*.pt")); random.Random(seed).shuffle(fs); out = []
    for f in fs[:n]:
        d = torch.load(f, map_location="cpu"); hh = d["activations"].float()
        if hh.shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        if use_act2:
            hh = torch.cat([hh, d["act2"].float()], dim=-1)             # [T, 514]
        out.append((hh, d["beat_targets"].float(), d["downbeat_targets"].float()))
    return out


def frontend_ceiling(val_files, n, seed):
    fs = sorted(glob.glob(f"{val_files}/*.pt")); random.Random(seed).shuffle(fs)
    gb = []; cnt = 0
    for f in fs:
        if cnt >= n: break
        d = torch.load(f, map_location="cpu")
        if d["activations"].shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        cnt += 1
        a2 = d["act2"].float().numpy(); bt = d["beat_targets"].numpy()
        ref = np.where(bt > 0.5)[0] / FPS
        if len(ref) >= 2: gb.append(fmeas(ref, peaks(a2[:, 0])))
    return float(np.nanmean(gb))


def train_eval(use_act2, steps, ntrain, nval, a_dim, z_dim, K, tag):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = load_aug("cache/acts/bt_train_rich", ntrain, 1, use_act2)
    val = load_aug("cache/acts/bt_val_rich", nval, 2, use_act2)
    h_dim = train[0][0].shape[-1]
    model = KVAEBarPointer(h_dim=h_dim, a_dim=a_dim, z_dim=z_dim, K=K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sc = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    pw = torch.tensor([8.0, 20.0], device=DEV)
    best = 0.0
    for step in range(1, steps + 1):
        H, Bt, Dt = batch(train, 256, 16)
        elbo, z, _ = kvae_elbo(model, H, sc)
        bl = model.head(z.reshape(-1, model.z_dim)).view(*z.shape[:2], 2)
        bce = F.binary_cross_entropy_with_logits(bl, torch.stack([Bt, Dt], -1), pos_weight=pw)
        loss = -elbo + 5.0 * bce
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 300 == 0 or step == steps:
            bF, dF = evaluate(model, val, "real"); best = max(best, bF)
            print(f"  [{tag}] step {step} | beat {bF:.3f} db {dF:.3f} (best {best:.3f})", flush=True)
    bFs, _ = evaluate(model, val, "shuffle"); bFz, _ = evaluate(model, val, "zero")
    print(f"  [{tag}] leak: shuf {bFs:.3f} zero {bFz:.3f}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ntrain", type=int, default=800); ap.add_argument("--nval", type=int, default=40)
    a = ap.parse_args()
    ceil = frontend_ceiling("cache/acts/bt_val_rich", a.nval, 2)
    print(f"VAE CEILING | frontend peak-pick ceiling (this val) = {ceil:.3f}\n", flush=True)
    print("A. scaled M1 (no act2, a/z=16, K=8):", flush=True)
    A = train_eval(False, a.steps, a.ntrain, a.nval, 16, 16, 8, "scaled")
    print("\nB. scaled M1 + act2 (frontend output concatenated):", flush=True)
    B = train_eval(True, a.steps, a.ntrain, a.nval, 16, 16, 8, "act2")
    print(f"\n==== VERDICT ====")
    print(f"  frontend ceiling : {ceil:.3f}")
    print(f"  scaled M1 (best) : {A:.3f}   (gap {ceil - A:.3f})")
    print(f"  + act2    (best) : {B:.3f}   (gap {ceil - B:.3f})")
    print("  IF B ~ ceiling -> VAE is info/capacity-limited (fixable); IF B << ceiling -> deeper read-out limit")
    print("DONE")


if __name__ == "__main__":
    main()
