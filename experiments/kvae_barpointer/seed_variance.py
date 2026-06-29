"""DIAGNOSTIC 4: seed variance. Almost everything ran at one seed. Re-train M1 (filter + learned head)
at 3 seeds and report mean +/- std of the deploy beat-F + leak, so we know whether our headline numbers
are trustworthy or noise.
"""
import sys, random, importlib.util
import numpy as np
import torch, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")
kr = importlib.util.spec_from_file_location("kr", f"{ROOT}/experiments/kvae_barpointer/kvae_run.py")
kvae_run = importlib.util.module_from_spec(kr); kr.loader.exec_module(kvae_run)
KVAEBarPointer, kvae_elbo, load, batch, evaluate = (kvae_run.KVAEBarPointer, kvae_run.kvae_elbo,
                                                    kvae_run.load, kvae_run.batch, kvae_run.evaluate)
from kvae.sample_control import SampleControl
DEV = kvae_run.DEV


def run_seed(seed, train, val, steps=500):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = KVAEBarPointer(h_dim=512, a_dim=8, z_dim=8, K=5).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sc = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    pw = torch.tensor([8.0, 20.0], device=DEV)
    for step in range(1, steps + 1):
        H, Bt, Dt = batch(train, 256, 16)
        elbo, z, _ = kvae_elbo(model, H, sc)
        bl = model.head(z.reshape(-1, model.z_dim)).view(*z.shape[:2], 2)
        bce = F.binary_cross_entropy_with_logits(bl, torch.stack([Bt, Dt], -1), pos_weight=pw)
        loss = -elbo + 5.0 * bce
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    bF, dF = evaluate(model, val, "real"); bFs, _ = evaluate(model, val, "shuffle"); bFz, _ = evaluate(model, val, "zero")
    return bF, dF, bFs, bFz


def main():
    train = load("cache/acts/bt_train_rich", 200, 1); val = load("cache/acts/bt_val_rich", 30, 2)
    print(f"SEED VARIANCE | train={len(train)} val={len(val)} | M1 filter+head, 500 steps/seed", flush=True)
    res = []
    for s in (0, 1, 2):
        bF, dF, bFs, bFz = run_seed(s, train, val)
        print(f"  seed {s}: beat {bF:.3f} db {dF:.3f} | shuf {bFs:.3f} zero {bFz:.3f}", flush=True)
        res.append((bF, dF, bFs, bFz))
    a = np.array(res)
    print(f"\n  beat-F: mean {a[:,0].mean():.3f} +/- {a[:,0].std():.3f}  (min {a[:,0].min():.3f} max {a[:,0].max():.3f})")
    print(f"  db-F  : mean {a[:,1].mean():.3f} +/- {a[:,1].std():.3f}")
    print(f"  leak  : shuf mean {a[:,2].mean():.3f}  zero mean {a[:,3].mean():.3f}  (must stay LOW)")
    print("DONE")


if __name__ == "__main__":
    main()
