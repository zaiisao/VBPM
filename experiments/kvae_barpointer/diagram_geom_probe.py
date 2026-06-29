"""DECISIVE re-examination: the diagram_arch GEOMETRIC read-out reaches ~0.78 then COLLAPSES (the old
RESULTS.md reported the post-collapse 0.03). Two questions:
  (1) Is the ~0.78 peak AUDIO-DRIVEN? -> eval geometric leak (real vs shuffle vs zero) at EVERY checkpoint,
      not just the end. A generic grid tops out ~0.38; real audio-lock >> that and collapses under leak.
  (2) Can we PREVENT the collapse? -> sweep free-bits fb (KL floor on the phase latent). If a higher fb
      holds the geometric read-out at ~0.78 (and leak stays collapsed), the model works -- the latent just
      needed anti-collapse, exactly the 'fix the latent, keep the structure' thesis.
Reports per checkpoint: decoder, geometric(real/shuf/zero), KL phi.
"""
import sys, importlib.util, argparse
import numpy as np
import torch

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
BPVAE, elbo_loss, evaluate, load_pool, sample_batch = da.BPVAE, da.elbo_loss, da.evaluate, da.load_pool, da.sample_batch
DEV = da.DEV
import random


def train_probe(fb, steps, train, val, eval_every=100):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = BPVAE(h_dim=512, hidden=64).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_geom = 0.0; best_geom_leak = (0.0, 0.0)
    for step in range(1, steps + 1):
        temp = 1.0 + (0.3 - 1.0) * min(step / steps, 1.0)
        h, b, db = sample_batch(train, 256, 16)
        loss, info = elbo_loss(model, h, b, db, temp, 8.0, 20.0, fb, 0.5)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % eval_every == 0 or step == steps:
            real = evaluate(model, val, give_beats=False, h_mode="real")
            shuf = evaluate(model, val, give_beats=False, h_mode="shuffle")
            zero = evaluate(model, val, give_beats=False, h_mode="zero")
            g, gs, gz = real["phase_beat"], shuf["phase_beat"], zero["phase_beat"]
            if g > best_geom: best_geom = g; best_geom_leak = (gs, gz)
            print(f"  [fb={fb}] step {step:4d} | KLphi {info['klp']:.1f} | "
                  f"decoder {real['dec_beat']:.3f} | GEOM real {g:.3f} shuf {gs:.3f} zero {gz:.3f}", flush=True)
    print(f"  [fb={fb}] BEST geom {best_geom:.3f} (at-peak leak shuf {best_geom_leak[0]:.3f} zero {best_geom_leak[1]:.3f})", flush=True)
    return best_geom, best_geom_leak


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n_train", type=int, default=400); ap.add_argument("--n_val", type=int, default=40)
    a = ap.parse_args()
    train = load_pool("cache/acts/bt_train_rich", a.n_train, seed=1)
    val = load_pool("cache/acts/bt_val_rich", a.n_val, seed=2)
    print(f"GEOM COLLAPSE PROBE | train={len(train)} val={len(val)} | generic-grid level ~0.38\n", flush=True)
    results = {}
    for fb in (0.1, 0.5, 1.0):
        print(f"=== free-bits fb={fb} ===", flush=True)
        results[fb] = train_probe(fb, a.steps, train, val)
    print("\n==== VERDICT ====")
    for fb, (g, (gs, gz)) in results.items():
        verdict = "AUDIO-DRIVEN" if (g > 0.55 and gs < g - 0.25 and gz < g - 0.25) else ("generic-grid" if g < 0.45 else "partial")
        print(f"  fb={fb}: best geom {g:.3f} | at-peak leak shuf {gs:.3f} zero {gz:.3f} -> {verdict}")
    print("  If any fb holds geom HIGH with leak COLLAPSED -> geometric works, collapse was the fixable problem")
    print("DONE")


if __name__ == "__main__":
    main()
