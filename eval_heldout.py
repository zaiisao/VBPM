"""
eval_heldout.py -- generalisation check on HELD-OUT data for both models.

Trains on a training set (seed 0) and evaluates the audio-only beat prediction on a FRESH
held-out set drawn with a different seed (never seen in training). Reports train vs held-out
accuracy so the gap shows whether the model generalises or merely memorised the batch.

    python eval_heldout.py --which cont    # continuous VAE-DBN  (predict mode)
    python eval_heldout.py --which disc    # discrete DBN oracle (forward-backward)
"""
import argparse, torch
import torch.nn.functional as F
from vae_dbn import synth

B_TR, B_TE, SEED_TR, SEED_TE = 24, 24, 0, 777   # held-out uses a different seed


def eval_cont(steps=200):
    from vae_dbn import train, load_model, predict_labels, DEV
    xtr, btr = synth(B=B_TR, seed=SEED_TR)
    xte, bte = synth(B=B_TE, seed=SEED_TE)                       # HELD-OUT (unseen seed)
    train(xtr.to(DEV), btr.to(DEV), steps=steps, log_name="heldout_cont")
    m = load_model()
    out = {}
    for name, (x, b) in [("train", (xtr, btr)), ("held-out", (xte, bte))]:
        bhat, _ = predict_labels(m, x.to(DEV), N=64)             # audio-only prior rollout
        out[name] = (bhat.cpu() == b).float().mean().item()
    print(f"\n[CONTINUOUS] audio-only predict accuracy  train={out['train']:.3f}  "
          f"held-out={out['held-out']:.3f}  (gap {out['train']-out['held-out']:+.3f})")
    return out


def eval_disc(steps=120):
    from vae_dbn_discrete import DiscreteDBN
    xtr, btr = synth(B=B_TR, seed=SEED_TR)
    xte, bte = synth(B=B_TE, seed=SEED_TE)                       # HELD-OUT (unseen seed)
    m = DiscreteDBN(x_dim=xtr.size(-1))
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for t in range(1, steps + 1):
        loss = -m.forward_ll(xtr, btr).mean()                   # exact NLL on train
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if t == 1 or t % 40 == 0:
            print(f"  step {t:3d} | train NLL = {loss.item():.1f}")
    out = {}
    for name, (x, b) in [("train", (xtr, btr)), ("held-out", (xte, bte))]:
        out[name] = (m.predict_labels(x)[0] == b).float().mean().item()
    print(f"\n[DISCRETE] audio-only predict accuracy  train={out['train']:.3f}  "
          f"held-out={out['held-out']:.3f}  (gap {out['train']-out['held-out']:+.3f})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["cont", "disc"], required=True)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    if a.which == "cont":
        eval_cont(steps=a.steps or 200)
    else:
        eval_disc(steps=a.steps or 120)
