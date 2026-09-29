"""Decompose the F 0.468 -> 0.752 jump WITHOUT retraining.

The anchor is a deployment-time READ-OUT. So on already-trained weights we can swap
which read-out is used and re-evaluate. That splits the jump into two parts:

    (a) what the READ-OUT contributes  -- same weights, different anchor
    (b) what TRAINING changed          -- same anchor, different weights

Grid: {baseline, anchor_k v1, anchor_k v2} x {offset head, learned k argmax, closed-form
circular moment}. No training, no gradient, no annotations on the inference path.
"""
from __future__ import annotations

import argparse
import types

import torch

from vbpm.data.dataset import split_songs
from vbpm.data.excerpts import ExcerptDataset
from vbpm.model import VBPM, TWO_PI
from vbpm.scoring.evaluation import evaluate
from vbpm.variants.anchor_k import AnchorKVAE


def circular_anchor(mu, act, mask=None, center=True, eps=1e-12):
    """[B, T] phase, [B, T] evidence -> [B] shift, s.t. ``mu + shift`` sits on the bar.

    The circular mean of the evidence folded under the model's own phase:
    ``shift = -arg(sum_t a_t e^{i mu_t})``. Zero parameters, nothing learned. This is a
    MEASURING INSTRUMENT for decomposing where F lives, not a model component -- no
    trained variant in this repo uses it.
    """
    w = torch.ones_like(mu) if mask is None else mask
    weight = w * act
    if center:
        total = w.sum(1, keepdim=True).clamp(min=eps)
        weight = weight - w * (weight.sum(1, keepdim=True) / total)
    real = (weight * torch.cos(mu)).sum(1)
    imag = (weight * torch.sin(mu)).sum(1)
    return -torch.atan2(imag, real)

CKPT = {"baseline": "checkpoints/excerpt_base45/seed0.pt",
        "v1_k_trunk": "checkpoints/excerpt_anchor_k/seed0.pt",
        "v2_k_folded": "checkpoints/excerpt_anchor_k_v2/seed0.pt"}


class V1AnchorK(VBPM):
    """anchor_k v1 as banked: a single Linear on the TIME-MEAN trunk -> 64 slot logits.

    Reconstructed from the checkpoint's own shapes (k_head.weight [64, 256], no hidden
    layer), i.e. k WITHOUT the phase-folded representation -- the rung that scored below
    baseline. Inference only; this class is never trained.
    """

    def __init__(self, input_dim, anchor_slots=64, hidden=128, **kw):
        super().__init__(input_dim, hidden=hidden, **kw)
        self.anchor_slots = anchor_slots
        self.k_head = torch.nn.Linear(2 * hidden, anchor_slots)
        shifts = TWO_PI * torch.arange(anchor_slots, dtype=torch.float32) / anchor_slots
        self.register_buffer("slot_shifts",
                             torch.atan2(torch.sin(shifts), torch.cos(shifts)))

    def slot_logits(self, h, mu, mask=None):
        trunk = self.encoder(h)
        w = torch.ones(mu.shape, device=mu.device) if mask is None else mask
        pooled = (trunk * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)
        return self.k_head(pooled)


def build(kind, input_dim):
    common = dict(emission="triangle", drift_bound=0.01, bar_rate=True,
                  kappa_physical=2000.0)
    if kind == "baseline":
        return VBPM(input_dim, **common)
    if kind == "v1_k_trunk":
        return V1AnchorK(input_dim, anchor_slots=64, **common)
    return AnchorKVAE(input_dim, anchor_slots=64, **common)


def with_anchor(model, anchor):
    """Patch infer_phase (emission_probs calls it too) to deploy the chosen read-out."""
    def infer_phase(self, h, delta=None, mask=None):
        assert not self.training
        mu, _ = self.encoder(h, delta)                      # includes the offset head
        if anchor == "offset":
            return mu
        if anchor == "moment":
            act = torch.sigmoid(h[..., -1])                 # downbeat channel
            return mu + circular_anchor(mu, act, mask).unsqueeze(-1)
        if anchor == "k":
            k = self.slot_logits(h, mu, mask).argmax(-1)
            return mu + self.slot_shifts[k].unsqueeze(-1)
        raise ValueError(anchor)
    model.infer_phase = types.MethodType(infer_phase, model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    frontend_mod = __import__("vbpm.data.frontends.beat_this", fromlist=["FRONTEND"])
    frontend = frontend_mod.FRONTEND(checkpoint="final0", device=f"cuda:{args.gpu}",
                                     output="features+activations")
    _, _, test_songs = split_songs(None)
    test_set = ExcerptDataset(test_songs, frontend, 45.0, deterministic=True)
    print(f"gtzan test: {len(test_set)} songs, input_dim {frontend.num_channels}",
          flush=True)

    rows = []
    for kind, path in CKPT.items():
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        anchors = ["offset", "moment"] + (["k"] if kind != "baseline" else [])
        for anchor in anchors:
            model = build(kind, frontend.num_channels)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not unexpected, f"{kind}: unexpected {unexpected}"
            assert not [m for m in missing if "emission_b_floor" not in m], \
                f"{kind}: missing {missing}"
            model = with_anchor(model.to(device).eval(), anchor)
            res = evaluate(model, test_set, frontend, device, args.batch_size)
            # evaluate -> {dataset: {mode: (mean, n)}}; pool the rule-g family
            got = {}
            for per_mode in res.values():
                for mode, (val, n) in per_mode.items():
                    if mode.startswith("rule-g"):
                        got[mode] = (val, n)
            rows.append((kind, anchor, float(model.emission_b), got))
            print(f"  {kind:12s} anchor={anchor:7s} b={float(model.emission_b):.2f}  "
                  + "  ".join(f"{m}={v:.4f}(n={n})" for m, (v, n) in sorted(got.items())),
                  flush=True)

    print("\n===== ANCHOR-SWAP LADDER (gtzan, no retraining) =====")
    print(f"{'weights':14s} {'anchor':9s} {'b':>5s}  {'F':>6s} {'CMLt':>6s} {'AMLt':>6s}")
    for kind, anchor, b, g in rows:
        f = g.get("rule-g", (float("nan"),))[0]
        c = g.get("rule-g CMLt", (float("nan"),))[0]
        a = g.get("rule-g AMLt", (float("nan"),))[0]
        print(f"{kind:14s} {anchor:9s} {b:5.2f}  {f:6.3f} {c:6.3f} {a:6.3f}")


if __name__ == "__main__":
    main()
