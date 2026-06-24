"""Train CHART as a structured VAE: fixed bar-pointer DBN prior p(z) + LEARNED emission.

The bar-pointer DBN (models/bar_pointer_dbn.py, fixed madmom lambdas) is p(z). A small
head maps the rich Beat-This features h_t -> per-class emission log-potentials (the
encoder / "posterior"); we run the DBN's differentiable forward-backward to get the
structured per-frame class marginal and train the head against GT beats THROUGH the
fixed prior (so the posterior refines, but is regularized by, the bar-pointer dynamics).
Deploy = Viterbi over the learned emission (audio-conditioned, the faithful inference).

Compares the trained learned-emission DBN against:
  * the fixed-madmom-emission DBN on the same frontend activation, and
  * peak-picking the activation (no dynamics).

    python tests/train_dbn.py --train_cache cache/acts/bt_train_rich \
        --val_cache cache/acts/bt_val_rich --num_beats 4 --steps 400
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.bar_pointer_dbn import BarPointerDBN
from evaluation.phase_converter import extract_beat_timestamps
from evaluation.score import evaluate_beats, evaluate_downbeats, frames_to_beat_times

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class Emission(nn.Module):
    """h_t [in_dim] -> log-softmax class potentials [C]  (per-frame; the DBN adds time)."""
    def __init__(self, in_dim, C, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, C))

    def forward(self, h):
        return torch.log_softmax(self.net(h), dim=-1)


def _load(cache, n):
    return [torch.load(f, map_location="cpu") for f in sorted(glob.glob(cache + "/*.pt"))[:n]]


def _dilate(x: torch.Tensor, w: int) -> torch.Tensor:
    k = torch.ones(1, 1, 2 * w + 1, device=x.device)
    return (F.conv1d(x.view(1, 1, -1), k, padding=w)[0, 0] > 0.5).float()


def _crop(rec, T, dev):
    act = rec["activations"].float().to(dev)
    bt = rec["beat_targets"].float().to(dev)
    db = rec["downbeat_targets"].float().to(dev)
    L = act.shape[0]
    bi = torch.where(bt > 0.5)[0]
    if len(bi):
        c = int(bi.float().mean()); s = min(max(c - T // 2, 0), max(L - T, 0))
    else:
        s = max(0, (L - T) // 2)
    e = min(s + T, L)
    return act[s:e], bt[s:e], db[s:e]


@torch.no_grad()
def _score(dbn, head, val, fps, dev, maxT):
    learned_b, learned_d, fixed_b, peak_b = [], [], [], []
    for r in val:
        act = r["activations"].float().to(dev)[:maxT]
        a2 = r["act2"].float().to(dev)[:maxT]
        bt = r["beat_targets"].numpy()[:maxT]; db = r["downbeat_targets"].numpy()[:maxT]
        ref = frames_to_beat_times(bt, fps); ref_db = frames_to_beat_times(db, fps)
        if len(ref) < 2:
            continue
        # learned-emission DBN (Viterbi)
        bfr, dfr = dbn.decode_emission(head(act))
        learned_b.append(evaluate_beats(ref, bfr.cpu().numpy().astype(float) / fps)["F-measure"])
        if len(ref_db) >= 2 and not dbn.beats_only:
            learned_d.append(evaluate_downbeats(ref_db, dfr.cpu().numpy().astype(float) / fps)["db_F-measure"])
        # fixed-madmom-emission DBN on the frontend activation
        bfr2, _ = dbn.decode(a2)
        fixed_b.append(evaluate_beats(ref, bfr2.cpu().numpy().astype(float) / fps)["F-measure"])
        # peak-pick ceiling
        peak_b.append(evaluate_beats(ref, extract_beat_timestamps(a2[:, 0].cpu().numpy(), fps=fps))["F-measure"])
    m = lambda x: float(np.mean(x)) if x else float("nan")
    return m(learned_b), m(learned_d), m(fixed_b), m(peak_b)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", default="cache/acts/bt_train_rich")
    p.add_argument("--val_cache", default="cache/acts/bt_val_rich")
    p.add_argument("--num_beats", type=int, default=4)
    p.add_argument("--num_intervals", type=int, default=40)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--frames", type=int, default=384)
    p.add_argument("--eval_frames", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max_train", type=int, default=400)
    p.add_argument("--max_val", type=int, default=24)
    p.add_argument("--dilate", type=int, default=3)
    p.add_argument("--tag", default="dbn")
    p.add_argument("--wandb_project", default="chart")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--no_wandb", action="store_true")
    cli = p.parse_args()

    use_wandb = _WANDB_AVAILABLE and not cli.no_wandb
    if use_wandb:
        _wandb.init(project=cli.wandb_project, name=cli.wandb_name or f"dbn_{cli.tag}",
                    config=vars(cli), tags=["bar_pointer_dbn"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = _load(cli.train_cache, cli.max_train)
    val = _load(cli.val_cache, cli.max_val)
    fps = float(train[0]["fps"])
    in_dim = train[0]["activations"].shape[-1]
    beats_only = cli.num_beats == 1
    dbn = BarPointerDBN(fps=fps, num_beats=cli.num_beats, num_intervals=cli.num_intervals,
                        beats_only=beats_only).to(dev)
    C = dbn.num_classes
    head = Emission(in_dim, C).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=cli.lr)
    print(f"[{cli.tag}] DBN {dbn.num_states} states (num_beats={cli.num_beats}, C={C}); "
          f"in_dim={in_dim}; train={len(train)} val={len(val)} fps={fps:.2f}", flush=True)

    for step in range(1, cli.steps + 1):
        r = train[step % len(train)]
        act, bt, db = _crop(r, cli.frames, dev)
        marg = dbn.forward_backward(dbn.class_logp_to_states(head(act)))        # [T, C] log
        if C == 3:
            p_beat = torch.logsumexp(marg[:, 1:], dim=1).exp().clamp(1e-6, 1 - 1e-6)
            p_down = marg[:, 2].exp().clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p_beat, _dilate(bt, cli.dilate)) \
                + F.binary_cross_entropy(p_down, _dilate(db, cli.dilate))
        else:
            p_beat = marg[:, 1].exp().clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p_beat, _dilate(bt, cli.dilate))
        if not torch.isfinite(loss):
            print(f"[{cli.tag}] non-finite at step {step}, skip", flush=True); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if step % cli.eval_every == 0 or step == cli.steps:
            lb, ld, fb, pk = _score(dbn, head, val, fps, dev, cli.eval_frames)
            print(f"[{cli.tag}] step {step:04d} | loss={loss.item():.3f} | "
                  f"LEARNED-DBN beatF={lb:.3f} dbF={ld:.3f} | fixed-DBN beatF={fb:.3f} | peak-pick={pk:.3f}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
