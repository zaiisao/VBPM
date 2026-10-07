"""L2 on real songs: does a teacher-forced emission read the oracle path of annotated excerpts."""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from ..model import VBPM
from ..nets import EmissionModel
from ..specs import EmissionSpec
from ..util.oracle import bar_meters, beat_positions, labels_from_beats, oracle_draw

FPS = 50.0
METERS = (3, 4)


def windows(song, frames, count, rng):
    """Up to ``count`` (draw, cls, meter) oracle windows of one song."""
    beat_times = song["beat_time"]
    downbeat_times = beat_times[song["beat_value"] == 1]
    beat_times = np.unique(np.asarray(beat_times, dtype=np.float64))
    downbeat_times = np.asarray(downbeat_times, dtype=np.float64)
    if len(downbeat_times) < 4 or len(beat_times) < 8:
        return []
    positions = beat_positions(beat_times, downbeat_times)
    meters = bar_meters(positions)
    closed = [i for i, m in enumerate(meters) if m is not None]
    if not closed:
        return []
    lo, hi = beat_times[closed[0]] + 0.05, beat_times[-1] - frames / FPS
    out = []
    for _ in range(4 * count):
        if len(out) == count or hi <= lo:
            break
        start = float(rng.uniform(lo, hi))
        inside = [m for t, m in zip(beat_times, meters) if start - 5 <= t <= start + frames / FPS]
        if any(m is not None and m not in METERS for m in inside):
            continue
        try:
            draw, crossings = oracle_draw(
                (beat_times - start) * FPS, positions, meters, frames, METERS
            )
        except ValueError:
            continue
        cls = labels_from_beats(beat_times - start, downbeat_times - start, 0, frames, FPS)
        meter = meters[int(np.searchsorted(beat_times, start))] or 4
        out.append((draw, cls, meter))
    return out


def perturb(draw, meter, name):
    """One of the L2 wrong paths, built from the oracle draw."""
    out = {k: v.clone() for k, v in draw.items()}
    if name == "half-beat shift":
        out["phase0"] += math.pi / meter
    elif name == "tempo x2":
        out["velocity"] *= 2.0
    elif name == "tempo x1/2":
        out["velocity"] /= 2.0
    elif name == "wrong meter":
        out["meter"] = out["meter"].flip(-1)
    return out


def replay(paths, draws, device, chunk=64):
    """(phi, velocity, meter) paths for a list of single-item draws."""
    parts = []
    for i in range(0, len(draws), chunk):
        batch = {k: torch.cat([d[k] for d in draws[i : i + chunk]]).to(device) for k in draws[0]}
        mask = torch.ones(batch["velocity"].shape, device=device)
        path = paths.draws_to_paths(batch, mask)
        parts.append((path["phi_path"], path["velocity_path"], path["meter_path"]))
    return [torch.cat(p) for p in zip(*parts)]


def train(emission, inputs, cls, steps, batch, seed):
    """Teacher-forced emission fit on fixed paths."""
    opt = torch.optim.Adam(emission.parameters(), lr=3e-4)
    gen = torch.Generator(device=cls.device).manual_seed(seed)
    mask = torch.ones(batch, cls.shape[1], device=cls.device)
    for step in range(steps):
        idx = torch.randint(0, len(cls), (batch,), generator=gen, device=cls.device)
        loss = (
            -emission.loglik(*[x[idx] for x in inputs], cls[idx], mask, None).mean() / cls.shape[1]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"  step {step:5d}  recon/frame {-float(loss):.4f}", flush=True)
    return emission


@torch.no_grad()
def recon(emission, inputs, cls, chunk=64):
    """Mean recon per frame."""
    total = 0.0
    for i in range(0, len(cls), chunk):
        mask = torch.ones(cls[i : i + chunk].shape, device=cls.device)
        total += float(
            emission.loglik(
                *[x[i : i + chunk] for x in inputs], cls[i : i + chunk], mask, None
            ).sum()
        )
    return total / cls.numel()


def main():
    """Teacher-forced emission on real oracle paths, scored against wrong paths."""
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-songs", type=int, default=400)
    p.add_argument("--held-songs", type=int, default=100)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--init", choices=["zero", "random"], default="zero")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    frames = int(round(args.seconds * FPS))
    from ..data.dataset import load_beat_this

    data = load_beat_this(7)
    train_songs, val_songs = data.train_dataset.items, data.val_dataset.items
    train_songs = [train_songs[i] for i in rng.permutation(len(train_songs))[: args.train_songs]]
    val_songs = [val_songs[i] for i in rng.permutation(len(val_songs))[: args.held_songs]]

    paths = VBPM(input_dim=8).to(device)
    train_w = [w for s in train_songs for w in windows(s, frames, args.windows, rng)]
    held_w = [w for s in val_songs for w in windows(s, frames, args.windows, rng)]
    print(f"windows: train {len(train_w)}  held-out {len(held_w)}", flush=True)

    train_in = replay(paths, [d for d, _, _ in train_w], device)
    train_cls = torch.cat([c for _, c, _ in train_w]).to(device)
    held_in = replay(paths, [d for d, _, _ in held_w], device)
    held_cls = torch.cat([c for _, c, _ in held_w]).to(device)

    held_paths = paths.draws_to_paths(
        {k: torch.cat([d[k] for d, _, _ in held_w[:64]]).to(device) for k in held_w[0][0]},
        torch.ones(min(64, len(held_w)), frames, device=device),
    )
    beats = held_paths["is_beat"]
    labelled = held_cls[: len(beats)] > 0
    near = torch.nn.functional.max_pool1d(labelled.float()[:, None], 5, 1, 2)[:, 0] > 0
    print(
        f"oracle replay: {float((beats & near).sum() / beats.sum().clamp(min=1)):.3f} of path "
        f"beats within 2 frames of a label, {int(beats.sum())} path beats vs "
        f"{int(labelled.sum())} labels",
        flush=True,
    )

    freq = torch.bincount(train_cls.flatten(), minlength=3).float() / train_cls.numel()

    def fresh():
        emission = EmissionModel(EmissionSpec(), METERS, 0).to(device)
        if args.init == "random":
            with torch.no_grad():
                torch.nn.init.kaiming_uniform_(emission.out.weight, a=math.sqrt(5))
                emission.out.bias.copy_(freq.log())
        return emission

    print(f"teacher-forced emission, {args.init} output init", flush=True)
    teacher = train(fresh(), train_in, train_cls, args.steps, args.batch, args.seed)
    print("shuffled-path control", flush=True)
    shuffled = [torch.roll(x, 1, 0) for x in train_in]
    control = train(fresh(), shuffled, train_cls, args.steps, args.batch, args.seed)
    prior_only = float(freq.log()[held_cls].mean())
    truth = recon(teacher, held_in, held_cls)
    rows = [
        ("class frequencies only", prior_only),
        ("shuffled-path control", recon(control, held_in, held_cls)),
        ("TRUTH", truth),
    ]
    for name in ("half-beat shift", "tempo x2", "tempo x1/2", "wrong meter"):
        wrong_in = replay(paths, [perturb(d, m, name) for d, _, m in held_w], device)
        rows.append((name, recon(teacher, wrong_in, held_cls)))
    print("\nheld-out recon per frame (teacher-forced emission unless noted)")
    for name, value in rows:
        print(f"  {name:24s} {value:8.4f}   truth margin {truth - value:+.4f}")


if __name__ == "__main__":
    main()
