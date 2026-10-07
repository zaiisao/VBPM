"""Cache unseen real windows for the staged GSNN, using fold-matched features."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vbpm.data.dataset import load_beat_this
from vbpm.data.excerpts import ExcerptDataset, collate_excerpts
from vbpm.frontends import build_frontend
from vbpm.util.oracle import bar_meters, beat_positions, oracle_draw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/beat_this"))


def main():
    """Run the command-line tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--context-seconds", type=float, default=256 / 50)
    parser.add_argument(
        "--save-context-features",
        action="store_true",
        help="Also preserve full audio-only frontend context for proposal diagnostics",
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--exclude-cache", type=Path, action="append", default=[])
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=256 / 50,
        help="Generator target duration; distinct from frontend context duration",
    )
    args = parser.parse_args()
    frames = round(args.window_seconds * 50)
    if frames < 2:
        parser.error("Generator windows need at least two frames")
    if (
        args.output.exists()
        or args.count < 1
        or args.context_seconds < args.window_seconds
        or args.feature_batch_size < 1
    ):
        parser.error(
            "Fresh output, positive count/batch size, and context at least 5.12 seconds required"
        )
    torch.set_num_threads(1)
    excluded = set()
    for path in args.exclude_cache:
        excluded.update(torch.load(path, weights_only=True)["songs"])
    checkpoint = Path.home() / f".cache/torch/hub/checkpoints/beat_this-final{args.fold}.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    frontend = build_frontend("beat_this", checkpoint=str(checkpoint), device=f"cuda:{args.gpu}")
    data = load_beat_this(args.fold)
    source = data.train_dataset if args.split == "train" else data.val_dataset
    excerpts = ExcerptDataset(source, frontend, args.window_seconds, centered=True)
    contexts = ExcerptDataset(source, frontend, args.context_seconds, centered=True)
    raws, context_raws, offsets, ids, draws, starts = [], [], [], [], [], []
    for i, (song, down, beats) in enumerate(excerpts.items):
        sid = str(song["spect_path"])
        if sid in excluded or int(np.max(song["beat_value"])) != 4:
            continue
        raw = excerpts[i]
        if (
            raw["cls"].shape != (frames,)
            or not bool((raw["mask"] == 1).all())
            or len(raw["downbeat_times"]) < 1
        ):
            continue
        start = round(float(raw["t0"]) * 50)
        positions = beat_positions(beats, down)
        meters = bar_meters(positions)
        if not set(meters).issubset({None, 3, 4}):
            continue
        try:
            draw, _ = oracle_draw(np.asarray(beats) * 50 - start, positions, meters, frames)
        except ValueError:
            # An excerpt without a preceding closed annotation bar cannot
            # provide the scoring-only phase reference for this diagnostic.
            continue
        if not bool((draw["meter"].argmax(-1) == 1).all()):
            continue
        context = contexts[i]
        offset = start - round(float(context["t0"]) * 50)
        if (
            offset < 0
            or offset + frames > len(context["mask"])
            or not bool((context["mask"][offset : offset + frames] == 1).all())
        ):
            raise RuntimeError("Target window must be fully observed within frontend context")
        context_raws.append(context)
        offsets.append(offset)
        raws.append(raw)
        ids.append(sid)
        draws.append(draw)
        starts.append(start)
        if len(raws) == args.count:
            break
    if len(raws) != args.count:
        raise RuntimeError(f"Only {len(raws)} eligible fixed-meter windows; requested {args.count}")
    collated = collate_excerpts(raws)
    features = []
    full_features, full_masks, full_labels = [], [], []
    with torch.no_grad():
        for left in range(0, len(raws), args.feature_batch_size):
            context_batch = collate_excerpts(context_raws[left : left + args.feature_batch_size])
            context_h = frontend.forward_features(context_batch["input"]).cpu()
            if args.save_context_features:
                full_labels.extend(context_batch["cls"].cpu())
            for row, context_mask, offset in zip(
                context_h, context_batch["mask"], offsets[left : left + args.feature_batch_size]
            ):
                features.append(row[offset : offset + frames])
                if args.save_context_features:
                    full_features.append(row)
                    full_masks.append(context_mask.cpu())
    h = torch.stack(features)
    draw = {k: torch.cat([d[k] for d in draws]) for k in draws[0]}
    phase = torch.cat(
        (draw["phase0"][:, None], draw["phase0"][:, None] + draw["velocity"][:, :-1].cumsum(1)), 1
    )
    result = dict(
        h=h,
        labels=collated["cls"],
        mask=collated["mask"],
        phi=phase,
        velocity=draw["velocity"],
        meter=draw["meter"],
        songs=ids,
        start_frames=starts,
        beat_times=[torch.as_tensor(r["beat_times"] - r["t0"], dtype=torch.float64) for r in raws],
        downbeat_times=[
            torch.as_tensor(r["downbeat_times"] - r["t0"], dtype=torch.float64) for r in raws
        ],
    )
    if args.save_context_features:
        result.update(
            full_context_h=torch.stack(full_features),
            full_context_mask=torch.stack(full_masks),
            full_context_labels=torch.stack(full_labels),
            context_offsets=offsets,
        )
    args.output.mkdir(parents=True)
    torch.save(result, args.output / "batch.pt")
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                split=args.split,
                fold=args.fold,
                checkpoint=str(checkpoint),
                songs=ids,
                frames=frames,
                fps=50,
                meter=4,
                window_seconds=args.window_seconds,
                save_context_features=args.save_context_features,
                context_seconds=args.context_seconds,
                feature_batch_size=args.feature_batch_size,
                context_offsets=offsets,
                excluded_cache_paths=[str(p) for p in args.exclude_cache],
                selection=(
                    "First eligible centered 4/4 windows with globally supported 3/4 o"
                    "r 4/4 annotation bars, in source order; excluding all four traine"
                    "d recordings; no model score used"
                ),
                annotation_usage=(
                    "Selection of fixed-meter cohort and scoring targets only; never f"
                    "eature extraction or prior input"
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(dict(status="complete", output=str(args.output), shape=list(h.shape), songs=ids))
    )


if __name__ == "__main__":
    main()
