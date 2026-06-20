"""Ceiling check: peak-pick the RAW cached frontend activations (the same tensors
fed to CHART as h_t) and score beat/downbeat F with the EXACT functions the CHART
free-run eval uses. If this scores high while CHART free-run is ~0, the gap is
CHART's free-run collapse, not a broken cache."""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.phase_converter import extract_beat_timestamps
from evaluation.score import evaluate_beats, evaluate_downbeats, frames_to_beat_times


def main() -> int:
    cache = sys.argv[1] if len(sys.argv) > 1 else "cache/acts/bt_val"
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = all
    fs = sorted(glob.glob(str(Path(cache) / "**" / "*.pt"), recursive=True))
    if maxn:
        fs = fs[:maxn]
    bF, bC, bA, dF = [], [], [], []
    n = 0
    for f in fs:
        r = torch.load(f, map_location="cpu")
        a = r["activations"]
        fps = float(r["fps"])
        bt = r["beat_targets"].cpu().numpy()
        db = r["downbeat_targets"].cpu().numpy()
        ref = frames_to_beat_times(bt, fps)
        if len(ref) < 2:
            continue
        # peak-pick the frontend's own probability channels — identical peak picker
        # to the one applied to CHART's decoder output in _heldout_freerun.
        est = extract_beat_timestamps(a[:, 0].cpu().numpy(), fps=fps)
        sb = evaluate_beats(ref, est)
        bF.append(sb["F-measure"]); bC.append(sb["CMLt"]); bA.append(sb["AMLt"])
        ref_db = frames_to_beat_times(db, fps)
        if len(ref_db) >= 2:
            est_db = extract_beat_timestamps(a[:, 1].cpu().numpy(), fps=fps)
            sd = evaluate_downbeats(ref_db, est_db)
            dF.append(sd["db_F-measure"])
        n += 1

    def m(x):
        return float(np.mean(x)) if x else float("nan")
    print(f"[peakpick ceiling] cache={cache}  songs={n}")
    print(f"  BEAT     F={m(bF):.3f}  CMLt={m(bC):.3f}  AMLt={m(bA):.3f}  (n={len(bF)})")
    print(f"  DOWNBEAT F={m(dF):.3f}  (n={len(dF)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
