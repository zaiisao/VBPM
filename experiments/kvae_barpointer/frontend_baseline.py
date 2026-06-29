"""Frontend baselines on the EXACT M1 val set (same load() as kvae_run): the frozen feature
extractor's own output act2 [T,2] = (beat, downbeat) probabilities, scored (a) with simple
peak-picking (no DBN) and (b) with madmom's DBN -- so M1's KVAE number has apples-to-apples refs.
"""
import sys, glob, random, importlib.util
import numpy as np
import torch

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
peaks, fmeas, FPS = da.peaks, da.fmeas, da.FPS

from madmom.features.downbeats import DBNDownBeatTrackingProcessor


def load(cd, n, seed):  # EXACT replica of kvae_run.load
    fs = sorted(glob.glob(f"{cd}/*.pt")); random.Random(seed).shuffle(fs); out = []
    for f in fs[:n]:
        d = torch.load(f, map_location="cpu")
        if d["activations"].shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        out.append((d["act2"].float().numpy(), d["beat_targets"].numpy(), d["downbeat_targets"].numpy()))
    return out


def main():
    val = load("cache/acts/bt_val_rich", 40, 2)
    print(f"M1 val set: {len(val)} songs | fps={FPS:.4f}", flush=True)

    nb, nd = [], []          # no-DBN peak-pick
    db_b, db_d = [], []      # madmom DBN
    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS, min_bpm=55, max_bpm=215)

    for act2, bt, dbt in val:
        T = len(act2)
        ref_b = np.where(bt[:T] > 0.5)[0] / FPS
        ref_d = np.where(dbt[:T] > 0.5)[0] / FPS

        # (a) no DBN: peak-pick the activation channels
        if len(ref_b) >= 2: nb.append(fmeas(ref_b, peaks(act2[:, 0])))
        if len(ref_d) >= 2: nd.append(fmeas(ref_d, peaks(act2[:, 1])))

        # (b) madmom DBN. madmom wants [P(beat-not-downbeat), P(downbeat)] with row sum <= 1,
        # but our beat channel includes downbeats -> split it: col0 = beat minus downbeat.
        beat_nd = np.clip(act2[:, 0] - act2[:, 1], 1e-6, 1 - 1e-6)
        dbeat = np.clip(act2[:, 1], 1e-6, 1 - 1e-6)
        act = np.stack([beat_nd, dbeat], axis=1).astype(np.float64)
        try:
            est = dbn(act)                   # [N,2] = (time_sec, beat_position_in_bar)
            est_b = est[:, 0]
            est_d = est[est[:, 1] == 1, 0]
            if len(ref_b) >= 2: db_b.append(fmeas(ref_b, est_b))
            if len(ref_d) >= 2: db_d.append(fmeas(ref_d, est_d))
        except Exception as e:
            print("  DBN skip:", e)

    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    print("\n=== FROZEN FEATURE EXTRACTOR on the M1 val set ===")
    print(f"  no DBN (peak-pick act2) : beat {m(nb):.3f}  downbeat {m(nd):.3f}")
    print(f"  + madmom DBN            : beat {m(db_b):.3f}  downbeat {m(db_d):.3f}")
    print("  (M1 KVAE filter deploy   : beat ~0.84  downbeat ~0.83)")


if __name__ == "__main__":
    main()
