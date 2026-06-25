"""KNOWN-ANSWER verification of the faithful/evaluate.py fix.

Builds ideal bar-pointer latents from GT (phase through every beat) and runs them through
the SAME read-out logic the fixed evaluate() uses: faithful.evaluate._estimate_meter +
beats_from_barphase (beat_phase) + downbeats_from_barphase (downbeat_phase) + f_measure.
If the fix is correct, beat_phase and downbeat_phase both score ~0.97 (NOT ~0.3)."""
import sys, math
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.data import FPS, iter_val_songs
import faithful.evaluate as ev

TWO_PI = 2.0 * math.pi
ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]


def ideal_phase_through_beats(bf, df, m, T):
    downs = np.sort(df); beats = np.sort(bf)
    at, ap = [], []
    for b in beats:
        bar = int(np.searchsorted(downs, b, side="right") - 1)
        if bar < 0:
            continue
        nb = downs[min(bar + 1, len(downs) - 1)] - downs[bar]
        k = int(round((b - downs[bar]) / max(1e-9, nb) * m)) if len(downs) > bar + 1 else 0
        at.append(b); ap.append(TWO_PI * bar + TWO_PI * (k % m) / m)
    for i, d in enumerate(downs):
        at.append(d); ap.append(TWO_PI * i)
    order = np.argsort(at); at = np.asarray(at)[order]; ap = np.asarray(ap)[order]
    kt, kp = [], []
    for t, p in zip(at, ap):
        if kt and abs(t - kt[-1]) < 1e-6:
            continue
        if kp and p < kp[-1]:
            p = kp[-1]
        kt.append(t); kp.append(p)
    Phi = np.interp(np.arange(T) / FPS, kt, kp, left=kp[0], right=kp[-1])
    return Phi % TWO_PI


bp, dp = [], []
for key, audio, beats, downs, meta in iter_val_songs(ROOT, DS, max_per_dataset=4):
    T = min(len(beats), 4000)
    bf = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
    df = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
    if len(bf) < 8 or len(df) < 3:
        continue
    m = ev._estimate_meter(bf, df)                 # <-- the exact helper evaluate() uses
    phi = ideal_phase_through_beats(bf, df, m, T)
    bp.append(ev.f_measure(bf, ev.beats_from_barphase(phi, m, FPS)))
    dp.append(ev.f_measure(df, ev.downbeats_from_barphase(phi, FPS)))

print(f"songs={len(bp)}")
print(f"beat_phase     mean={np.nanmean(bp):.4f}  min={np.nanmin(bp):.3f}   EXPECT ~0.97")
print(f"downbeat_phase mean={np.nanmean(dp):.4f}  min={np.nanmin(dp):.3f}   EXPECT ~0.97")
ok = np.nanmean(bp) > 0.9 and np.nanmean(dp) > 0.9
print(f"VERIFY: {'PASS' if ok else 'FAIL'}")
