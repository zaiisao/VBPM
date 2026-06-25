"""KNOWN-ANSWER TEST of the deploy-path read-out + mir_eval.

Synthesize the IDEAL bar-pointer latent from ground-truth annotations and run it through the
SAME geometric read-out the model uses (faithful.evaluate.beats_from_barphase /
downbeats_from_barphase) and the SAME mir_eval scorer. If ideal latents don't score ~1.0,
the read-out or eval pipeline is broken and every reported free-run beat-F is suspect.

Two ideal constructions:
  (B) PURE read-out test: global phase Phi increases 2*pi per bar and passes through
      Phi = 2*pi*bar + 2*pi*k/m at EVERY annotated beat (k = beat index within bar);
      phi = Phi % 2*pi. By construction the read-out should recover the exact annotations.
  (A) REALISM test: phi ramps linearly 0->2*pi between consecutive downbeats only
      (steady tempo within a bar). Tests how much within-bar tempo drift the read-out costs.
"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.data import FPS, iter_val_songs
from faithful.evaluate import beats_from_barphase, downbeats_from_barphase, f_measure

TWO_PI = 2.0 * math.pi
ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]


def ideal_phase_through_beats(beat_f, down_f, m, T):
    """(B) piecewise-linear GLOBAL phase through every beat; returns phi=Phi%2pi over [0,T)."""
    downs = np.sort(down_f)
    beats = np.sort(beat_f)
    anchors_t, anchors_phi = [], []
    for b in beats:
        bar = int(np.searchsorted(downs, b, side="right") - 1)   # which bar this beat is in
        if bar < 0:
            continue
        k = int(round((b - downs[bar]) / max(1e-9, (downs[min(bar + 1, len(downs) - 1)] - downs[bar])) * m)) if len(downs) > bar + 1 else 0
        anchors_t.append(b); anchors_phi.append(TWO_PI * bar + TWO_PI * (k % m) / m)
    # ensure each downbeat is an exact integer-multiple anchor
    for i, d in enumerate(downs):
        anchors_t.append(d); anchors_phi.append(TWO_PI * i)
    order = np.argsort(anchors_t)
    at = np.asarray(anchors_t)[order]; ap = np.asarray(anchors_phi)[order]
    # dedup identical times (keep first), enforce monotone phase
    keep_t, keep_p = [], []
    for t, p in zip(at, ap):
        if keep_t and abs(t - keep_t[-1]) < 1e-6:
            continue
        if keep_p and p < keep_p[-1]:
            p = keep_p[-1]
        keep_t.append(t); keep_p.append(p)
    frames = np.arange(T)
    Phi = np.interp(frames / FPS, keep_t, keep_p, left=keep_p[0], right=keep_p[-1])
    return Phi % TWO_PI


def ideal_phase_linear_bars(down_f, T):
    """(A) phi ramps 0->2pi linearly between consecutive downbeats; phi=Phi%2pi."""
    downs = np.sort(down_f)
    frames = np.arange(T)
    Phi = np.interp(frames / FPS, downs, TWO_PI * np.arange(len(downs)),
                    left=0.0, right=TWO_PI * (len(downs) - 1))
    return Phi % TWO_PI


def main():
    rowsB, rowsA = [], []
    n = 0
    for key, audio, beats, downs, meta in iter_val_songs(ROOT, DS, max_per_dataset=4):
        T = min(len(beats), 4000)
        bf = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
        df = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
        if len(bf) < 8 or len(df) < 3:
            continue
        # meter from GT: median beats per bar
        bpb = np.median([np.sum((bf >= df[i]) & (bf < df[i + 1])) for i in range(len(df) - 1)])
        m = int(round(bpb)) if bpb >= 2 else 4
        m = max(2, min(m, 4))
        n += 1
        # (B) pure read-out
        phiB = ideal_phase_through_beats(bf, df, m, T)
        bB = beats_from_barphase(phiB, m, FPS)
        dB = downbeats_from_barphase(phiB, FPS)
        rowsB.append((f_measure(bf, bB), f_measure(df, dB), m, len(bf), len(bB)))
        # (A) linear-between-downbeats
        phiA = ideal_phase_linear_bars(df, T)
        bA = beats_from_barphase(phiA, m, FPS)
        dA = downbeats_from_barphase(phiA, FPS)
        rowsA.append((f_measure(bf, bA), f_measure(df, dA)))

    B = np.array([r[:2] for r in rowsB], float)
    A = np.array(rowsA, float)
    print(f"songs={n}\n")
    print("=== (B) PURE READ-OUT KNOWN-ANSWER (ideal phase through every beat) ===")
    print(f"  beat-F     mean={np.nanmean(B[:,0]):.4f}  min={np.nanmin(B[:,0]):.3f}")
    print(f"  downbeat-F mean={np.nanmean(B[:,1]):.4f}  min={np.nanmin(B[:,1]):.3f}")
    print(f"  EXPECTED ~1.0 for both. If not -> read-out/eval is BROKEN.\n")
    print("=== (A) REALISM (phi linear between downbeats; within-bar steady tempo) ===")
    print(f"  beat-F     mean={np.nanmean(A[:,0]):.4f}")
    print(f"  downbeat-F mean={np.nanmean(A[:,1]):.4f}\n")
    print("per-song (B): beatF downbeatF  m  n_ref n_est")
    for r in rowsB:
        print(f"   {r[0]:.3f}  {r[1]:.3f}   m={r[2]}  ref={r[3]} est={r[4]}")


if __name__ == "__main__":
    main()
