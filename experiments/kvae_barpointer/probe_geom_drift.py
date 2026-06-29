"""DECISIVE: given a PERFECT per-song tempo, does the GEOMETRIC read-out (phi = integral of tempo, +best
offset) actually recover beats -- or does integrating a constant tempo DRIFT out of phase? No learning.
Compare phi from: (a) autocorr tempo (self-sup, 1.00-accurate), (b) GT median tempo, (c) GT full phase.
If even (b) with best offset is low -> integration drift is the wall (need per-frame phase correction=DBN).
"""
import sys, math, importlib.util
import numpy as np, torch
ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
load_pool, fmeas, phase_beats = da.load_pool, da.fmeas, da.phase_beats
FPS = 86.1328125; M = 4; TWO_PI = 2 * math.pi


def autocorr_period(beat_act, min_bpm=55, max_bpm=200):
    a = np.asarray(beat_act, float); a = a - a.mean()
    if a.std() < 1e-6: return None
    ac = np.correlate(a, a, "full")[len(a) - 1:]
    lo = int(60 * FPS / max_bpm); hi = int(60 * FPS / min_bpm)
    seg = ac[lo:hi + 1]
    return (lo + int(np.argmax(seg))) if len(seg) >= 2 else None


def geom_beatF(period, T, ref, best_offset=True):
    adv = TWO_PI / (M * period)
    base = (np.arange(T) * adv)
    best = 0.0
    offs = np.linspace(0, TWO_PI, 24, endpoint=False) if best_offset else [0.0]
    for off in offs:
        phi = (base + off) % TWO_PI
        best = max(best, fmeas(ref, phase_beats(phi, M)))
    return best


def main():
    val = load_pool("cache/acts/bt_val_rich", 40, seed=2)
    # need act2 for autocorr; load_pool may not include it -> load raw
    import glob
    fs = sorted(glob.glob("cache/acts/bt_val_rich/*.pt"))
    auto, gtm, full = [], [], []
    cnt = 0
    for f in fs:
        if cnt >= 40: break
        d = torch.load(f, map_location="cpu")
        if d["activations"].shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        cnt += 1
        T = min(d["activations"].shape[0], 1600)
        bt = d["beat_targets"].numpy()[:T]; bf = np.where(bt > 0.5)[0]
        if len(bf) < 3: continue
        ref = bf / FPS
        # (a) autocorr tempo
        p_auto = autocorr_period(d["act2"][:T, 0].numpy())
        if p_auto: auto.append(geom_beatF(p_auto, T, ref))
        # (b) GT median tempo
        p_gt = float(np.median(np.diff(bf))); gtm.append(geom_beatF(p_gt, T, ref))
        # (c) GT full phase (piecewise-linear ramp through actual beats) = the clamp ceiling
        phi = np.zeros(T); vals = np.arange(len(bf)) * (TWO_PI / M)
        for k in range(len(bf) - 1):
            phi[bf[k]:bf[k + 1]] = np.linspace(vals[k], vals[k + 1], bf[k + 1] - bf[k], endpoint=False)
        phi[bf[-1]:] = vals[-1]; full.append(fmeas(ref, phase_beats(phi % TWO_PI, M)))
    m = lambda x: float(np.nanmean(x))
    print(f"GEOMETRIC read-out given a PERFECT tempo (best offset), {len(gtm)} songs:")
    print(f"  (a) integral of AUTOCORR tempo (self-sup) : beat {m(auto):.3f}")
    print(f"  (b) integral of GT median tempo           : beat {m(gtm):.3f}")
    print(f"  (c) GT full phase ramp (clamp ceiling)    : beat {m(full):.3f}")
    print("  IF (b) << (c): integrating a CONSTANT tempo DRIFTS -> geometric-from-tempo is the wall (need DBN phase correction)")
    print("  IF (b) ~ (c):  integration is fine -> the only problem is grounding the tempo")


if __name__ == "__main__":
    main()
