"""VALID OOD test that needs no re-extraction: run the structured DBN inference on SMC's OWN frontend
activations (smc_rich_heldout act2 = Beat-This beat/downbeat probs @ 50fps), vs peak-picking the same
activations. Tests whether our geometric DBN's tempo-phase inference helps OOD on a strong frontend.
ref: Beat-This noDBN 0.626 / +DBN 0.575 / madmom 0.570 (beat-F on SMC-MIREX).
"""
import sys, glob, os, importlib.util
import numpy as np, torch, mir_eval

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
from models.bar_pointer_dbn import BarPointerDBN
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANN = "/home/sogang/jaehoon/Analyze-SMC/smc_metadata/annotations"
FEAT = "cache/acts/smc_rich_heldout"; FPS = 50.0


def gt_beats(num):
    g = glob.glob(f"{ANN}/SMC_{num}_*.txt")
    if not g: return None
    a = np.loadtxt(g[0]); a = a[:, 0] if a.ndim > 1 else a
    a = np.sort(np.atleast_1d(a)[np.isfinite(np.atleast_1d(a))])
    return a if len(a) >= 2 else None


def peaks_fps(prob, fps, thr=0.5, md=0.10):
    p = np.asarray(prob); md = int(md * fps)
    cand = [t for t in range(1, len(p) - 1) if p[t] >= thr and p[t] >= p[t - 1] and p[t] >= p[t + 1]]
    out, last = [], -10 ** 9
    for t in cand:
        if t - last >= md: out.append(t); last = t
    return np.array(out, float) / fps


def amlt(ref, est):
    if len(ref) < 2 or len(est) < 2: return 0.0
    try: return float(mir_eval.beat.continuity(ref, est)[3])
    except Exception: return 0.0


@torch.no_grad()
def main():
    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=50).to(DEV)
    Fd, Fp, Ad, Ap = [], [], [], []
    for f in sorted(glob.glob(f"{FEAT}/*.pt")):
        r = torch.load(f, map_location="cpu"); num = r["tid"].split("_")[1]
        ref = gt_beats(num)
        if ref is None: continue
        act2 = r["act2"].float().to(DEV)
        beats = dbn.decode(act2)[0].cpu().numpy().astype(float) / FPS          # DBN inference
        peak = peaks_fps(act2[:, 0].cpu().numpy(), FPS)                         # peak-pick same act
        Fd.append(float(mir_eval.beat.f_measure(ref, beats))); Ad.append(amlt(ref, beats))
        Fp.append(float(mir_eval.beat.f_measure(ref, peak))); Ap.append(amlt(ref, peak))
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    print("==== SMC-MIREX OOD: structured DBN inference vs peak-pick on SMC's own Beat-This activations ====")
    print(f"  n={len(Fd)}")
    print(f"  Beat-This + our DBN : beat-F {m(Fd):.3f}  AMLt {m(Ad):.3f}")
    print(f"  Beat-This peak-pick : beat-F {m(Fp):.3f}  AMLt {m(Ap):.3f}")
    print(f"  (ref: BeatThis noDBN 0.626 / +DBN 0.575 / madmom 0.570)")
    print("SMC_FIXED_DONE")


if __name__ == "__main__":
    main()
