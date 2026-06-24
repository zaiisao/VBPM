"""Per-song ORACLE transition-lambda: the upper bound of lambda-adaptation.

For each held-out SMC track, sweep lambda (incl ~0) with the FIXED madmom emission, take
the best beat-F per song. If oracle-per-song >> best-uniform-lambda, per-song adaptation
has real headroom (precompute-supervise or richer features could exploit it). If they're
~equal, the end-to-end gradient (which collapsed to a uniform low lambda) already captured
everything and there's nothing per-song to gain.

    python tests/dbn_lambda_oracle.py
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.bar_pointer_dbn import BarPointerDBN

SMC = "/home/sogang/jaehoon/Analyze-SMC"
RICH = "/home/sogang/jaehoon/CHART/cache/acts/smc_rich_heldout"
FPS = 50.0
LAMBDAS = [0.05, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def _peakpick(prob, thresh=0.5, width=7):
    t = torch.from_numpy(np.ascontiguousarray(prob)).float().unsqueeze(0)
    peaks = t.masked_fill(t != F.max_pool1d(t, width, 1, width // 2), -1000.0)
    fr = torch.nonzero(peaks.squeeze(0) > thresh).numpy()[:, 0]
    if len(fr):
        keep = [fr[0]]
        for x in fr[1:]:
            if x - keep[-1] > 1:
                keep.append(x)
        fr = np.array(keep)
    return fr / FPS


def _load():
    GTd = SMC + "/beat_this_annotations/smc/annotations/beats"
    data = {}
    for f in sorted(glob.glob(RICH + "/*.pt")):
        r = torch.load(f, map_location="cpu"); tid = r["tid"]
        gt = None
        for nm in (tid, tid.upper()):
            p = os.path.join(GTd, nm + ".beats")
            if os.path.exists(p):
                d = np.loadtxt(p); gt = d if d.ndim == 1 else d[:, 1]; break
        if gt is None or len(gt) < 2:
            continue
        data[tid] = {"act2": r["act2"].float(), "gt": gt}
    return data


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load(); tids = sorted(data)
    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=None, learnable_lambda=False).to(dev)
    Fm = lambda gt, fr: mir_eval.beat.evaluate(gt, fr.cpu().numpy().astype(float) / FPS)["F-measure"]
    print(f"[oracle] {len(tids)} tracks, lambda sweep {LAMBDAS}", flush=True)

    log_l = {l: torch.tensor([float(np.log(l))], device=dev) for l in LAMBDAS}
    elps = {l: dbn._edge_logp(log_lambda=log_l[l]) for l in LAMBDAS}
    pk, per_lambda, oracle, best_lam = [], {l: [] for l in LAMBDAS}, [], []
    with torch.no_grad():
        for t in tids:
            d = data[t]; a2 = d["act2"].to(dev); obs = dbn.observation_logp(a2)
            pk.append(mir_eval.beat.evaluate(d["gt"], _peakpick(d["act2"][:, 0].numpy()))["F-measure"])
            bestF, bestL = -1.0, None
            for l in LAMBDAS:
                path = dbn._viterbi(obs, elp=elps[l])
                bfr, _ = dbn._path_to_beats(path, beat_snap=a2[:, 0])
                f = Fm(d["gt"], bfr)
                per_lambda[l].append(f)
                if f > bestF:
                    bestF, bestL = f, l
            oracle.append(bestF); best_lam.append(bestL)

    print(f"\n[oracle] peak-pick                 = {np.mean(pk):.4f}")
    for l in LAMBDAS:
        print(f"[oracle] fixed lambda={l:<6g}        = {np.mean(per_lambda[l]):.4f}")
    bl = max(LAMBDAS, key=lambda l: np.mean(per_lambda[l]))
    print(f"\n[oracle] BEST-UNIFORM lambda ({bl:g})     = {np.mean(per_lambda[bl]):.4f}")
    print(f"[oracle] PER-SONG ORACLE lambda      = {np.mean(oracle):.4f}   <- ceiling of lambda-adaptation")
    print(f"[oracle] headroom (oracle - best-uniform) = {np.mean(oracle) - np.mean(per_lambda[bl]):.4f}")
    vals, cnts = np.unique(best_lam, return_counts=True)
    print(f"[oracle] per-song best-lambda distribution: " +
          ", ".join(f"{v:g}:{c}" for v, c in zip(vals, cnts)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
