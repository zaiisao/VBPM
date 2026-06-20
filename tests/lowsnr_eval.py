"""Low-SNR repositioning test (Ng & Jordan crossover). Degrade the cached frontend
activations with additive noise and compare, as a function of noise level:
  - DISCRIMINATIVE peak-pick of the degraded activations (should crater as SNR drops)
  - CHART's particle filter (closed-loop bar-pointer prior; should be ROBUST because
    the tempo/phase-continuity dynamics reject spurious noise peaks via resampling).

If the structured prior is worth anything, peak-pick crosses BELOW CHART-PF as noise
rises -- the prior's value is at low SNR, exactly as the diagnosis predicted.

NOTE: degrading the activations is a fast proxy for "noisy audio -> noisier frontend";
the faithful version re-runs the frontend on degraded AUDIO (heavier; pf_eval_smc).

    python tests/lowsnr_eval.py cache/diag/ckpt2.pt cache/acts/bt_val 16 300
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
dg = _load("dg", str(Path(__file__).resolve().parent / "diag_freerun.py"))
from evaluation.phase_converter import extract_beat_timestamps, extract_beats_from_phase_trajectory
from evaluation.score import evaluate_beats, evaluate_downbeats, frames_to_beat_times


def bF(ref, est):
    return evaluate_beats(ref, est)["F-measure"] if len(ref) >= 2 and len(est) >= 1 else 0.0

def dF(ref, est):
    return evaluate_downbeats(ref, est)["db_F-measure"] if len(ref) >= 2 and len(est) >= 1 else 0.0


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "cache/diag/ckpt2.pt"
    cache = sys.argv[2] if len(sys.argv) > 2 else "cache/acts/bt_val"
    nsongs = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    npart = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    maxT = 1024
    levels = ([float(x) for x in sys.argv[5].split(",")]
              if len(sys.argv) > 5 else [0.0, 0.1, 0.2, 0.35, 0.5, 0.7])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, a = dg.build_from_ckpt(ckpt, dev)
    fs = sorted(glob.glob(str(Path(cache) / "**" / "*.pt"), recursive=True))[:nsongs]
    print(f"[lowsnr] ckpt={ckpt} songs={len(fs)} n_particles={npart}")
    print(f"{'noise':>6} | {'peakpick beat':>13} {'CHART-PF beat':>13} | {'peakpick db':>11} {'CHART-PF db':>11}")

    for sigma in levels:
        pp_b, pf_b, pp_d, pf_d = [], [], [], []
        g = torch.Generator(device=dev).manual_seed(0)   # same noise draw across levels' songs
        for f in fs:
            r = torch.load(f, map_location=dev)
            act = r["activations"][:maxT]
            fps = float(r["fps"])
            ref = frames_to_beat_times(r["beat_targets"][:maxT].cpu().numpy(), fps)
            ref_db = frames_to_beat_times(r["downbeat_targets"][:maxT].cpu().numpy(), fps)
            if len(ref) < 3:
                continue
            noise = torch.randn(act.shape, generator=g, device=dev) * sigma
            act_n = (act + noise).clamp(0.0, 1.0)
            # discriminative peak-pick of the degraded activations
            pp_b.append(bF(ref, extract_beat_timestamps(act_n[:, 0].cpu().numpy(), fps=fps)))
            if len(ref_db) >= 2:
                pp_d.append(dF(ref_db, extract_beat_timestamps(act_n[:, 1].cpu().numpy(), fps=fps)))
            # CHART particle filter on the degraded activations
            with torch.no_grad():
                p = model.sample_from_prior_pf(act_n.unsqueeze(0), n_particles=npart, obs_sigma=0.3, temperature=0.1)
            pf_b.append(bF(ref, extract_beats_from_phase_trajectory(p["phase_mu"][0].cpu().numpy(), fps=fps)))
            if p.get("bar_phase_mu") is not None and len(ref_db) >= 2:
                pf_d.append(dF(ref_db, extract_beats_from_phase_trajectory(p["bar_phase_mu"][0].cpu().numpy(), fps=fps)))

        def mn(x):
            return float(np.mean(x)) if x else float("nan")
        star = "  <-- PF wins" if mn(pf_b) > mn(pp_b) else ""
        print(f"{sigma:>6.2f} | {mn(pp_b):>13.3f} {mn(pf_b):>13.3f} | {mn(pp_d):>11.3f} {mn(pf_d):>11.3f}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
