"""PF-vs-mean read-out on CACHED activations. The diagnosis measured the
deterministic MEAN free-run (sample_from_prior) = audio-blind. This runs the
audio-aware particle filter (sample_from_prior_pf, Dir-1B) on the SAME cached
val activations and compares -- does Bayesian resampling against the emission
likelihood fix the audio-blind drift, keeping the bar-pointer prior exact?

    python tests/pf_eval_cache.py cache/diag/ckpt2.pt cache/acts/bt_val 16 300
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
    obs_sigma = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    maxT = 1024
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, a = dg.build_from_ckpt(ckpt, dev)
    print(f"[pf-cache] ckpt={ckpt} n_particles={npart} obs_sigma={obs_sigma}")

    acc = {k: [] for k in ("mean_phaseF", "mean_decF", "mean_barwrapDB",
                            "pf_phaseF", "pf_decF", "pf_actF", "pf_barwrapDB")}
    fs = sorted(glob.glob(str(Path(cache) / "**" / "*.pt"), recursive=True))[:nsongs]
    with torch.no_grad():
        for f in fs:
            r = torch.load(f, map_location=dev)
            act = r["activations"][:maxT].unsqueeze(0).float()   # fp16 rich cache -> fp32
            fps = float(r["fps"])
            ref = frames_to_beat_times(r["beat_targets"][:maxT].cpu().numpy(), fps)
            ref_db = frames_to_beat_times(r["downbeat_targets"][:maxT].cpu().numpy(), fps)
            if len(ref) < 3:
                continue
            # ---- MEAN free-run (audio-blind) ----
            m = model.sample_from_prior(act, temperature=0.1)
            acc["mean_phaseF"].append(bF(ref, extract_beats_from_phase_trajectory(m["phase_mu"][0].cpu().numpy(), fps=fps)))
            acc["mean_decF"].append(bF(ref, extract_beat_timestamps(torch.sigmoid(m["beat_logits"][0, :, 0]).cpu().numpy(), fps=fps)))
            if m.get("bar_phase_mu") is not None and len(ref_db) >= 2:
                acc["mean_barwrapDB"].append(dF(ref_db, extract_beats_from_phase_trajectory(m["bar_phase_mu"][0].cpu().numpy(), fps=fps)))
            # ---- PARTICLE FILTER (audio-aware) ----
            p = model.sample_from_prior_pf(act, n_particles=npart, obs_sigma=obs_sigma, temperature=0.1)
            acc["pf_phaseF"].append(bF(ref, extract_beats_from_phase_trajectory(p["phase_mu"][0].cpu().numpy(), fps=fps)))
            acc["pf_decF"].append(bF(ref, extract_beat_timestamps(torch.sigmoid(p["beat_logits"][0, :, 0]).cpu().numpy(), fps=fps)))
            acc["pf_actF"].append(bF(ref, extract_beat_timestamps(p["beat_activation"][0].cpu().numpy(), fps=fps)))
            if p.get("bar_phase_mu") is not None and len(ref_db) >= 2:
                acc["pf_barwrapDB"].append(dF(ref_db, extract_beats_from_phase_trajectory(p["bar_phase_mu"][0].cpu().numpy(), fps=fps)))

    def mn(k):
        return float(np.mean(acc[k])) if acc[k] else float("nan")
    print(f"\n[pf-cache] over {len(acc['pf_decF'])} songs\n")
    print(f"  MEAN free-run (audio-BLIND):  beat phase-wrap={mn('mean_phaseF'):.3f}  decoder={mn('mean_decF'):.3f}  downbeat barwrap={mn('mean_barwrapDB'):.3f}")
    print(f"  PARTICLE FILTER (audio-aware): beat phase-wrap={mn('pf_phaseF'):.3f}  decoder={mn('pf_decF'):.3f}  activation={mn('pf_actF'):.3f}  downbeat barwrap={mn('pf_barwrapDB'):.3f}")
    print(f"\n  (ceiling: peak-pick the activations = 0.94 beat / 0.90 downbeat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
