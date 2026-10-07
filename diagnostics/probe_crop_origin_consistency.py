"""Shift feature crops while comparing the same overlapping physical times."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from diagnostics.audio_generator import AudioPhaseFirstGSNN
from diagnostics.readouts import phase_events
from vbpm.scoring.evaluation import f_measure


def main():
    """Run the command-line tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--batch-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Fresh output required")
    torch.set_num_threads(1)
    b = torch.load(a.batch_cache, weights_only=True)
    full = b["full_context_h"]
    frames = b["h"].shape[1]
    offsets = b["context_offsets"]
    for i, offset in enumerate(offsets):
        assert torch.equal(full[i, offset : offset + frames], b["h"][i])
    results = []
    for seed in (0, 1):
        c = torch.load(a.model_dir / f"seed{seed}.pt", weights_only=True)
        assert c.get("bernoulli_clock_emission") and not c.get("phase_attention")
        model = AudioPhaseFirstGSNN(
            c["learn_tempo"],
            c["aligned_context"],
            c.get("proposal_max_bpm"),
            c.get("tempo_log_scale", 0.1),
            c.get("tempo_basis", "framewise"),
            c["smooth_concentration"],
            bernoulli_clock_emission=True,
            proposal_min_probability=c.get("proposal_min_probability"),
            tempo_feedback=c.get("tempo_feedback", False),
        )
        if c.get("temporal_phase_context"):
            model.enable_temporal_phase_context()
        model.periodic_context_only = c.get("periodic_context_only", False)
        model.recover_missing_beats = c.get("recover_missing_beats", False)
        model.load_state_dict(c["state"])
        model.eval()
        with torch.no_grad():
            base_phase, base_velocity = model.trajectory(b["h"])
            base_guess = model.prediction_for(b["h"])["phase"]
            rows = []
            for shift in (-100, -50, 0, 50, 100):
                h = torch.stack(
                    [
                        full[i, offset + shift : offset + shift + frames]
                        for i, offset in enumerate(offsets)
                    ]
                )
                phase, velocity = model.trajectory(h)
                guess = model.prediction_for(h)["phase"]
                lo, hi = max(0, shift), min(frames - 1, frames - 1 + shift)
                for i in range(len(h)):
                    difference = phase[i, lo - shift : hi - shift + 1] - base_phase[i, lo : hi + 1]
                    difference = torch.atan2(difference.sin(), difference.cos())
                    gd = guess[i, lo - shift : hi - shift + 1] - base_guess[i, lo : hi + 1]
                    gd = torch.atan2(gd.sin(), gd.cos())
                    events = phase_events(phase[i].numpy())
                    original_events = phase_events(base_phase[i].numpy())
                    actual, original = [], []
                    for e, be, truth in zip(
                        events, original_events, (b["beat_times"][i], b["downbeat_times"][i])
                    ):
                        e = e + shift / 50
                        truth = np.asarray(truth)

                        def keep(values):
                            return values[(values >= lo / 50) & (values <= hi / 50)]

                        actual.append(f_measure(keep(e), keep(truth))[0])
                        original.append(f_measure(keep(be), keep(truth))[0])
                    rows.append(
                        dict(
                            song=b["songs"][i],
                            crop_shift_frames=shift,
                            overlap_seconds=[lo / 50, hi / 50],
                            circular_phase_difference_MAE_deg=float(
                                difference.abs().mean() * 180 / math.pi
                            ),
                            proposal_phase_difference_MAE_deg=float(
                                gd.abs().mean() * 180 / math.pi
                            ),
                            beat_F1_70ms=actual[0],
                            downbeat_F1_70ms=actual[1],
                            original_overlap_beat_F1=original[0],
                            original_overlap_downbeat_F1=original[1],
                        )
                    )
        results.append(dict(seed=seed, windows=rows))
    a.output.write_text(
        json.dumps(
            dict(
                scope=(
                    "Same cached frontend features on overlap; crop-only perturbation;"
                    " references score overlap only, never generate phase"
                ),
                model_dir=str(a.model_dir),
                batch_cache=str(a.batch_cache),
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )
    for r in results:
        shifted = [x for x in r["windows"] if x["crop_shift_frames"]]
        print(
            json.dumps(
                dict(
                    seed=r["seed"],
                    mean_circular_phase_difference_deg=float(
                        np.mean([x["circular_phase_difference_MAE_deg"] for x in shifted])
                    ),
                    max_circular_phase_difference_deg=max(
                        x["circular_phase_difference_MAE_deg"] for x in shifted
                    ),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
