"""Reference-only bar-origin probe; no candidate selection is used for inference."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    batch = torch.load(a.batch_cache, weights_only=True)
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
            phase, velocity = model.trajectory(batch["h"])
            audio_logits = F.linear(batch["h"], model.prediction_weight, model.prediction_bias)[
                ..., 1
            ].numpy()
            candidates = []
            for k in range(4):
                shifted = phase + k * math.pi / 2
                logits = model.decoder(shifted, velocity)
                ce = (
                    F.cross_entropy(
                        logits.reshape(-1, 3), batch["labels"].reshape(-1), reduction="none"
                    )
                    .reshape_as(phase)
                    .sum(-1)
                )
                rows = []
                for i in range(len(phase)):
                    beats, downbeats = phase_events(shifted[i].numpy())
                    scores = np.interp(downbeats * 50, np.arange(phase.shape[1]), audio_logits[i])
                    rows.append(
                        dict(
                            shift_quarters=k,
                            observation_CE=float(ce[i]),
                            beat_F1_70ms=f_measure(beats, batch["beat_times"][i])[0],
                            downbeat_F1_70ms=f_measure(downbeats, batch["downbeat_times"][i])[0],
                            predicted_downbeats=downbeats.tolist(),
                            audio_downbeat_logits=scores.tolist(),
                            mean_audio_downbeat_logit=float(scores.mean()) if len(scores) else None,
                        )
                    )
                candidates.append(rows)
        windows = []
        for i in range(len(phase)):
            rows = [x[i] for x in candidates]
            windows.append(
                dict(
                    song=batch["songs"][i],
                    candidates=rows,
                    diagnostic_min_CE_shift=min(rows, key=lambda r: r["observation_CE"])[
                        "shift_quarters"
                    ],
                    diagnostic_best_downbeat_F1=max(r["downbeat_F1_70ms"] for r in rows),
                )
            )
        results.append(dict(seed=seed, windows=windows))
    a.output.write_text(
        json.dumps(
            dict(
                scope=(
                    "Reference-only likelihood and physical representability probe; ac"
                    "tual prediction is shift zero; no oracle substitution"
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
        print(
            json.dumps(
                dict(
                    seed=r["seed"],
                    actual_downbeat_F1=float(
                        np.mean([w["candidates"][0]["downbeat_F1_70ms"] for w in r["windows"]])
                    ),
                    diagnostic_best_downbeat_F1=float(
                        np.mean([w["diagnostic_best_downbeat_F1"] for w in r["windows"]])
                    ),
                    diagnostic_min_CE_shifts=[w["diagnostic_min_CE_shift"] for w in r["windows"]],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
