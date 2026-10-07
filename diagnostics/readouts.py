"""Compare physical prior events, decoder peaks, and argmax classifications.

All readouts use predictions only. Annotation references appear only in
matching. Sampling is from the conditional prior, without posterior calls.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from diagnostics.audio_generator import ROOT, AudioPhaseAttentionGSNN, AudioPhaseFirstGSNN
from diagnostics.audio_metrics import score
from diagnostics.real_event_references import attach_references
from vbpm.scoring.evaluation import f_measure, peak_times


def phase_events(phase, fps=50.0, meter=4):
    """Extract beat and downbeat crossings from a phase trajectory."""
    phase = np.asarray(phase, dtype=np.float64)
    if len(phase) < 2 or not np.all(np.diff(phase) > 0):
        return np.array([]), np.array([])
    step = 2 * math.pi / meter
    k = np.arange(math.ceil(phase[0] / step), math.floor(phase[-1] / step) + 1)
    target = k * step
    index = np.searchsorted(phase, target).clip(1, len(phase) - 1)
    times = (index - 1 + (target - phase[index - 1]) / (phase[index] - phase[index - 1])) / fps
    return times, times[k % meter == 0]


def main():
    """Run the command-line tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-cache", type=Path, default=ROOT / "runs/generator_isolation/oracle_batch.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() or args.draws < 1:
        parser.error("Fresh output and positive draw count required")
    torch.set_num_threads(1)
    batch = torch.load(args.batch_cache, weights_only=True)
    data = dict(
        x=batch["h"], labels=batch["labels"], phase=batch["phi"], velocity=batch["velocity"][:, :-1]
    )
    if "beat_times" in batch:
        data.update(beat_times=batch["beat_times"], downbeat_times=batch["downbeat_times"])
    else:
        attach_references(data, batch)
    count, frames = batch["h"].shape[:2]
    generator = torch.Generator().manual_seed(194781)
    noise = dict(
        uniform=torch.rand(args.draws, count, generator=generator).clamp(1e-6, 1 - 1e-6),
        initial=torch.randn(args.draws, count, generator=generator),
        increments=torch.randn(args.draws, count, frames - 2, generator=generator),
    )
    results = []
    for seed in (0, 1):
        checkpoint = torch.load(args.model_dir / f"seed{seed}.pt", weights_only=True)
        cls = AudioPhaseAttentionGSNN if checkpoint.get("phase_attention") else AudioPhaseFirstGSNN
        extra = (
            dict(phase_residual=checkpoint.get("phase_residual", False))
            if checkpoint.get("phase_attention", False)
            else {}
        )
        extra["angular_frame_bin_emission"] = checkpoint.get("angular_frame_bin_emission", False)
        extra["log_tempo_noise"] = checkpoint.get("log_tempo_noise", False)
        extra["clock_mass_emission"] = checkpoint.get("clock_mass_emission", False)
        extra["bernoulli_clock_emission"] = checkpoint.get("bernoulli_clock_emission", False)
        extra["proposal_min_probability"] = checkpoint.get("proposal_min_probability")
        extra["tempo_feedback"] = checkpoint.get("tempo_feedback", False)
        model = cls(
            checkpoint["learn_tempo"],
            checkpoint.get("aligned_context", False),
            checkpoint.get("proposal_max_bpm"),
            checkpoint.get("tempo_log_scale", 0.1),
            checkpoint.get("tempo_basis", "framewise"),
            checkpoint.get("smooth_concentration", False),
            checkpoint.get("frame_bin_emission", False),
            **extra,
        )
        if checkpoint.get("temporal_phase_context", False):
            model.enable_temporal_phase_context()
        model.periodic_context_only = checkpoint.get("periodic_context_only", False)
        model.recover_missing_beats = checkpoint.get("recover_missing_beats", False)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        with torch.no_grad():
            result = dict(seed=seed, scores=score(model, data))
            phase, velocity = model.trajectory(data["x"])
            probabilities = model.decoder(phase, velocity).softmax(-1)
            sampled, sampled_velocity = model.draw(data["x"], noise)

        def match(p):
            beats, downbeats = [], []
            for i in range(count):
                be, de = phase_events(p[i].numpy())
                beats.append(f_measure(be, data["beat_times"][i])[0])
                downbeats.append(f_measure(de, data["downbeat_times"][i])[0])
            return beats, downbeats

        be, de = match(phase)
        result["latent_mean"] = dict(
            beat_F1_70ms=float(np.mean(be)), downbeat_F1_70ms=float(np.mean(de))
        )
        result["deterministic_trajectory_definition"] = (
            (
                "Zero-noise parameter trajectory; log-tempo is median, not the ari"
                "thmetic expected velocity"
            )
            if checkpoint.get("log_tempo_noise", False)
            else "Conditional mean parameter trajectory"
        )
        pb, pd = [], []
        for i in range(count):
            probs = probabilities[i].numpy()
            period = 2 * math.pi / float(velocity[i].mean()) / 50
            pb.append(
                f_measure(peak_times(probs[:, 1:].sum(-1), 50, period / 4), data["beat_times"][i])[
                    0
                ]
            )
            pd.append(f_measure(peak_times(probs[:, 2], 50, period), data["downbeat_times"][i])[0])
        result["decoder_relative_peaks"] = dict(
            beat_F1_70ms=float(np.mean(pb)), downbeat_F1_70ms=float(np.mean(pd))
        )
        samples = [match(p) for p in sampled]
        be = np.asarray([p[0] for p in samples])
        de = np.asarray([p[1] for p in samples])
        result["latent_samples"] = dict(
            draws=args.draws,
            beat_F1_70ms=float(be.mean()),
            downbeat_F1_70ms=float(de.mean()),
            beat_F1_per_window=be.mean(0).tolist(),
            downbeat_F1_per_window=de.mean(0).tolist(),
            nonpositive_velocity_fraction=float((sampled_velocity <= 0).float().mean()),
        )
        result["kappa"] = model.concentration(data["x"]).detach().tolist()
        results.append(result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("seed", "latent_mean", "decoder_relative_peaks", "latent_samples")
                }
            ),
            flush=True,
        )
    report = dict(
        model_dir=str(args.model_dir),
        batch_cache=str(args.batch_cache),
        songs=batch["songs"],
        scope="Fixed cached cohort; prior means and fixed-seed prior samples; no posterior",
        readouts=(
            "Argmax classified regions, canonical physical phase crossings, an"
            "d existing prediction-only relative peak_times with period from p"
            "redicted velocity"
        ),
        truth_usage=(
            "References used only for +/-70ms one-to-one matching; no acceptance gates rewritten"
        ),
        results=results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
