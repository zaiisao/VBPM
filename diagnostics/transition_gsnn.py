"""Test latent trajectories and emission predictions in the fresh GSNN model."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from vbpm.model import VBPM
from vbpm.nets import LatentState
from vbpm.scoring.evaluation import decode_event_times, f_measure


FPS = 50


def tutorial_batch():
    """Ancestral fixed-meter data for the tutorial's 16-by-32 overfit check."""
    batch_size, frames = 16, 32
    phase = torch.rand(batch_size) * 2 * math.pi - math.pi
    velocity = 0.35 + 0.05 * torch.randn(batch_size)
    phases, velocities, labels = [], [], []
    centers = torch.arange(4) * math.pi / 2
    for _ in range(frames):
        phase = torch.distributions.VonMises(phase + velocity, 20.0).sample()
        velocity = velocity + 0.03 * torch.randn(batch_size)
        distance = phase[:, None] - centers
        distance = torch.atan2(distance.sin(), distance.cos())
        bumps = torch.exp(-(distance / 0.3).square())
        logits = torch.stack((torch.zeros_like(phase), 8 * bumps[:, 1:].amax(-1),
                              8 * bumps[:, 0]), dim=-1)
        labels.append(torch.distributions.Categorical(logits=logits).sample())
        phases.append(phase)
        velocities.append(velocity)
    phase = torch.stack(phases, dim=1)
    velocity = torch.stack(velocities, dim=1)
    h = torch.stack((phase.cos(), phase.sin(), torch.zeros_like(phase),
                     torch.zeros_like(phase)), dim=-1)
    h = h + 0.1 * torch.randn_like(h)
    return dict(h=h, labels=torch.stack(labels, dim=1), phi=phase, velocity=velocity)


@torch.no_grad()
def measure_tutorial(model, batch):
    """Separate prior-only prediction from label-conditioned posterior reconstruction."""
    result = {}
    for name in ('prior', 'posterior'):
        if name == 'prior':
            logits, state = model.rollout(batch['h'], samples=16)
        else:
            logits, state, _ = model.posterior_rollout(batch['h'], batch['labels'], samples=16)
        probabilities = logits.softmax(-1).mean(0)
        predictions = probabilities.argmax(-1)
        confusion = torch.bincount(
            (3 * batch['labels'] + predictions).flatten(), minlength=9
        ).reshape(3, 3)
        error = state.phase - batch['phi']
        circular_error = torch.atan2(error.sin(), error.cos()).abs()
        result[name] = dict(
            accuracy=float((predictions == batch['labels']).float().mean()),
            confusion_true_rows_predicted_columns=confusion.tolist(),
            circular_phase_mae_rad=float(circular_error.mean()),
            velocity_rmse_rad_per_step=float(
                (state.velocity - batch['velocity']).square().mean().sqrt()
            ),
        )
    return result


def synthetic_batch():
    """Thirty-second clips with explicit phase and tempo cues, used only as inputs."""
    bpm = torch.tensor([100.0, 140.0])
    velocity = bpm * (2 * math.pi / 4) / 60
    timeline = torch.arange(1500) / FPS
    phase = torch.tensor([0.4, 1.2])[:, None] + velocity[:, None] * timeline
    beat_distance = torch.atan2((4 * phase).sin(), (4 * phase).cos()) / 4
    down_distance = torch.atan2(phase.sin(), phase.cos())
    h = torch.stack(
        (
            phase.cos(),
            phase.sin(),
            (4 * phase).cos(),
            (4 * phase).sin(),
            torch.exp(-0.5 * (beat_distance / 0.08).square()),
            torch.exp(-0.5 * (down_distance / 0.08).square()),
            velocity[:, None].expand_as(phase),
            torch.ones_like(phase),
        ),
        dim=-1,
    )
    labels = torch.zeros_like(phase, dtype=torch.long)
    beat_times, downbeat_times = [], []
    for i in range(2):
        events = []
        for spacing, label in ((math.pi / 2, 1), (2 * math.pi, 2)):
            indices = np.arange(
                math.ceil(float(phase[i, 0]) / spacing),
                math.floor(float(phase[i, -1]) / spacing) + 1,
            )
            times = (indices * spacing - float(phase[i, 0])) / float(velocity[i])
            labels[i, torch.tensor(np.rint(times * FPS), dtype=torch.long)] = label
            events.append(times.tolist())
        beat_times.append(events[0])
        downbeat_times.append(events[1])
    return dict(
        h=h,
        labels=labels,
        phi=phase,
        velocity=(velocity[:, None] / FPS).expand_as(phase),
        beat_times=beat_times,
        downbeat_times=downbeat_times,
    )


def event_scores(probabilities, batch):
    """Beat/downbeat F1 for each clip, with a fixed 70 ms tolerance."""
    scores = []
    for i in range(len(probabilities)):
        events = decode_event_times(probabilities[i], FPS)
        scores.append(
            [
                f_measure(events[0], batch["beat_times"][i])[0],
                f_measure(events[1], batch["downbeat_times"][i])[0],
            ]
        )
    return scores


@torch.no_grad()
def measure(model, batch):
    """Score physical latents separately from labels, with 70 ms event tolerance."""
    h, labels = batch["h"], batch["labels"]
    logits, states = model.rollout(h, sample=False)
    _, sampled = model.rollout(h, samples=4)
    phase_error = states.phase[0] - batch["phi"]
    phase_error = torch.atan2(phase_error.sin(), phase_error.cos()).abs()
    true_speed = batch["velocity"].abs() * FPS
    time_error = phase_error / true_speed.clamp_min(1e-6) * 1000
    tempo_error = (states.velocity[0] - batch["velocity"] * FPS) * (4 * 60 / (2 * math.pi))
    sampled_error = sampled.phase - batch["phi"]
    sampled_error = torch.atan2(sampled_error.sin(), sampled_error.cos()).abs()
    probabilities = model.emission_model(sampled).softmax(-1).mean(0).numpy()
    scores = event_scores(probabilities, batch)
    zero = torch.zeros_like(states.phase[0])
    ablated = model.emission_model(LatentState(zero, zero))
    full_nll = F.cross_entropy(logits[0].reshape(-1, 3), labels.reshape(-1))
    ablated_nll = F.cross_entropy(ablated.reshape(-1, 3), labels.reshape(-1))
    phase_removed = (
        model.emission_model(LatentState(torch.zeros_like(sampled.phase), sampled.velocity))
        .softmax(-1)
        .mean(0)
        .numpy()
    )
    velocity_removed = (
        model.emission_model(LatentState(sampled.phase, torch.zeros_like(sampled.velocity)))
        .softmax(-1)
        .mean(0)
        .numpy()
    )
    phase_removed_scores = np.mean(event_scores(phase_removed, batch), axis=0)
    velocity_removed_scores = np.mean(event_scores(velocity_removed, batch), axis=0)
    return dict(
        phase_p95_ms=float(torch.quantile(time_error, 0.95)),
        phase_p95_after_5s_ms=float(torch.quantile(time_error[:, 250:], 0.95)),
        sampled_phase_p95_ms=float(torch.quantile(sampled_error / true_speed * 1000, 0.95)),
        tempo_rmse_bpm=float(tempo_error.square().mean().sqrt()),
        minimum_mean_bpm=float(states.velocity.min() * 4 * 60 / (2 * math.pi)),
        maximum_mean_bpm=float(states.velocity.max() * 4 * 60 / (2 * math.pi)),
        beat_f1_70ms=float(np.mean(scores, axis=0)[0]),
        downbeat_f1_70ms=float(np.mean(scores, axis=0)[1]),
        deterministic_label_nll=float(full_nll),
        zero_latent_label_nll=float(ablated_nll),
        zero_phase_beat_f1_70ms=float(phase_removed_scores[0]),
        zero_phase_downbeat_f1_70ms=float(phase_removed_scores[1]),
        zero_velocity_beat_f1_70ms=float(velocity_removed_scores[0]),
        zero_velocity_downbeat_f1_70ms=float(velocity_removed_scores[1]),
    )


def main():
    """Run GSNN controls or the tutorial's hybrid single-batch overfit check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tutorial", action="store_true")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    alpha = args.alpha if args.alpha is not None else (0.7 if args.tutorial else 0.0)
    batch = (
        torch.load(args.cache, weights_only=False, map_location="cpu")
        if args.cache
        else (tutorial_batch() if args.tutorial else synthetic_batch())
    )
    if not args.tutorial:
        batch = {
            key: value[:2] if isinstance(value, (torch.Tensor, list)) else value
            for key, value in batch.items()
        }
    model = VBPM(
        SimpleNamespace(num_channels=batch["h"].shape[-1],
                        output_fps=1 if args.tutorial else FPS),
        d_model=64 if args.tutorial else 32, samples=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    history = []
    training_history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.tutorial:
        torch.save(batch, args.output.with_suffix('.batch.pt'))
    for step in range(args.steps + 1):
        if step % 50 == 0 or step == args.steps:
            metrics = dict(step=step, **(
                measure_tutorial(model, batch) if args.tutorial else measure(model, batch)
            ))
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(dict(seed=args.seed, cache=str(args.cache), alpha=alpha,
                                beta=1.0, tutorial=args.tutorial, history=history,
                                training_history=training_history), indent=2)
                + "\n"
            )
            torch.save(model.state_dict(), args.output.with_suffix(".pt"))
        if step == args.steps:
            break
        optimizer.zero_grad()
        result = model(batch["h"], torch.ones_like(batch["labels"]), batch["labels"],
                       gsnn_only=alpha == 0)
        loss = -(alpha * result['elbo'] + (1 - alpha) * result['prior_recon']).mean()
        if not args.tutorial:
            loss = loss / batch['h'].shape[1]
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise RuntimeError(f"Nonfinite loss or gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        if args.tutorial and (step == 0 or (step + 1) % 50 == 0):
            groups = {name: module for name, module in (
                ('prior', model.prior_model), ('posterior', model.posterior_model),
                ('emission', model.emission_model)
            )}
            training_metrics = dict(
                update=step + 1, loss=float(loss.detach()),
                posterior_nll=float(-result['recon'].mean().detach()),
                prior_nll=float(-result['prior_recon'].mean().detach()),
                kl_terms={k: float(v.mean().detach()) for k, v in result['kl_terms'].items()},
                gradient_norms={name: float(torch.stack([
                    p.grad.norm() for p in module.parameters() if p.grad is not None
                ]).norm()) for name, module in groups.items()},
            )
            training_history.append(training_metrics)
            print(json.dumps(training_metrics), flush=True)
        optimizer.step()
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
