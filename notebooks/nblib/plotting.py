"""Every figure the notebook renders: training history, deployment phase trajectories, the leak table,
and the raw-material panel.

Model/songs are passed in; fixed constants and the palette come from ``setup``. Each plotting function
ends in ``plt.show()`` so it renders inline exactly as it did when defined in the notebook.
"""
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from .setup import (TWO_PI, FRAMES_PER_SECOND, FEATURE_DIM, DEVICE,
                    COLOR_BLUE, COLOR_AQUA, COLOR_YELLOW, COLOR_VIOLET, COLOR_RED)
from .evaluation import evaluate_geometric_readout, DEFAULT_EVAL_MAX_FRAMES


def plot_training_history(history, title):
    figure, (kl_axis, f_axis) = plt.subplots(1, 2, figsize=(11, 3.4))
    kl_axis.plot(history["step"], history["kl_phase"], color=COLOR_BLUE, lw=2, label="phase")
    kl_axis.plot(history["step"], history["kl_tempo"], color=COLOR_AQUA, lw=2, label="tempo")
    kl_axis.plot(history["step"], history["kl_meter"], color=COLOR_YELLOW, lw=2, label="meter")
    kl_axis.set_xlabel("training step")
    kl_axis.set_ylabel("KL (nats / sequence)")
    kl_axis.set_yscale("log")
    kl_axis.set_title(f"{title}: per-latent KL streams")
    kl_axis.legend(frameon=False)
    f_axis.plot(history["val_step"], history["val_beat_f"], color=COLOR_BLUE, lw=2, label="beat F")
    f_axis.plot(history["val_step"], history["val_downbeat_f"], color=COLOR_RED, lw=2, label="downbeat F")
    f_axis.set_xlabel("training step")
    f_axis.set_ylabel("geometric read-out F-measure")
    f_axis.set_ylim(-0.02, 1.0)
    f_axis.set_title(f"{title}: deployment score")
    f_axis.legend(frameon=False)
    for axis in (kl_axis, f_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.show()


def plot_phase_trajectories(model, songs, title, num_songs_to_plot=3, num_seconds=10):
    num_frames = int(num_seconds * FRAMES_PER_SECOND)
    figure, axes = plt.subplots(num_songs_to_plot, 1, figsize=(10, 1.9 * num_songs_to_plot), sharex=True)
    with torch.no_grad():
        model.eval()
        for axis, song in zip(axes, songs[:num_songs_to_plot]):
            frames = min(song.features.shape[0], num_frames)
            silent_channel = torch.zeros(1, frames, device=DEVICE)
            rollout = model.rollout(song.features[:frames].unsqueeze(0).to(DEVICE),
                                    silent_channel, silent_channel, sample=False, compute_kl=False)
            seconds = np.arange(frames) / FRAMES_PER_SECOND
            axis.plot(seconds, rollout.bar_phase[0].cpu().numpy(), color=COLOR_BLUE, lw=1.5)
            beat_seconds = np.where(song.beat_targets[:frames].numpy() > 0.5)[0] / FRAMES_PER_SECOND
            downbeat_seconds = np.where(song.downbeat_targets[:frames].numpy() > 0.5)[0] / FRAMES_PER_SECOND
            axis.vlines(beat_seconds, 0, TWO_PI, color=COLOR_YELLOW, lw=0.8, alpha=0.8)
            axis.vlines(downbeat_seconds, 0, TWO_PI, color=COLOR_RED, lw=1.4)
            axis.set_ylabel("$\\varphi_t$")
            axis.set_yticks([0, math.pi, TWO_PI], ["0", "$\\pi$", "$2\\pi$"])
            axis.spines[["top", "right"]].set_visible(False)
        model.train()
    axes[0].set_title(f"{title} — bar phase at deployment (yellow = annotated beats, red = downbeats)")
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()


def print_leak_table(model, songs, label, max_frames=DEFAULT_EVAL_MAX_FRAMES):
    results = {}
    for condition in ("real", "shuffle", "zero"):
        results[condition] = evaluate_geometric_readout(model, songs, condition, max_frames)
    print(f"\n{label} — geometric read-out on {len(songs)} validation songs:")
    print(f"  {'condition':<10} {'beat F':>8} {'downbeat F':>11}")
    for condition, result in results.items():
        print(f"  {condition:<10} {result['beat_f']:>8.3f} {result['downbeat_f']:>11.3f}")
    real = results["real"]
    print(f"  mechanism (real): phase coverage {real['phase_coverage']:.2f}, "
          f"rotation ratio {real['rotation_ratio']:.2f}")
    return results


def plot_raw_material(song, num_seconds=6):
    # A look at the raw material: an excerpt of frontend features with the annotated beats/downbeats.
    excerpt_frames = int(num_seconds * FRAMES_PER_SECOND)
    excerpt_seconds = np.arange(excerpt_frames) / FRAMES_PER_SECOND
    figure, (feature_axis, event_axis) = plt.subplots(
        2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    feature_axis.imshow(song.features[:excerpt_frames].T.numpy(), aspect="auto", origin="lower",
                        cmap="Blues", extent=[0, excerpt_seconds[-1], 0, FEATURE_DIM])
    feature_axis.set_ylabel("feature dim")
    feature_axis.set_title("Frozen frontend features $\\mathbf{h}_t$ (6 s excerpt) and the events to explain")
    beat_times = np.where(song.beat_targets[:excerpt_frames].numpy() > 0.5)[0] / FRAMES_PER_SECOND
    downbeat_times = np.where(song.downbeat_targets[:excerpt_frames].numpy() > 0.5)[0] / FRAMES_PER_SECOND
    event_axis.vlines(beat_times, 0, 0.6, color=COLOR_BLUE, lw=2, label="beat")
    event_axis.vlines(downbeat_times, 0, 1.0, color=COLOR_RED, lw=2, label="downbeat")
    event_axis.set_yticks([])
    event_axis.set_xlabel("time (s)")
    event_axis.legend(loc="upper right", frameon=False, ncol=2)
    for axis in (feature_axis, event_axis):
        axis.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
