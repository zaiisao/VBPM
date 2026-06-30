"""Ground-truth derived targets: beat/downbeat times, tempo, and the (optional) sawtooth phase target.

All of these are built from the SAME beat/downbeat annotations the decoder already reconstructs -- the
sawtooth target is just those labels reshaped into a dense, rate-informative phase ramp. It is used only
when the sawtooth-supervision divergence is enabled.

Provenance of the sawtooth-supervision idea (supervise *phase* via a dense GT ramp, not sparse beats):
  - Oyama, Ishizuka & Yoshii, "Phase-Aware Joint Beat and Downbeat Estimation Based on Periodicity of
    Metrical Structure," ISMIR 2021 -- a per-beat 0->2*pi sawtooth, trained as K-class classification.
  - Chen & Su, "Toward Postprocessing-Free Neural Networks for Joint Beat and Downbeat Estimation,"
    ISMIR 2022 -- a triangular "distance-to-nearest-beat" target (their `label2period`, used for a
    label-embedding / structural regularization). Their (TensorFlow) code is not callable from this
    PyTorch pipeline; it lives in the archived tree (see external/README.md).
`build_sawtooth_phase_target` below is OUR variant: a single UNIFIED bar-phase ramp encoding beats AND
downbeats together (it differs from both papers' targets; chosen to avoid a bar-vs-beat loss conflict).
"""
from __future__ import annotations

import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi


def ground_truth_beat_times(beat_targets_1d: np.ndarray, frames_per_second: float) -> np.ndarray:
    """Frame-indexed beat indicators -> beat times in seconds."""
    return np.where(beat_targets_1d > 0.5)[0] / frames_per_second


def ground_truth_tempo_bpm(beat_targets_1d: np.ndarray, frames_per_second: float) -> float:
    """Median-inter-beat-interval tempo in BPM, or NaN if fewer than two beats."""
    beat_frames = np.where(beat_targets_1d > 0.5)[0]
    if len(beat_frames) < 2:
        return float("nan")
    median_interval_frames = float(np.median(np.diff(beat_frames)))
    return 60.0 * frames_per_second / median_interval_frames


def build_sawtooth_phase_target(beat_targets_1d: np.ndarray, downbeat_targets_1d: np.ndarray,
                                beats_per_bar: int, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Single unified bar-phase sawtooth: 0 at downbeats AND 2*pi*k/beats_per_bar at beat k.

    Returns (phase_target, valid_mask), both shape [num_frames]:
      * phase_target: bar phase in [0, 2*pi), rising linearly so that it wraps once per bar and passes
        through the correct sub-position at every beat. This encodes beats AND downbeats in ONE ramp, so
        supervising the model's phase against it has no internal bar-vs-beat conflict.
      * valid_mask: 1.0 only on frames spanned by the annotated beats (phase is undefined before the
        first / after the last beat), so the loss ignores the unspanned frames.
    """
    phase_target = np.zeros(num_frames, dtype=np.float32)
    valid_mask = np.zeros(num_frames, dtype=np.float32)
    beat_frames = np.where(beat_targets_1d > 0.5)[0]
    downbeat_frame_set = set(np.where(downbeat_targets_1d > 0.5)[0].tolist())
    if len(beat_frames) < 2:
        return phase_target, valid_mask

    # Anchor the cumulative phase so the first downbeat-aligned beat sits at a multiple of 2*pi
    # (=> downbeats land at phase 0). Each beat advances the unwrapped phase by 2*pi/beats_per_bar.
    first_downbeat_index = next((i for i, frame in enumerate(beat_frames) if frame in downbeat_frame_set), 0)
    unwrapped_phase_at_beat = np.array(
        [(beat_index - first_downbeat_index) * (TWO_PI / beats_per_bar) for beat_index in range(len(beat_frames))],
        dtype=np.float32,
    )
    for beat_index in range(len(beat_frames) - 1):
        start_frame, end_frame = beat_frames[beat_index], beat_frames[beat_index + 1]
        phase_target[start_frame:end_frame] = np.linspace(
            unwrapped_phase_at_beat[beat_index], unwrapped_phase_at_beat[beat_index + 1],
            end_frame - start_frame, endpoint=False,
        )
        valid_mask[start_frame:end_frame] = 1.0
    phase_target = phase_target % TWO_PI
    return phase_target, valid_mask


def build_sawtooth_phase_target_batch(beat_targets: torch.Tensor, downbeat_targets: torch.Tensor,
                                      beats_per_bar: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched wrapper around build_sawtooth_phase_target. Inputs/outputs are [batch, num_frames]."""
    batch_size, num_frames = beat_targets.shape
    phase_target = np.zeros((batch_size, num_frames), dtype=np.float32)
    valid_mask = np.zeros((batch_size, num_frames), dtype=np.float32)
    beat_numpy = beat_targets.detach().cpu().numpy()
    downbeat_numpy = downbeat_targets.detach().cpu().numpy()
    for example_index in range(batch_size):
        phase_target[example_index], valid_mask[example_index] = build_sawtooth_phase_target(
            beat_numpy[example_index], downbeat_numpy[example_index], beats_per_bar, num_frames,
        )
    device = beat_targets.device
    return torch.from_numpy(phase_target).to(device), torch.from_numpy(valid_mask).to(device)
