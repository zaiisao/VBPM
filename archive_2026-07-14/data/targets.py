"""Ground-truth-derived supervision targets: the sawtooth bar-phase ramp (and its finite-difference
tempo slope) plus per-crop annotated beats-per-bar classes for the meter emission.

Everything here is built from the SAME beat/downbeat annotations the emission already reconstructs;
the sawtooth target is those labels reshaped into a dense, rate-informative phase ramp.

Provenance of the sawtooth-supervision idea (supervise *phase* via a dense GT ramp, not sparse beats):
  - Oyama, Ishizuka & Yoshii, "Phase-Aware Joint Beat and Downbeat Estimation Based on Periodicity of
    Metrical Structure," ISMIR 2021 -- a per-beat 0->2*pi sawtooth, trained as K-class classification.
  - Chen & Su, "Toward Postprocessing-Free Neural Networks for Joint Beat and Downbeat Estimation,"
    ISMIR 2022 -- a triangular distance-to-nearest-beat label embedding.
Ours differs from both: a unified BAR-level circular regression on a LATENT, treated as a von Mises /
wrapped Cauchy emission with concentration kappa (see losses.py). Post-2026-07-10 (NOSAW verdict) the
sawtooth emission itself defaults OFF; the ramp is still built because the TEMPO-SLOPE emission
supervises log-tempo with the ramp's finite differences.
"""
import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi


def crop_beats_per_bar_classes(beat_targets, downbeat_targets, num_meters):
    """Per-crop annotated beats-per-bar (median beats between downbeats) as a class index, plus a
    validity mask (crops without >=2 downbeats and >=3 beats are excluded from the meter emission)."""
    beats = beat_targets.detach().cpu().numpy()
    downbeats = downbeat_targets.detach().cpu().numpy()
    classes, valid = [], []
    for i in range(beats.shape[0]):
        beat_idx = np.where(beats[i] > 0.5)[0]
        down_idx = np.where(downbeats[i] > 0.5)[0]
        bars = ([int(((beat_idx >= down_idx[j]) & (beat_idx < down_idx[j + 1])).sum())
                 for j in range(len(down_idx) - 1)] if len(down_idx) >= 2 else [])
        bars = [b for b in bars if 1 <= b <= num_meters]
        if len(beat_idx) >= 3 and bars:
            classes.append(int(np.median(bars)) - 1)
            valid.append(1.0)
        else:
            classes.append(0)
            valid.append(0.0)
    device = beat_targets.device
    return torch.tensor(classes, device=device), torch.tensor(valid, device=device)


def build_sawtooth_phase_targets(beat_targets, downbeat_targets, beats_per_bar=4):
    """Per-crop unified bar-phase ramp and its validity mask, both [batch, frames].

    ``beats_per_bar`` sets the ramp's beat spacing (2*pi/M per beat). The default 4 is an explicit
    labelled fallback matching the pinned recipe -- callers with per-song meter available should
    pass the annotated value (never rely on 4 as a modeling assumption)."""
    batch_size, num_frames = beat_targets.shape
    phase_targets = np.zeros((batch_size, num_frames), dtype=np.float32)
    valid_masks = np.zeros((batch_size, num_frames), dtype=np.float32)
    beats_numpy = beat_targets.detach().cpu().numpy()
    downbeats_numpy = downbeat_targets.detach().cpu().numpy()
    for example_index in range(batch_size):
        beat_frames = np.where(beats_numpy[example_index] > 0.5)[0]
        downbeat_frame_set = set(np.where(downbeats_numpy[example_index] > 0.5)[0].tolist())
        anchor_candidates = [i for i, frame in enumerate(beat_frames) if frame in downbeat_frame_set]
        if len(beat_frames) < 2 or not anchor_candidates:
            continue        # nothing to anchor on: leave the mask at zero (no wrong supervision)
        anchor_beat_index = anchor_candidates[0]
        unwrapped_phase_at_beats = (np.arange(len(beat_frames)) - anchor_beat_index) * (TWO_PI / beats_per_bar)
        for beat_index in range(len(beat_frames) - 1):
            start_frame, end_frame = beat_frames[beat_index], beat_frames[beat_index + 1]
            phase_targets[example_index, start_frame:end_frame] = np.linspace(
                unwrapped_phase_at_beats[beat_index], unwrapped_phase_at_beats[beat_index + 1],
                end_frame - start_frame, endpoint=False)
            valid_masks[example_index, start_frame:end_frame] = 1.0
    phase_targets = phase_targets % TWO_PI
    return (torch.from_numpy(phase_targets).to(beat_targets.device),
            torch.from_numpy(valid_masks).to(beat_targets.device))


def build_tempo_slope_targets(phase_targets, valid_masks):
    """Log-advance targets from the ramp's finite differences: [batch, frames-1] target log-advance
    and its validity mask (both frames on a valid ramp, wrapped increment positive and sane)."""
    slope = (phase_targets[:, 1:] - phase_targets[:, :-1]) % TWO_PI
    slope_valid = (valid_masks[:, 1:] * valid_masks[:, :-1]
                   * (slope > 1e-4).float() * (slope < 1.0).float())
    target_log_advance = torch.log(slope.clamp(min=1e-4))
    return target_log_advance, slope_valid
