"""Geometric read-out: turn the latent bar phase into beat/downbeat *times*.

This is the VBPM deployment path: at inference we discard the decoder and read events directly
from the phase latent. A downbeat is where the bar phase phi wraps (completes a revolution); a beat is
where the beat phase (beats_per_bar * phi) wraps. So beats/downbeats are pieced together purely from z.
A separate peak-picker is provided for probability sequences (the MLP decoder / a frontend baseline).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import find_peaks

try:
    import mir_eval
except ImportError:  # mir_eval is only needed for evaluation, not training
    mir_eval = None

TWO_PI = 2.0 * math.pi
FRAMES_PER_SECOND_DEFAULT = 22050.0 / 256.0


def _phase_wrap_frames(phase_1d: np.ndarray, min_separation_frames: int) -> np.ndarray:
    """Frames where a monotonically-advancing phase wraps from ~2*pi back to ~0 (a revolution boundary)."""
    phase_step = np.diff(phase_1d)
    # A wrap shows up as a large negative step (phase fell by nearly 2*pi); de-duplicate nearby wraps.
    wrap_candidate_frames = np.where(phase_step < -math.pi)[0] + 1
    if len(wrap_candidate_frames) == 0:
        return wrap_candidate_frames
    kept = [wrap_candidate_frames[0]]
    for frame in wrap_candidate_frames[1:]:
        if frame - kept[-1] >= min_separation_frames:
            kept.append(frame)
    return np.array(kept)


def phase_to_downbeat_times(phase_1d: np.ndarray, frames_per_second: float = FRAMES_PER_SECOND_DEFAULT,
                            min_separation_seconds: float = 0.30) -> np.ndarray:
    """Downbeats = where the bar phase phi completes one revolution."""
    min_separation_frames = int(min_separation_seconds * frames_per_second)
    return _phase_wrap_frames(phase_1d, min_separation_frames) / frames_per_second


def phase_to_beat_times(phase_1d: np.ndarray, beats_per_bar: int,
                        frames_per_second: float = FRAMES_PER_SECOND_DEFAULT,
                        min_separation_seconds: float = 0.10) -> np.ndarray:
    """Beats = where the beat phase (beats_per_bar * phi) wraps, i.e. phi passes each 2*pi/beats_per_bar."""
    beat_phase = (beats_per_bar * phase_1d) % TWO_PI
    min_separation_frames = int(min_separation_seconds * frames_per_second)
    return _phase_wrap_frames(beat_phase, min_separation_frames) / frames_per_second


def peak_pick_times(probability_1d: np.ndarray, frames_per_second: float = FRAMES_PER_SECOND_DEFAULT,
                    threshold: float = 0.5, min_separation_seconds: float = 0.10) -> np.ndarray:
    """Peak-pick a probability sequence into event times (for the MLP decoder / frontend baselines)."""
    min_separation_frames = max(1, int(min_separation_seconds * frames_per_second))
    peak_frames, _ = find_peaks(probability_1d, height=threshold, distance=min_separation_frames)
    return peak_frames / frames_per_second


def f_measure(reference_times: np.ndarray, estimated_times: np.ndarray,
              tolerance_seconds: float = 0.07) -> float:
    """mir_eval beat F-measure. NaN if no reference; 0 if reference exists but nothing was estimated."""
    reference_times = np.asarray(reference_times, dtype=float)
    estimated_times = np.asarray(estimated_times, dtype=float)
    if len(reference_times) == 0:
        return float("nan")
    if len(estimated_times) == 0:
        return 0.0
    return float(mir_eval.beat.f_measure(reference_times, estimated_times, f_measure_threshold=tolerance_seconds))
