"""Annotation-derived phase and velocity references for diagnostic scoring."""

from __future__ import annotations

import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi


def beat_positions(beat_times, downbeat_times, tol: float = 0.02):
    """Each beat's index within its bar, 0 on downbeats; None before the first downbeat."""
    beat_times = np.asarray(beat_times, dtype=np.float64)
    downbeat_times = np.asarray(downbeat_times, dtype=np.float64)
    bar = np.searchsorted(downbeat_times, beat_times + tol, side="right") - 1
    positions = [None] * len(beat_times)
    for i, k in enumerate(bar):
        if k >= 0:
            positions[i] = int(np.sum((beat_times[:i] >= downbeat_times[k] - tol) & (bar[:i] == k)))
    return positions


def bar_meters(positions):
    """Each beat's bar length in beats, or None where the bar is not closed by a downbeat."""
    meters = [None] * len(positions)
    starts = [i for i, p in enumerate(positions) if p == 0]
    for a, b in zip(starts, starts[1:]):
        for i in range(a, b):
            meters[i] = b - a
    return meters


def oracle_draw(beat_frames, positions, meters, frames: int, meter_values=(3, 4)):
    """The draw whose bar-pointer path crosses every beat at its annotated frame.

    Beats may extend past the window on either side; the ones outside set the rates.
    """
    b = np.asarray(beat_frames, dtype=np.float64)
    j = int(np.searchsorted(b, 0.0, side="right"))
    if j == 0 or j >= len(b) or meters[j - 1] is None:
        raise ValueError("oracle_draw needs a closed bar before the window's first beat")

    meter_index = {m: i for i, m in enumerate(meter_values)}
    seg_meter = meters[j - 1]
    spacing = TWO_PI / seg_meter
    phi_prev_beat = positions[j - 1] * spacing
    rate = spacing / (b[j] - b[j - 1])
    phi0 = phi_prev_beat - rate * b[j - 1]

    velocity = np.full(frames, rate)
    meter = np.zeros((frames, len(meter_values)))
    meter[:, meter_index[seg_meter]] = 1.0

    crossings = []
    seg_start, seg_phi = 0, phi0
    landmark = phi_prev_beat + spacing
    while j < len(b):
        c = seg_start + max(1, math.ceil((landmark - seg_phi) / rate - 1e-9))
        if c >= frames:
            break
        downbeat = positions[j] == 0
        crossings.append((c, float(b[j]), downbeat))
        phi_c = seg_phi + rate * (c - seg_start)
        if downbeat:
            seg_meter = meters[j] if meters[j] is not None else seg_meter
            meter[c:, :] = 0.0
            meter[c:, meter_index[seg_meter]] = 1.0
            landmark = math.floor(phi_c / TWO_PI + 1e-9) * TWO_PI
        spacing = TWO_PI / seg_meter
        landmark = landmark + spacing
        if j + 1 >= len(b):
            break
        new_rate = (landmark - phi_c) / (b[j + 1] - c)
        velocity[c:] = new_rate
        seg_start, seg_phi, rate = c, phi_c, new_rate
        j += 1

    def as_tensor(x):
        return torch.tensor(x, dtype=torch.float32)[None]

    draw = {
        "phase0": as_tensor(phi0).reshape(1),
        "phase": torch.zeros(1, frames),
        "velocity": as_tensor(velocity),
        "meter": as_tensor(meter),
    }
    return draw, crossings


def labels_from_beats(beat_times, downbeat_times, start: int, frames: int, fps: float):
    """cls exactly as ExcerptDataset._targets writes it with zero tolerance."""
    cls = np.zeros(frames, dtype=np.int64)
    lo_t, hi_t = start / fps, (start + frames) / fps
    for times, label in ((np.asarray(beat_times), 1), (np.asarray(downbeat_times), 2)):
        for t in times[(times >= lo_t) & (times <= hi_t)]:
            centre = int(round(t * fps)) - start
            if 0 <= centre < frames:
                cls[centre] = label
    return torch.from_numpy(cls)[None]


def synthetic_song(
    bars, bpm: float, fps: float = 50.0, rubato: float = 0.0, offset: float = 0.37, seed: int = 0
):
    """(beat_times, downbeat_times) for a list of bar lengths at bpm, jittered by rubato."""
    rng = np.random.default_rng(seed)
    n = int(sum(bars))
    log_period = (
        math.log(60.0 / bpm) + np.cumsum(rng.normal(0.0, rubato, n))
        if rubato
        else np.full(n, math.log(60.0 / bpm))
    )
    beat_times = offset + np.concatenate([[0.0], np.cumsum(np.exp(log_period[:-1]))])
    starts = np.concatenate([[0], np.cumsum(bars)[:-1]]).astype(int)
    return beat_times, beat_times[starts]
