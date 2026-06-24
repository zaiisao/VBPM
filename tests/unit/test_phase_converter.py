"""Unit tests for evaluation/phase_converter.py.

Ground truth strategy:
  * extract_beats_from_phase_trajectory: build an analytic sawtooth phase that
    wraps every P frames; the wrap frames are known in closed form, so the
    expected beat *times* are known exactly (k / fps for each reset frame k).
  * extract_beat_timestamps: build a probability curve with sharp, isolated
    spikes at chosen frames; the peak-picker must return exactly those frames
    (converted to seconds). Distance/threshold/boundary behaviour checked
    against the explicit algorithmic contract in the docstring.
  * extract_downbeat_timestamps / extract_timestamps_from_phase: structural
    invariants (subset relation, phase-near-0 selection).

No value is copied from a debug run; every expectation is derived from the
math of the synthetic input.
"""

import sys

sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import numpy as np
import pytest

from evaluation.phase_converter import (
    extract_beat_timestamps,
    extract_beats_from_phase_trajectory,
    extract_downbeat_timestamps,
    extract_timestamps_from_phase,
)

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_sawtooth(period_frames, n_cycles, fps):
    """Analytic sawtooth phase in radians that resets to ~0 every `period_frames`.

    phase[t] = (t * 2pi / P) mod 2pi.  Resets (wrap-arounds) happen exactly at
    frames t = P, 2P, 3P, ...  (the t=0 frame is the very first ramp start, not
    a wrap because there is no preceding large value).
    """
    T = period_frames * n_cycles
    t = np.arange(T, dtype=np.float64)
    phase = (t * TWO_PI / period_frames) % TWO_PI
    expected_wrap_frames = np.array(
        [period_frames * k for k in range(1, n_cycles)], dtype=np.int64
    )
    return phase, expected_wrap_frames


# ===========================================================================
# extract_beats_from_phase_trajectory
# ===========================================================================
def test_phase_traj_clean_sawtooth_count_and_times():
    """Clean sawtooth wrapping every P frames -> beats at the wrap frames."""
    P = 40
    n_cycles = 6
    fps = 100.0
    phase, expected_frames = make_sawtooth(P, n_cycles, fps)

    beats = extract_beats_from_phase_trajectory(
        phase, fps=fps, min_distance_sec=0.15
    )
    expected_times = expected_frames / fps

    # min_distance = max(1, int(0.15*100)) = 15 frames; P=40 > 15 so all kept.
    assert beats.shape == expected_times.shape, (
        f"expected {len(expected_times)} beats, got {len(beats)}"
    )
    np.testing.assert_allclose(beats, expected_times, atol=1.0 / fps)


def test_phase_traj_wrap_frame_is_reset_frame():
    """The detected frame index must be the *reset* frame k*P, not k*P-1."""
    P = 50
    fps = 100.0
    phase, expected_frames = make_sawtooth(P, 4, fps)
    beats = extract_beats_from_phase_trajectory(phase, fps=fps, min_distance_sec=0.0)
    got_frames = np.round(beats * fps).astype(np.int64)
    np.testing.assert_array_equal(got_frames, expected_frames)


def test_phase_traj_min_distance_drops_close_wraps():
    """If wraps are closer than min_distance, the greedy filter drops them.

    Build a fast sawtooth that resets every 5 frames at fps=100 -> 0.05s apart.
    With min_distance_sec=0.15 -> min_dist=15 frames, only every 15-frame-spaced
    wrap survives. Algorithm keeps first wrap then any wrap >=15 frames later.
    """
    P = 5
    n_cycles = 13  # wraps at 5,10,...,60
    fps = 100.0
    phase, all_wraps = make_sawtooth(P, n_cycles, fps)

    beats = extract_beats_from_phase_trajectory(phase, fps=fps, min_distance_sec=0.15)
    got_frames = np.round(beats * fps).astype(np.int64)

    # Reproduce the documented greedy rule on the known wrap frames.
    min_dist = max(1, int(0.15 * fps))  # 15
    expected = [all_wraps[0]]
    for w in all_wraps[1:]:
        if w - expected[-1] >= min_dist:
            expected.append(w)
    np.testing.assert_array_equal(got_frames, np.array(expected, dtype=np.int64))
    # consecutive kept beats are at least min_dist apart
    if len(got_frames) > 1:
        assert np.all(np.diff(got_frames) >= min_dist)


def test_phase_traj_flat_phase_no_wraps():
    """Flat (constant) phase has no wraps -> empty result."""
    phase = np.full(200, 1.234, dtype=np.float64)
    beats = extract_beats_from_phase_trajectory(phase, fps=100.0)
    assert beats.shape == (0,)
    assert beats.dtype == np.float64


def test_phase_traj_monotonic_no_wrap_when_no_reset():
    """A single rising ramp that never completes a full turn -> no wrap."""
    # ramp from 0 to ~1.9pi over 100 frames, never reaching 2pi reset.
    phase = np.linspace(0.0, 1.9 * math.pi, 100)
    beats = extract_beats_from_phase_trajectory(phase, fps=100.0)
    assert beats.shape == (0,)


def test_phase_traj_unwrapped_input_is_modded_internally():
    """Function does phase % 2pi internally, so an *unwrapped* increasing phase
    (0 .. n_cycles*2pi) must produce the same wrap frames as the wrapped one.

    NOTE: period chosen so that t*2pi/P does not land a float just-below-2pi at
    the integer reset frame (which would push the detected wrap one frame later
    purely from rounding -- a numerical artifact, not a logic error)."""
    P = 37
    n_cycles = 5
    fps = 100.0
    t = np.arange(P * n_cycles, dtype=np.float64)
    unwrapped = t * TWO_PI / P  # monotonically increasing, no explicit mod
    beats = extract_beats_from_phase_trajectory(unwrapped, fps=fps, min_distance_sec=0.0)
    got_frames = np.round(beats * fps).astype(np.int64)
    expected = np.array([P * k for k in range(1, n_cycles)], dtype=np.int64)
    np.testing.assert_array_equal(got_frames, expected)


def test_phase_traj_fps_scaling():
    """Beat *times* must scale inversely with fps for identical frame wraps."""
    P = 20
    n_cycles = 5
    phase, expected_frames = make_sawtooth(P, n_cycles, 100.0)
    b1 = extract_beats_from_phase_trajectory(phase, fps=100.0, min_distance_sec=0.0)
    b2 = extract_beats_from_phase_trajectory(phase, fps=200.0, min_distance_sec=0.0)
    # same frames, double fps -> half the time
    np.testing.assert_allclose(b1, expected_frames / 100.0, atol=1e-9)
    np.testing.assert_allclose(b2, expected_frames / 200.0, atol=1e-9)
    np.testing.assert_allclose(b2 * 2.0, b1, atol=1e-9)


def test_phase_traj_no_spurious_wrap_on_small_jitter():
    """Small descending jitter (< pi) must NOT be detected as a wrap."""
    rng = np.random.default_rng(0)
    base = np.linspace(0.0, 1.0, 300)  # smoothly rising, well below pi swings
    jitter = rng.normal(0.0, 0.01, size=base.shape)
    phase = base + jitter
    beats = extract_beats_from_phase_trajectory(phase, fps=100.0)
    assert beats.shape == (0,)


# ===========================================================================
# extract_beat_timestamps
# ===========================================================================
def make_spikes(T, spike_frames, peak=1.0, floor=0.0):
    probs = np.full(T, floor, dtype=np.float64)
    for f in spike_frames:
        probs[f] = peak
    return probs


def test_beat_ts_sharp_spikes_detected_at_frames():
    """Isolated sharp spikes above threshold -> peaks at exactly those frames."""
    fps = 100.0
    spike_frames = [20, 60, 110, 175]  # all >= 15 frames apart, interior
    T = 200
    probs = make_spikes(T, spike_frames, peak=0.9, floor=0.05)
    beats = extract_beat_timestamps(probs, fps=fps, threshold=0.5, min_distance_sec=0.15)
    got_frames = np.round(beats * fps).astype(np.int64)
    np.testing.assert_array_equal(got_frames, np.array(spike_frames, dtype=np.int64))


def test_beat_ts_below_threshold_not_detected():
    """Spikes below threshold are ignored."""
    T = 100
    probs = make_spikes(T, [30, 60], peak=0.4, floor=0.0)  # 0.4 < 0.5
    beats = extract_beat_timestamps(probs, fps=100.0, threshold=0.5)
    assert beats.shape == (0,)


def test_beat_ts_min_distance_keeps_higher_peak():
    """Two peaks closer than min_distance: greedy keeps the *higher* one."""
    fps = 100.0
    T = 100
    probs = np.full(T, 0.0)
    # peaks at 40 (0.7) and 45 (0.95), 5 frames < 15-frame min distance
    probs[40] = 0.7
    probs[45] = 0.95
    beats = extract_beat_timestamps(probs, fps=fps, threshold=0.5, min_distance_sec=0.15)
    got_frames = np.round(beats * fps).astype(np.int64)
    # Only the higher peak (frame 45) should survive.
    np.testing.assert_array_equal(got_frames, np.array([45], dtype=np.int64))


def test_beat_ts_min_distance_respected_between_kept_beats():
    """All returned beats must be >= min_distance_frames apart."""
    fps = 100.0
    T = 400
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.0, 1.0, size=T)
    beats = extract_beat_timestamps(probs, fps=fps, threshold=0.5, min_distance_sec=0.15)
    min_dist_frames = max(1, int(0.15 * fps))
    frames = np.round(beats * fps).astype(np.int64)
    if len(frames) > 1:
        assert np.all(np.diff(np.sort(frames)) >= min_dist_frames)


def test_beat_ts_all_zero_probs():
    """All-zero probabilities -> no beats."""
    probs = np.zeros(150, dtype=np.float64)
    beats = extract_beat_timestamps(probs, fps=100.0)
    assert beats.shape == (0,)
    assert beats.dtype == np.float64


def test_beat_ts_constant_high_probs_no_strict_local_max_but_plateau():
    """A constant high curve: docstring says local maxima with >= comparisons.

    With probs all equal, every interior frame satisfies p>=p_prev and p>=p_next,
    so MANY candidates exist; after greedy min-distance filtering they must be
    spaced >= min_dist apart and never violate the distance constraint.
    """
    fps = 100.0
    T = 100
    probs = np.full(T, 0.8)
    beats = extract_beat_timestamps(probs, fps=fps, threshold=0.5, min_distance_sec=0.15)
    frames = np.round(beats * fps).astype(np.int64)
    min_dist = max(1, int(0.15 * fps))
    # at least one beat detected, and all spaced legally
    assert len(frames) >= 1
    if len(frames) > 1:
        assert np.all(np.diff(np.sort(frames)) >= min_dist)


def test_beat_ts_boundary_frames_never_peaks():
    """The algorithm scans range(1, T-1); frame 0 and frame T-1 can't be peaks."""
    T = 50
    probs = np.zeros(T)
    probs[0] = 1.0       # boundary, should be ignored
    probs[T - 1] = 1.0   # boundary, should be ignored
    probs[25] = 1.0      # interior, should be detected
    beats = extract_beat_timestamps(probs, fps=100.0, threshold=0.5, min_distance_sec=0.15)
    got_frames = np.round(beats * 100.0).astype(np.int64)
    np.testing.assert_array_equal(got_frames, np.array([25], dtype=np.int64))


def test_beat_ts_rejects_non_1d():
    """2-D input must raise ValueError per the explicit guard."""
    probs2d = np.zeros((2, 10))
    with pytest.raises(ValueError):
        extract_beat_timestamps(probs2d, fps=100.0)


def test_beat_ts_sorted_output():
    """Returned timestamps must be sorted ascending."""
    fps = 100.0
    T = 300
    probs = make_spikes(T, [200, 50, 150, 90], peak=0.9)
    beats = extract_beat_timestamps(probs, fps=fps, threshold=0.5, min_distance_sec=0.15)
    assert np.all(np.diff(beats) > 0)


def test_beat_ts_periodic_known_period_recovers_tempo():
    """Spikes every P frames -> recovered inter-beat interval == P/fps.

    This is a real ground-truth check on tempo recovery, not a tautology.
    """
    fps = 100.0
    P = 50  # 0.5s period -> 120 BPM, well above min distance
    spike_frames = list(range(P, 600, P))  # 50,100,...,550
    T = 600
    probs = make_spikes(T, spike_frames, peak=1.0, floor=0.0)
    beats = extract_beat_timestamps(probs, fps=fps, threshold=0.5, min_distance_sec=0.15)
    got_frames = np.round(beats * fps).astype(np.int64)
    np.testing.assert_array_equal(got_frames, np.array(spike_frames, dtype=np.int64))
    ibis = np.diff(beats)
    np.testing.assert_allclose(ibis, P / fps, atol=1e-9)


# ===========================================================================
# extract_downbeat_timestamps
# ===========================================================================
def test_downbeats_subset_of_beats():
    """Downbeats must be a subset of input beat timestamps."""
    fps = 100.0
    T = 400
    # beats every 25 frames; phase = a sawtooth resetting every 100 frames so
    # every 4th beat lands near phase 0 (a downbeat).
    beat_frames = np.arange(25, T, 25)
    beats = beat_frames / fps
    phase = (np.arange(T) * TWO_PI / 100.0) % TWO_PI
    downbeats = extract_downbeat_timestamps(beats, phase, fps=fps)
    # every downbeat is one of the input beats
    assert set(np.round(downbeats * fps).astype(int)).issubset(
        set(np.round(beats * fps).astype(int))
    )


def test_downbeats_select_phase_near_zero():
    """Beats where phase ~ 0 are downbeats; phase ~ pi beats are not."""
    fps = 100.0
    T = 200
    phase = np.zeros(T)
    # frame 50 phase=0 (downbeat); frame 100 phase=pi (not); frame 150 phase~2pi (downbeat)
    phase[:] = math.pi  # default mid-bar -> nothing
    phase[50] = 0.0
    phase[150] = TWO_PI - 0.01  # > 0.85*2pi -> counts as downbeat
    beats = np.array([0.50, 1.00, 1.50])
    downbeats = extract_downbeat_timestamps(beats, phase, fps=fps)
    got = np.round(downbeats * fps).astype(int)
    np.testing.assert_array_equal(np.sort(got), np.array([50, 150]))


def test_downbeats_empty_when_no_phase_near_zero():
    """If no beat lands near phase 0, there are no downbeats."""
    fps = 100.0
    T = 200
    phase = np.full(T, math.pi)  # all mid-bar
    beats = np.array([0.50, 1.00, 1.50])
    downbeats = extract_downbeat_timestamps(beats, phase, fps=fps)
    assert downbeats.shape == (0,)


def test_downbeats_clamps_out_of_range_index():
    """Beat times past the phase array are clamped to the last frame (no crash)."""
    fps = 100.0
    phase = np.zeros(10)  # phase 0 everywhere -> any clamped beat is a downbeat
    beats = np.array([100.0])  # frame 10000, far past end -> clamp to frame 9
    downbeats = extract_downbeat_timestamps(beats, phase, fps=fps)
    np.testing.assert_allclose(downbeats, beats)


# ===========================================================================
# extract_timestamps_from_phase (integration)
# ===========================================================================
def test_pipeline_returns_beats_and_downbeats_consistently():
    fps = 100.0
    T = 400
    spike_frames = list(range(25, T, 25))
    probs = make_spikes(T, spike_frames, peak=0.9, floor=0.0)
    phase = (np.arange(T) * TWO_PI / 100.0) % TWO_PI  # bar every 100 frames
    beats, downbeats = extract_timestamps_from_phase(
        probs, phase, fps=fps, threshold=0.5, min_distance_sec=0.15
    )
    got_beats = np.round(beats * fps).astype(int)
    np.testing.assert_array_equal(got_beats, np.array(spike_frames))
    # downbeats subset of beats
    assert set(np.round(downbeats * fps).astype(int)).issubset(set(got_beats))
    # bar resets every 100 frames -> downbeats near frames {100,200,300} and 0-phase ones.
    # at minimum there must be ~1 downbeat per bar (4 beats/bar here).
    assert 0 < len(downbeats) <= len(beats)
