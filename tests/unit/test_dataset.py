"""Unit tests for training/dataset.py structured conversions.

Tested against closed-form math (the paper's unit conversions), structural
invariants (shapes, one-hot simplex, finiteness), and edge cases (wrap
detection at boundaries, meter clipping, legacy 3-col default, shift init).

Ground truth = independent numpy recomputation of the documented conversion,
NOT the function's own output.
"""

import sys
sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import numpy as np
import torch
import pytest

from training.dataset import (
    _phase_npy_to_structured,
    _build_prev_shifted,
    TWO_PI,
    _DEFAULT_NUM_METER_CLASSES,
)

RTOL = 1e-5
ATOL = 1e-6


# ----------------------------------------------------------------------------
# Helpers: build synthetic phase arrays [T, C].
# ----------------------------------------------------------------------------
def make_phase(tempo_bpf, beat_phase01, bar_phase01=None, meter_class=None):
    """Assemble a [T, 3] or [T, 4] phase array from column lists."""
    tempo_bpf = np.asarray(tempo_bpf, dtype=np.float64)
    beat_phase01 = np.asarray(beat_phase01, dtype=np.float64)
    T = len(tempo_bpf)
    if bar_phase01 is None:
        bar_phase01 = np.zeros(T, dtype=np.float64)
    else:
        bar_phase01 = np.asarray(bar_phase01, dtype=np.float64)
    cols = [tempo_bpf, beat_phase01, bar_phase01]
    if meter_class is not None:
        cols.append(np.asarray(meter_class, dtype=np.float64))
    return np.stack(cols, axis=1)


# ----------------------------------------------------------------------------
# log_tempo == log(2*pi * tempo_bpf)  (closed-form ground truth)
# ----------------------------------------------------------------------------
def test_log_tempo_matches_closed_form():
    tempo = [0.01, 0.02, 0.05, 0.1, 0.005]
    phase = make_phase(tempo, [0.0, 0.25, 0.5, 0.75, 0.9])
    out = _phase_npy_to_structured(phase)

    expected = np.log(np.maximum(np.asarray(tempo) * TWO_PI, 1e-8))
    got = out["log_tempo"].squeeze(-1).numpy()
    assert out["log_tempo"].shape == (5, 1)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_log_tempo_floor_on_zero_tempo():
    # tempo_bpf = 0 -> tempo_rad = 0 -> log(max(0,1e-8)) = log(1e-8), finite.
    phase = make_phase([0.0, 0.0], [0.1, 0.2])
    out = _phase_npy_to_structured(phase)
    got = out["log_tempo"].squeeze(-1).numpy()
    expected = np.full(2, math.log(1e-8), dtype=np.float64)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert np.isfinite(got).all()


# ----------------------------------------------------------------------------
# phase == 2*pi * beat_phase01  (closed-form ground truth)
# ----------------------------------------------------------------------------
def test_phase_matches_two_pi_scaling():
    bp = [0.0, 0.1, 0.25, 0.5, 0.75, 0.999]
    phase = make_phase([0.02] * 6, bp)
    out = _phase_npy_to_structured(phase)
    expected = np.asarray(bp) * TWO_PI
    got = out["phase"].squeeze(-1).numpy()
    assert out["phase"].shape == (6, 1)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_phase_range_within_zero_two_pi():
    rng = np.random.default_rng(0)
    bp = rng.uniform(0.0, 1.0, size=200)
    phase = make_phase([0.02] * 200, bp)
    out = _phase_npy_to_structured(phase)
    got = out["phase"].squeeze(-1).numpy()
    assert (got >= 0.0).all()
    assert (got < TWO_PI + ATOL).all()


# ----------------------------------------------------------------------------
# meter one-hot matches class index; simplex invariants.
# ----------------------------------------------------------------------------
def test_meter_onehot_matches_class():
    K = _DEFAULT_NUM_METER_CLASSES
    classes = [0, 1, 2, 3, K - 1]
    phase = make_phase([0.02] * 5, [0.1, 0.2, 0.3, 0.4, 0.5],
                       meter_class=classes)
    out = _phase_npy_to_structured(phase, num_meter_classes=K)

    oh = out["meter_onehot"].numpy()
    idx = out["meter_index"].numpy()
    assert oh.shape == (5, K)
    # exactly one hot per row
    assert np.array_equal(oh.sum(axis=1), np.ones(5))
    # argmax == declared class
    assert np.array_equal(oh.argmax(axis=1), np.asarray(classes))
    assert np.array_equal(idx, np.asarray(classes))
    # one-hot value exactly at the class column
    for t, c in enumerate(classes):
        assert oh[t, c] == 1.0


def test_meter_class_clipped_to_valid_range():
    # Out-of-range classes must be clipped to [0, K-1] (np.clip in source).
    K = _DEFAULT_NUM_METER_CLASSES
    raw = [-5, 0, K - 1, K, K + 100]
    phase = make_phase([0.02] * 5, [0.1] * 5, meter_class=raw)
    out = _phase_npy_to_structured(phase, num_meter_classes=K)
    idx = out["meter_index"].numpy()
    expected = np.clip(np.asarray(raw), 0, K - 1)
    assert np.array_equal(idx, expected)
    # one-hot must remain valid (sums to 1) even for clipped values
    assert np.array_equal(out["meter_onehot"].numpy().sum(axis=1), np.ones(5))


def test_meter_class_float_truncation():
    # meter column is cast via astype(np.int64): truncation toward zero.
    phase = make_phase([0.02] * 3, [0.1] * 3, meter_class=[2.9, 3.1, 0.7])
    out = _phase_npy_to_structured(phase)
    # int64 cast truncates: 2.9->2, 3.1->3, 0.7->0
    assert np.array_equal(out["meter_index"].numpy(), np.array([2, 3, 0]))


# ----------------------------------------------------------------------------
# beat_targets == 1 exactly where beat_phase01 wraps (decreases).
# ----------------------------------------------------------------------------
def test_beat_targets_mark_wraps():
    # phase ramps and wraps at indices 3 and 6.
    bp = [0.0, 0.4, 0.8, 0.1, 0.5, 0.9, 0.2, 0.6]
    #            wrap at idx3 (0.8->0.1), wrap at idx6 (0.9->0.2)
    phase = make_phase([0.02] * len(bp), bp)
    out = _phase_npy_to_structured(phase)
    bt = out["beat_targets"].numpy()

    # Independent ground-truth: target[t]=1 iff bp[t] < bp[t-1], target[0]=0.
    expected = np.zeros(len(bp), dtype=np.float32)
    bp_arr = np.asarray(bp)
    expected[1:] = (bp_arr[1:] < bp_arr[:-1]).astype(np.float32)
    np.testing.assert_array_equal(bt, expected)
    # explicit: wraps at 3 and 6
    assert bt[3] == 1.0 and bt[6] == 1.0
    assert bt.sum() == 2.0


def test_beat_targets_first_frame_always_zero():
    # Even if frame 0 "would" be a wrap relative to nothing, it is 0.
    bp = [0.99, 0.0, 0.0]  # second frame: 0.0 < 0.99 -> wrap
    phase = make_phase([0.02] * 3, bp)
    out = _phase_npy_to_structured(phase)
    bt = out["beat_targets"].numpy()
    assert bt[0] == 0.0
    assert bt[1] == 1.0
    # 0.0 < 0.0 is False -> no wrap at idx 2
    assert bt[2] == 0.0


def test_beat_targets_equal_phase_no_wrap():
    # Strictly-decreasing check: equal consecutive values are NOT a wrap.
    bp = [0.5, 0.5, 0.5]
    phase = make_phase([0.02] * 3, bp)
    out = _phase_npy_to_structured(phase)
    assert out["beat_targets"].numpy().sum() == 0.0


def test_beat_targets_monotone_increasing_no_wrap():
    bp = np.linspace(0.0, 0.99, 50)
    phase = make_phase([0.02] * 50, bp)
    out = _phase_npy_to_structured(phase)
    assert out["beat_targets"].numpy().sum() == 0.0


def test_beat_targets_count_matches_period():
    # A perfect sawtooth with period P over T frames should have ~T/P wraps.
    T = 100
    P = 10  # wrap every 10 frames
    t = np.arange(T)
    bp = (t % P) / P  # 0,0.1,...,0.9,0,0.1,...
    phase = make_phase([1.0 / P] * T, bp)
    out = _phase_npy_to_structured(phase)
    bt = out["beat_targets"].numpy()
    # wraps occur at t = P, 2P, ... (T/P - 1) interior + the boundary ones
    n_wraps = int(bt.sum())
    # exactly floor((T-1)/P) wraps for this construction
    expected_wraps = (T - 1) // P
    assert n_wraps == expected_wraps


# ----------------------------------------------------------------------------
# Legacy 3-col defaults meter to 4/4 (source comment: "class index 2").
# ----------------------------------------------------------------------------
def test_three_col_defaults_meter_to_class_2():
    # 3-col array: no meter column. Source defaults to class 2.
    phase = make_phase([0.02, 0.02], [0.1, 0.2])  # shape [2,3]
    assert phase.shape == (2, 3)
    out = _phase_npy_to_structured(phase)
    idx = out["meter_index"].numpy()
    assert np.array_equal(idx, np.array([2, 2]))
    # one-hot hot at column 2
    oh = out["meter_onehot"].numpy()
    assert (oh[:, 2] == 1.0).all()
    assert oh.sum(axis=1).tolist() == [1.0, 1.0]


def test_four_col_uses_explicit_meter_not_default():
    # When a 4th column is present, that value (not the legacy default 2) is used.
    phase = make_phase([0.02, 0.02], [0.1, 0.2], meter_class=[5, 6])
    assert phase.shape == (2, 4)
    out = _phase_npy_to_structured(phase)
    assert np.array_equal(out["meter_index"].numpy(), np.array([5, 6]))


# ----------------------------------------------------------------------------
# Output dict structure / dtypes / shapes / finiteness.
# ----------------------------------------------------------------------------
def test_structured_keys_and_dtypes():
    phase = make_phase([0.02] * 4, [0.0, 0.3, 0.6, 0.9], meter_class=[0, 1, 2, 3])
    out = _phase_npy_to_structured(phase)
    assert set(out.keys()) == {
        "phase", "log_tempo", "meter_index", "meter_onehot", "beat_targets"
    }
    assert out["phase"].dtype == torch.float32
    assert out["log_tempo"].dtype == torch.float32
    assert out["meter_index"].dtype == torch.long
    assert out["meter_onehot"].dtype == torch.float32
    assert out["beat_targets"].dtype == torch.float32
    # shapes
    assert out["phase"].shape == (4, 1)
    assert out["log_tempo"].shape == (4, 1)
    assert out["meter_index"].shape == (4,)
    assert out["meter_onehot"].shape == (4, _DEFAULT_NUM_METER_CLASSES)
    assert out["beat_targets"].shape == (4,)


def test_all_outputs_finite():
    rng = np.random.default_rng(1)
    T = 64
    phase = make_phase(
        rng.uniform(0.005, 0.1, T),
        rng.uniform(0.0, 1.0, T),
        meter_class=rng.integers(0, _DEFAULT_NUM_METER_CLASSES, T),
    )
    out = _phase_npy_to_structured(phase)
    for k, v in out.items():
        assert torch.isfinite(v.float()).all(), f"non-finite in {k}"


def test_custom_num_meter_classes_changes_onehot_width():
    for K in (3, 4, 8, 12):
        phase = make_phase([0.02] * 3, [0.1, 0.2, 0.3], meter_class=[0, 1, 2])
        out = _phase_npy_to_structured(phase, num_meter_classes=K)
        assert out["meter_onehot"].shape == (3, K)


# ----------------------------------------------------------------------------
# _build_prev_shifted: output[t] == input[t-1]; first-frame init.
# ----------------------------------------------------------------------------
def test_prev_shifted_phase_log_tempo_shift():
    phase = make_phase([0.01, 0.02, 0.03, 0.04], [0.1, 0.4, 0.7, 0.95],
                       meter_class=[0, 1, 2, 3])
    structured = _phase_npy_to_structured(phase)
    prev = _build_prev_shifted(structured)

    ph = structured["phase"]
    lt = structured["log_tempo"]

    # output[t] == input[t-1] for t>=1
    assert torch.equal(prev["phase_prev"][1:], ph[:-1])
    assert torch.equal(prev["log_tempo_prev"][1:], lt[:-1])
    # first frame zero-initialized
    assert torch.equal(prev["phase_prev"][0], torch.zeros_like(ph[0]))
    assert torch.equal(prev["log_tempo_prev"][0], torch.zeros_like(lt[0]))


def test_prev_shifted_meter_first_frame_uniform():
    K = _DEFAULT_NUM_METER_CLASSES
    phase = make_phase([0.02] * 4, [0.1, 0.2, 0.3, 0.4],
                       meter_class=[1, 2, 3, 4])
    structured = _phase_npy_to_structured(phase, num_meter_classes=K)
    prev = _build_prev_shifted(structured)

    moh = structured["meter_onehot"]
    moh_prev = prev["meter_onehot_prev"]

    # shifted: prev[t] == onehot[t-1] for t>=1
    assert torch.equal(moh_prev[1:], moh[:-1])
    # first frame is uniform 1/K
    expected_first = torch.full((K,), 1.0 / K)
    assert torch.allclose(moh_prev[0], expected_first, atol=1e-7)
    # uniform row sums to 1 (valid distribution)
    assert torch.allclose(moh_prev[0].sum(), torch.tensor(1.0), atol=1e-6)


def test_prev_shifted_keys_and_shapes():
    phase = make_phase([0.02] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])
    structured = _phase_npy_to_structured(phase)
    prev = _build_prev_shifted(structured)
    assert set(prev.keys()) == {"phase_prev", "log_tempo_prev", "meter_onehot_prev"}
    assert prev["phase_prev"].shape == structured["phase"].shape
    assert prev["log_tempo_prev"].shape == structured["log_tempo"].shape
    assert prev["meter_onehot_prev"].shape == structured["meter_onehot"].shape


def test_prev_shifted_does_not_mutate_input():
    phase = make_phase([0.02] * 4, [0.1, 0.2, 0.3, 0.4], meter_class=[0, 1, 2, 3])
    structured = _phase_npy_to_structured(phase)
    phase_before = structured["phase"].clone()
    lt_before = structured["log_tempo"].clone()
    moh_before = structured["meter_onehot"].clone()
    _ = _build_prev_shifted(structured)
    assert torch.equal(structured["phase"], phase_before)
    assert torch.equal(structured["log_tempo"], lt_before)
    assert torch.equal(structured["meter_onehot"], moh_before)


def test_prev_shifted_single_frame():
    # T=1: shift produces only the init frame, no history.
    phase = make_phase([0.02], [0.3], meter_class=[2])
    structured = _phase_npy_to_structured(phase)
    prev = _build_prev_shifted(structured)
    K = _DEFAULT_NUM_METER_CLASSES
    assert torch.equal(prev["phase_prev"][0], torch.zeros(1))
    assert torch.equal(prev["log_tempo_prev"][0], torch.zeros(1))
    assert torch.allclose(prev["meter_onehot_prev"][0], torch.full((K,), 1.0 / K))


# ----------------------------------------------------------------------------
# End-to-end consistency: phase wrap implies a beat target.
# A reconstructed beat from phase/log_tempo should be self-consistent.
# ----------------------------------------------------------------------------
def test_phase_and_beat_targets_consistent_on_sawtooth():
    # Build a clean sawtooth; every wrap in phase (2pi->0) must coincide
    # with a beat_target == 1.
    T = 60
    P = 12
    bp = (np.arange(T) % P) / P
    phase = make_phase([1.0 / P] * T, bp)
    out = _phase_npy_to_structured(phase)
    phase_rad = out["phase"].squeeze(-1).numpy()
    bt = out["beat_targets"].numpy()
    # detect 2pi wraps in radian phase independently
    drad = np.diff(phase_rad, prepend=phase_rad[0])
    rad_wraps = (drad < -math.pi).astype(np.float32)  # large negative jump
    np.testing.assert_array_equal(bt, rad_wraps)
