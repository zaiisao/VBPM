"""Unit tests for the schedule helper functions in training/train.py.

These functions are pure (epoch -> float) annealing schedules:

    _gumbel_temperature(epoch, num_epochs, start, end)
    _kl_beta(epoch, anneal_epochs)
    _ss_eps(epoch, anneal_epochs, max_eps)

We test them against closed-form linear-interpolation ground truth and
against structural invariants (boundary values, monotonicity, clamping,
range bounds). No tautologies: every assertion compares the implementation
to an independently-computed reference or a mathematical property.
"""

import sys

sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import pytest

from training.train import _gumbel_temperature, _kl_beta, _ss_eps


# ---------------------------------------------------------------------------
# Closed-form reference implementations (independent of the code under test).
# Derived from the docstrings / intended semantics, NOT copied from source.
# ---------------------------------------------------------------------------

def ref_gumbel_temperature(epoch, num_epochs, start, end):
    """Linear interp from `start` (epoch 0) to `end` (epoch num_epochs-1),
    clamped so epochs beyond the last stay at `end`."""
    if num_epochs <= 1:
        return end
    last = num_epochs - 1
    t = min(epoch / last, 1.0)
    return start + (end - start) * t


def ref_kl_beta(epoch, anneal_epochs):
    """Linear ramp 0 -> 1 over `anneal_epochs`; constant 1.0 if no annealing."""
    if anneal_epochs <= 0:
        return 1.0
    return min(epoch / anneal_epochs, 1.0)


# ---------------------------------------------------------------------------
# _gumbel_temperature
# ---------------------------------------------------------------------------

def test_gumbel_start_at_epoch_zero():
    # At epoch 0 the temperature must equal `start` exactly.
    for start, end, N in [(5.0, 0.5, 10), (1.0, 0.1, 50), (2.0, 2.0, 7), (0.3, 3.0, 4)]:
        assert _gumbel_temperature(0, N, start, end) == pytest.approx(start)


def test_gumbel_end_at_last_epoch():
    # At the final epoch (num_epochs - 1) the temperature must equal `end`.
    for start, end, N in [(5.0, 0.5, 10), (1.0, 0.1, 50), (3.0, 0.2, 2)]:
        assert _gumbel_temperature(N - 1, N, start, end) == pytest.approx(end)


def test_gumbel_matches_closed_form():
    # Compare against an independent linear-interp reference over a grid.
    for start, end, N in [(5.0, 0.5, 10), (0.2, 4.0, 13), (1.0, 1.0, 8)]:
        for epoch in range(0, N + 3):  # include epochs past the end (clamp region)
            got = _gumbel_temperature(epoch, N, start, end)
            exp = ref_gumbel_temperature(epoch, N, start, end)
            assert got == pytest.approx(exp), (epoch, N, start, end, got, exp)


def test_gumbel_monotonic_decreasing_when_start_gt_end():
    start, end, N = 5.0, 0.5, 20
    vals = [_gumbel_temperature(e, N, start, end) for e in range(N)]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12, (a, b)
    # And strictly decreasing somewhere (not flat) since start != end.
    assert vals[0] > vals[-1]


def test_gumbel_monotonic_increasing_when_start_lt_end():
    start, end, N = 0.1, 3.0, 15
    vals = [_gumbel_temperature(e, N, start, end) for e in range(N)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-12, (a, b)
    assert vals[-1] > vals[0]


def test_gumbel_stays_within_bounds():
    # Output never leaves [min(start,end), max(start,end)], including past-end epochs.
    start, end, N = 4.0, 0.25, 12
    lo, hi = min(start, end), max(start, end)
    for epoch in range(0, N + 5):
        v = _gumbel_temperature(epoch, N, start, end)
        assert lo - 1e-12 <= v <= hi + 1e-12, (epoch, v)


def test_gumbel_clamps_past_last_epoch():
    # Epochs beyond num_epochs-1 should clamp to `end`, never overshoot.
    start, end, N = 5.0, 0.5, 10
    for epoch in [N - 1, N, N + 1, 100]:
        assert _gumbel_temperature(epoch, N, start, end) == pytest.approx(end)


def test_gumbel_degenerate_num_epochs():
    # num_epochs <= 1 has no interpolation interval; must return `end`.
    assert _gumbel_temperature(0, 1, 5.0, 0.5) == pytest.approx(0.5)
    assert _gumbel_temperature(0, 0, 5.0, 0.5) == pytest.approx(0.5)
    assert _gumbel_temperature(3, 1, 5.0, 0.5) == pytest.approx(0.5)


def test_gumbel_midpoint_value():
    # At the exact midpoint epoch of an odd-length schedule, value is the
    # arithmetic mean of start and end.
    start, end, N = 6.0, 0.0, 5  # last epoch = 4, midpoint epoch = 2 -> t = 0.5
    mid = _gumbel_temperature(2, N, start, end)
    assert mid == pytest.approx((start + end) / 2.0)


def test_gumbel_constant_when_start_eq_end():
    start = end = 1.7
    for epoch in range(0, 12):
        assert _gumbel_temperature(epoch, 10, start, end) == pytest.approx(1.7)


# ---------------------------------------------------------------------------
# _kl_beta
# ---------------------------------------------------------------------------

def test_kl_beta_zero_at_epoch_zero():
    # With positive annealing, beta starts at 0.
    for anneal in [1, 5, 20]:
        assert _kl_beta(0, anneal) == pytest.approx(0.0)


def test_kl_beta_reaches_one_at_anneal_and_stays():
    anneal = 8
    assert _kl_beta(anneal, anneal) == pytest.approx(1.0)
    for epoch in range(anneal, anneal + 10):
        assert _kl_beta(epoch, anneal) == pytest.approx(1.0)


def test_kl_beta_matches_closed_form_ramp():
    for anneal in [1, 4, 10]:
        for epoch in range(0, anneal + 5):
            assert _kl_beta(epoch, anneal) == pytest.approx(ref_kl_beta(epoch, anneal))


def test_kl_beta_one_when_no_annealing():
    # anneal <= 0 => annealing disabled => beta is always 1.0.
    for anneal in [0, -1, -7]:
        for epoch in [0, 1, 3, 100]:
            assert _kl_beta(epoch, anneal) == pytest.approx(1.0)


def test_kl_beta_monotonic_nondecreasing():
    anneal = 12
    vals = [_kl_beta(e, anneal) for e in range(0, anneal + 6)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-12, (a, b)


def test_kl_beta_in_unit_interval():
    anneal = 9
    for epoch in range(0, anneal + 6):
        v = _kl_beta(epoch, anneal)
        assert 0.0 - 1e-12 <= v <= 1.0 + 1e-12, (epoch, v)


def test_kl_beta_half_at_midpoint():
    anneal = 10
    assert _kl_beta(5, anneal) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _ss_eps  (scheduled-sampling probability)
# ---------------------------------------------------------------------------

def test_ss_eps_zero_when_max_zero():
    # max_eps == 0 (or negative) disables scheduled sampling entirely.
    for max_eps in [0.0, -0.1]:
        for epoch in range(0, 30):
            assert _ss_eps(epoch, 5, max_eps) == pytest.approx(0.0)


def test_ss_eps_zero_until_kl_anneal_finishes():
    # Stays exactly 0 for every epoch strictly before `anneal_epochs`.
    anneal, max_eps = 10, 0.3
    for epoch in range(0, anneal):
        assert _ss_eps(epoch, anneal, max_eps) == pytest.approx(0.0), epoch
    # And is still 0 right at the start of the ramp (epoch == anneal).
    assert _ss_eps(anneal, anneal, max_eps) == pytest.approx(0.0)


def test_ss_eps_ramps_to_max_and_clamps():
    # After ramping for `ramp_len` epochs past anneal, eps reaches max and clamps.
    anneal, max_eps = 6, 0.4
    ramp_len = max(1, anneal)  # 6
    # End of ramp: epoch = anneal + ramp_len -> exactly max.
    assert _ss_eps(anneal + ramp_len, anneal, max_eps) == pytest.approx(max_eps)
    # Beyond the ramp it must clamp, never exceed max.
    for epoch in range(anneal + ramp_len, anneal + ramp_len + 20):
        v = _ss_eps(epoch, anneal, max_eps)
        assert v == pytest.approx(max_eps), (epoch, v)


def test_ss_eps_never_exceeds_max_and_nonnegative():
    anneal, max_eps = 7, 0.25
    for epoch in range(0, 60):
        v = _ss_eps(epoch, anneal, max_eps)
        assert 0.0 - 1e-12 <= v <= max_eps + 1e-12, (epoch, v)


def test_ss_eps_matches_closed_form_in_ramp_region():
    # In the ramp region eps = max_eps * (epoch - anneal) / ramp_len.
    anneal, max_eps = 8, 0.5
    ramp_len = max(1, anneal)
    for epoch in range(anneal, anneal + ramp_len + 1):
        expected = max_eps * min(1.0, max(0.0, (epoch - anneal) / ramp_len))
        assert _ss_eps(epoch, anneal, max_eps) == pytest.approx(expected), epoch


def test_ss_eps_monotonic_nondecreasing():
    anneal, max_eps = 5, 0.3
    vals = [_ss_eps(e, anneal, max_eps) for e in range(0, anneal * 3 + 5)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-12, (a, b)


def test_ss_eps_halfway_value():
    # At the midpoint of the ramp the value is max_eps/2.
    anneal, max_eps = 6, 0.4
    ramp_len = max(1, anneal)  # 6
    mid_epoch = anneal + ramp_len // 2  # 6 + 3 = 9 -> t = 3/6 = 0.5
    assert _ss_eps(mid_epoch, anneal, max_eps) == pytest.approx(max_eps / 2.0)


def test_ss_eps_anneal_nonpositive_ramp_len_floor():
    # With anneal <= 0 the ramp_len is floored at 1 and the offset is (epoch-anneal).
    # Verify it still produces a valid, bounded, nondecreasing ramp toward max.
    anneal, max_eps = 0, 0.5
    ramp_len = max(1, anneal)  # 1
    for epoch in range(0, 6):
        expected = max_eps * min(1.0, max(0.0, (epoch - anneal) / ramp_len))
        assert _ss_eps(epoch, anneal, max_eps) == pytest.approx(expected), epoch
    # Reaches max quickly (epoch 1) and clamps.
    assert _ss_eps(1, anneal, max_eps) == pytest.approx(max_eps)
    assert _ss_eps(5, anneal, max_eps) == pytest.approx(max_eps)


# ---------------------------------------------------------------------------
# Cross-function finiteness sanity (no NaN/inf for ordinary inputs).
# ---------------------------------------------------------------------------

def test_all_schedules_finite():
    for epoch in range(0, 25):
        assert math.isfinite(_gumbel_temperature(epoch, 20, 5.0, 0.5))
        assert math.isfinite(_kl_beta(epoch, 10))
        assert math.isfinite(_ss_eps(epoch, 8, 0.3))
