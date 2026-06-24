"""Unit tests for evaluation/score.py.

Ground truth strategy:
- frames_to_beat_times: closed-form idx/fps.
- evaluate_beats / evaluate_downbeats: cross-checked against direct
  mir_eval.beat calls on the SAME (trimmed/sorted/unique) arrays, plus
  known invariants (ref==est -> F==1 and CMLt==1; >70ms shift -> F<1).

IMPORTANT mir_eval subtlety: mir_eval.beat.evaluate() internally calls
trim_beats(beats, min_beat_time=5.0), which DROPS every beat before 5.0s.
So to exercise continuity metrics (CMLt etc.) the synthetic beat grids
must extend well past 5 seconds. The F-measure cross-check below therefore
compares against f_measure(trim_beats(ref), trim_beats(est)) -- NOT the raw
arrays -- because that is exactly what evaluate() computes.
"""

import sys
sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import warnings

import numpy as np
import pytest
import mir_eval.beat

from evaluation.score import (
    frames_to_beat_times,
    evaluate_beats,
    evaluate_downbeats,
)


# ---------------------------------------------------------------------------
# frames_to_beat_times
# ---------------------------------------------------------------------------

def test_frames_to_beat_times_closed_form():
    fps = 50.0
    targets = np.zeros(20, dtype=np.float64)
    beat_idx = np.array([0, 3, 7, 12, 19])
    targets[beat_idx] = 1.0
    out = frames_to_beat_times(targets, fps)
    expected = beat_idx.astype(np.float64) / fps
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("fps", [25.0, 44.1, 100.0])
def test_frames_to_beat_times_various_fps(fps):
    targets = np.zeros(30, dtype=np.float64)
    idx = np.array([2, 11, 23])
    targets[idx] = 1.0
    out = frames_to_beat_times(targets, fps)
    np.testing.assert_allclose(out, idx / fps)


def test_frames_to_beat_times_threshold_is_half():
    # Values <= 0.5 must NOT count; values > 0.5 must count.
    targets = np.array([0.0, 0.5, 0.50001, 0.49, 1.0])
    out = frames_to_beat_times(targets, 10.0)
    # only indices 2 (0.50001) and 4 (1.0) exceed 0.5
    np.testing.assert_allclose(out, np.array([2, 4]) / 10.0)


def test_frames_to_beat_times_empty_and_no_beats():
    # No beats -> empty array, no crash.
    out = frames_to_beat_times(np.zeros(10), 50.0)
    assert out.shape == (0,)
    # Empty input -> empty output.
    out2 = frames_to_beat_times(np.array([]), 50.0)
    assert out2.shape == (0,)


def test_frames_to_beat_times_is_sorted_and_seconds():
    # Indices come from np.where (already ascending) -> times ascending.
    targets = np.zeros(100)
    targets[[5, 50, 99]] = 1.0
    out = frames_to_beat_times(targets, 25.0)
    assert np.all(np.diff(out) > 0)
    # first beat at frame 5, fps 25 -> 0.2s
    assert out[0] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Helpers for beat-grid construction (past the 5s trim window)
# ---------------------------------------------------------------------------

def _beat_grid(period=0.5, n=40, start=0.5):
    """Beats from `start` to start+(n-1)*period; n=40,period=0.5 -> up to ~20s."""
    return start + np.arange(n) * period


# ---------------------------------------------------------------------------
# evaluate_beats
# ---------------------------------------------------------------------------

def test_evaluate_beats_return_keys():
    ref = _beat_grid()
    out = evaluate_beats(ref, ref.copy())
    assert set(out.keys()) == {"F-measure", "CMLc", "CMLt", "AMLc", "AMLt"}
    for v in out.values():
        assert np.isfinite(v)
        assert 0.0 <= v <= 1.0


def test_evaluate_beats_perfect_match():
    # ref == est -> F-measure == 1.0 and CMLt == 1.0
    ref = _beat_grid()
    out = evaluate_beats(ref, ref.copy())
    assert out["F-measure"] == pytest.approx(1.0)
    assert out["CMLt"] == pytest.approx(1.0)
    assert out["CMLc"] == pytest.approx(1.0)
    assert out["AMLt"] == pytest.approx(1.0)


def test_evaluate_beats_f_measure_cross_check_perfect():
    # Cross-check F-measure against a direct mir_eval.beat.f_measure call on
    # the SAME arrays that evaluate() uses internally (trimmed).
    ref = _beat_grid()
    est = ref.copy()
    out = evaluate_beats(ref, est)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = mir_eval.beat.f_measure(
            mir_eval.beat.trim_beats(np.sort(np.unique(ref))),
            mir_eval.beat.trim_beats(np.sort(np.unique(est))),
        )
    assert out["F-measure"] == pytest.approx(direct)


def test_evaluate_beats_f_measure_cross_check_noisy():
    # Add sub-threshold jitter so some beats hit and some structure remains,
    # then cross-check the wrapper's F-measure against direct mir_eval.
    rng = np.random.default_rng(0)
    ref = _beat_grid(period=0.5, n=40)
    est = ref + rng.uniform(-0.04, 0.04, size=ref.shape)  # < 70 ms jitter
    out = evaluate_beats(ref, est)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = mir_eval.beat.f_measure(
            mir_eval.beat.trim_beats(np.sort(np.unique(ref))),
            mir_eval.beat.trim_beats(np.sort(np.unique(est))),
        )
    assert out["F-measure"] == pytest.approx(direct)


def test_evaluate_beats_shift_over_threshold_drops_f():
    # mir_eval f_measure threshold is 70 ms. A uniform 80 ms shift makes every
    # beat miss -> F < 1.0 (in fact 0.0).
    ref = _beat_grid()
    est = ref + 0.080
    out = evaluate_beats(ref, est)
    assert out["F-measure"] < 1.0
    assert out["F-measure"] == pytest.approx(0.0)


def test_evaluate_beats_shift_under_threshold_keeps_f():
    # A 50 ms shift is below the 70 ms tolerance -> all beats still hit, F==1.
    ref = _beat_grid()
    est = ref + 0.050
    out = evaluate_beats(ref, est)
    assert out["F-measure"] == pytest.approx(1.0)


def test_evaluate_beats_full_cross_check_all_metrics():
    # Cross-check EVERY returned metric against mir_eval.beat.evaluate on the
    # canonicalized arrays the wrapper builds.
    rng = np.random.default_rng(7)
    ref = _beat_grid(period=0.5, n=40)
    est = ref + rng.uniform(-0.03, 0.03, size=ref.shape)
    ref_c = np.sort(np.unique(ref))
    est_c = np.sort(np.unique(est))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = mir_eval.beat.evaluate(ref_c, est_c)
    out = evaluate_beats(ref, est)
    assert out["F-measure"] == pytest.approx(gt["F-measure"])
    assert out["CMLc"] == pytest.approx(gt["Correct Metric Level Continuous"])
    assert out["CMLt"] == pytest.approx(gt["Correct Metric Level Total"])
    assert out["AMLc"] == pytest.approx(gt["Any Metric Level Continuous"])
    assert out["AMLt"] == pytest.approx(gt["Any Metric Level Total"])


def test_evaluate_beats_empty_inputs_return_zeros_no_crash():
    out = evaluate_beats(np.array([]), np.array([]))
    assert out == {"F-measure": 0.0, "CMLc": 0.0, "CMLt": 0.0,
                   "AMLc": 0.0, "AMLt": 0.0}
    # one side empty
    out2 = evaluate_beats(_beat_grid(), np.array([]))
    assert out2["F-measure"] == 0.0
    out3 = evaluate_beats(np.array([]), _beat_grid())
    assert out3["F-measure"] == 0.0


def test_evaluate_beats_handles_unsorted_and_duplicates():
    # Wrapper sorts and uniques; perfect match modulo order/dupes -> F==1.
    ref = _beat_grid()
    est = np.concatenate([ref[::-1], ref[:3]])  # reversed + duplicates
    out = evaluate_beats(ref, est)
    assert out["F-measure"] == pytest.approx(1.0)


def test_evaluate_beats_2d_input_is_raveled():
    # Wrapper ravels; column vector should behave like the 1-D version.
    ref = _beat_grid()
    out_1d = evaluate_beats(ref, ref.copy())
    out_2d = evaluate_beats(ref.reshape(-1, 1), ref.reshape(-1, 1))
    assert out_2d["F-measure"] == pytest.approx(out_1d["F-measure"])


def test_evaluate_beats_does_not_crash_single_beat():
    # Degenerate: a single beat each. mir_eval warns but must not raise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = evaluate_beats(np.array([6.0]), np.array([6.0]))
    for v in out.values():
        assert np.isfinite(v)


# ---------------------------------------------------------------------------
# evaluate_downbeats
# ---------------------------------------------------------------------------

def test_evaluate_downbeats_return_keys():
    ref = _beat_grid(period=2.0, n=12, start=2.0)  # downbeats up to ~24s
    out = evaluate_downbeats(ref, ref.copy())
    assert set(out.keys()) == {"db_F-measure", "db_CMLc", "db_CMLt",
                               "db_AMLc", "db_AMLt"}
    for v in out.values():
        assert np.isfinite(v)
        assert 0.0 <= v <= 1.0


def test_evaluate_downbeats_perfect_match():
    ref = _beat_grid(period=2.0, n=12, start=2.0)
    out = evaluate_downbeats(ref, ref.copy())
    assert out["db_F-measure"] == pytest.approx(1.0)
    assert out["db_CMLt"] == pytest.approx(1.0)


def test_evaluate_downbeats_f_measure_cross_check():
    ref = _beat_grid(period=2.0, n=12, start=2.0)
    rng = np.random.default_rng(3)
    est = ref + rng.uniform(-0.03, 0.03, size=ref.shape)
    out = evaluate_downbeats(ref, est)
    ref_c = np.sort(np.unique(ref))
    est_c = np.sort(np.unique(est))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = mir_eval.beat.f_measure(
            mir_eval.beat.trim_beats(ref_c),
            mir_eval.beat.trim_beats(est_c),
        )
    assert out["db_F-measure"] == pytest.approx(direct)


def test_evaluate_downbeats_shift_over_threshold_drops_f():
    ref = _beat_grid(period=2.0, n=12, start=2.0)
    est = ref + 0.080
    out = evaluate_downbeats(ref, est)
    assert out["db_F-measure"] < 1.0


def test_evaluate_downbeats_fewer_than_two_returns_zeros():
    # Guard: len < 2 on either side -> all-zero dict, no crash.
    z = {"db_F-measure": 0.0, "db_CMLc": 0.0, "db_CMLt": 0.0,
         "db_AMLc": 0.0, "db_AMLt": 0.0}
    assert evaluate_downbeats(np.array([6.0]), _beat_grid(period=2.0, n=12, start=2.0)) == z
    assert evaluate_downbeats(_beat_grid(period=2.0, n=12, start=2.0), np.array([6.0])) == z
    assert evaluate_downbeats(np.array([]), np.array([])) == z


def test_evaluate_downbeats_full_cross_check_all_metrics():
    ref = _beat_grid(period=2.0, n=12, start=2.0)
    rng = np.random.default_rng(11)
    est = ref + rng.uniform(-0.02, 0.02, size=ref.shape)
    ref_c = np.sort(np.unique(ref))
    est_c = np.sort(np.unique(est))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = mir_eval.beat.evaluate(ref_c, est_c)
    out = evaluate_downbeats(ref, est)
    assert out["db_F-measure"] == pytest.approx(gt["F-measure"])
    assert out["db_CMLc"] == pytest.approx(gt["Correct Metric Level Continuous"])
    assert out["db_CMLt"] == pytest.approx(gt["Correct Metric Level Total"])
    assert out["db_AMLc"] == pytest.approx(gt["Any Metric Level Continuous"])
    assert out["db_AMLt"] == pytest.approx(gt["Any Metric Level Total"])


# ---------------------------------------------------------------------------
# end-to-end: frames -> times -> evaluate
# ---------------------------------------------------------------------------

def test_pipeline_frames_to_eval_perfect():
    # Build a binary frame target, convert to times, score against itself.
    fps = 50.0
    T = 1200  # 24s at 50 fps
    targets = np.zeros(T)
    # a beat every 25 frames (0.5s) -> tempo 120 bpm, extends well past 5s
    targets[::25] = 1.0
    times = frames_to_beat_times(targets, fps)
    assert times[-1] > 5.0  # ensure trim window is cleared
    out = evaluate_beats(times, times.copy())
    assert out["F-measure"] == pytest.approx(1.0)
    assert out["CMLt"] == pytest.approx(1.0)
