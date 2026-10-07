import numpy as np
import pytest

from vbpm.scoring.evaluation import decode_event_times, f_measure


def test_adjacent_positive_frames_form_one_peak():
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
            [0.9, 0.05, 0.05],
        ]
    )
    beats, downbeats = decode_event_times(probs, 50)
    np.testing.assert_allclose(beats, [0.04, 0.10])
    np.testing.assert_allclose(downbeats, [0.10])


def test_separate_regions_and_empty_predictions():
    beats, downbeats = decode_event_times(
        np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]), 50
    )
    np.testing.assert_allclose(beats, [0.0, 0.04])
    assert not len(downbeats)
    beats, downbeats = decode_event_times(np.empty((0, 3)), 50)
    assert not len(beats) and not len(downbeats)


def test_seventy_ms_acceptance_and_one_to_one_matching():
    assert f_measure([0.07], [0.0])[0] == 1.0
    assert f_measure([-0.07], [0.0])[0] == 1.0
    assert f_measure([0.071], [0.0])[0] == 0.0
    assert f_measure([0.97, 1.03], [1.0])[0] == pytest.approx(2 / 3)


@pytest.mark.parametrize("origin", [1.0, 1000.0, -2.0])
def test_acceptance_boundary_at_nonzero_timestamps(origin):
    assert f_measure([origin + 0.07], [origin])[0] == 1.0
    assert f_measure([origin - 0.07], [origin])[0] == 1.0
    assert f_measure([origin + 0.071], [origin])[0] == 0.0
    assert f_measure([origin - 0.071], [origin])[0] == 0.0
