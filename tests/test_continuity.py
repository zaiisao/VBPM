import numpy as np

from vbpm.scoring.evaluation import continuity_scores


def test_double_time_separates_cmlt_from_the_octave_forgiving_score():
    truth = np.arange(0, 30.0, 0.5)
    doubled = np.arange(0, 30.0, 0.25)

    cmlt, amlt = continuity_scores(truth, doubled)

    assert cmlt < 0.05
    assert amlt > 0.95


def test_perfect_estimate_scores_one():
    truth = np.arange(0, 30.0, 0.5)

    cmlt, amlt = continuity_scores(truth, truth.copy())

    assert cmlt > 0.99
    assert amlt > 0.99


def test_degenerate_inputs_score_zero():
    truth = np.arange(0, 30.0, 0.5)

    assert continuity_scores(truth, np.zeros(0)) == (0.0, 0.0)
    assert continuity_scores(truth, np.array([1.0])) == (0.0, 0.0)
    assert continuity_scores(np.array([1.0]), truth) == (0.0, 0.0)
