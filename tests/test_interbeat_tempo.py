import numpy as np
import pytest
import torch

from diagnostics.real_event_references import interbeat_tempo_errors


def test_annotated_intervals_measure_tempo_independent_of_phase_origin():
    time = torch.arange(256, dtype=torch.float64) / 50
    # 120 BPM in meter four: pi radians per second.
    phase = (np.pi * time)[None]
    beats = [np.arange(0.0, 5.01, 0.5)]
    result = interbeat_tempo_errors(phase, beats)
    shifted = interbeat_tempo_errors(phase + 2.3, beats)
    assert result["interbeat_tempo_RMSE_BPM"] == pytest.approx(0.0, abs=1e-10)
    assert shifted["interbeat_tempo_RMSE_BPM"] == pytest.approx(0.0, abs=1e-10)
    assert result["interbeat_interval_count"] == 10


def test_outside_window_annotations_do_not_create_observed_intervals():
    phase = torch.zeros(1, 50)
    result = interbeat_tempo_errors(phase, [[-1.0, 0.5, 2.0]])
    assert result["interbeat_interval_count"] == 0
    assert result["interbeat_tempo_RMSE_BPM"] is None
