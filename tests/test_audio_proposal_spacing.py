import math
from types import SimpleNamespace

import pytest
import torch

from diagnostics.audio_proposals import extract_audio_phase


def test_official_confidence_threshold_rejects_false_subdivision_peak():
    predictor = SimpleNamespace(
        prediction_weight=torch.tensor([[1.0], [0.0]]),
        prediction_bias=torch.zeros(2),
        proposal_meter=4,
        fps=50.0,
    )
    audio = torch.full((1, 60, 1), -10.0)
    audio[0, [10, 40], 0] = torch.logit(torch.tensor(0.9))
    audio[0, 25, 0] = torch.logit(torch.tensor(0.42))
    mask = torch.ones(1, 60)
    old = extract_audio_phase(predictor, audio, mask, max_bpm=300.0)
    repaired = extract_audio_phase(predictor, audio, mask, max_bpm=300.0, min_probability=0.5)
    conversion = 4 * 50 * 60 / (2 * math.pi)
    torch.testing.assert_close(old["physical_velocity"] * conversion, torch.full((1, 60), 200.0))
    torch.testing.assert_close(
        repaired["physical_velocity"] * conversion, torch.full((1, 60), 100.0)
    )


def test_optional_audio_peak_spacing_rejects_implausibly_fast_duplicate():
    predictor = SimpleNamespace(
        prediction_weight=torch.tensor([[1.0], [0.0]]),
        prediction_bias=torch.tensor([0.0, -10.0]),
        proposal_meter=4,
        fps=50.0,
    )
    audio = torch.full((1, 60, 1), -10.0)
    audio[0, [10, 15, 40], 0] = torch.tensor([15.0, 14.0, 15.0])
    mask = torch.ones(1, 60)
    original = extract_audio_phase(predictor, audio, mask)
    limited = extract_audio_phase(predictor, audio, mask, max_bpm=300.0)
    conversion = 4 * 50 * 60 / (2 * math.pi)
    assert original["physical_velocity"].max() * conversion > 300
    assert limited["physical_velocity"].max() * conversion <= 300
    # The weaker duplicate at frame 15 disappears; the two remaining peaks
    # at 10 and 40 define the 100 BPM proposal, including its extrapolation.
    torch.testing.assert_close(
        limited["physical_velocity"] * conversion, torch.full((1, 60), 100.0)
    )
    unchanged = extract_audio_phase(predictor, audio, mask, max_bpm=math.inf)
    torch.testing.assert_close(original["phase"], unchanged["phase"], rtol=0, atol=0)
    with pytest.raises(ValueError):
        extract_audio_phase(predictor, audio, mask, max_bpm=0)


@pytest.mark.parametrize(
    "positions, expected_periods",
    [
        ([10, 40, 100, 130], [30, 30, 30]),
        ([10, 40, 72, 106, 142], [30, 32, 34, 36]),
        ([10, 40, 70, 130, 190, 250], [30, 30, 60, 60, 60]),
    ],
)
def test_optional_missing_peak_recovery_preserves_ordinal_and_tempo(positions, expected_periods):
    predictor = SimpleNamespace(
        prediction_weight=torch.tensor([[1.0], [0.0]]),
        prediction_bias=torch.zeros(2),
        proposal_meter=4,
        fps=50.0,
    )
    audio = torch.full((1, positions[-1] + 10, 1), -10.0)
    audio[0, positions, 0] = 10.0
    mask = torch.ones(audio.shape[:2])
    default = extract_audio_phase(predictor, audio, mask, min_probability=0.5)
    explicit_default = extract_audio_phase(
        predictor, audio, mask, min_probability=0.5, recover_missing_beats=False
    )
    torch.testing.assert_close(default["phase"], explicit_default["phase"], rtol=0, atol=0)
    recovered = extract_audio_phase(
        predictor, audio, mask, min_probability=0.5, recover_missing_beats=True
    )
    conversion = 4 * 50 * 60 / (2 * math.pi)
    for left, right, period in zip(positions, positions[1:], expected_periods):
        bpm = recovered["physical_velocity"][0, (left + right) // 2] * conversion
        torch.testing.assert_close(bpm, torch.tensor(3000.0 / period))
    if positions == [10, 40, 100, 130]:
        # The gap contains two beats, without shifting subsequent bar indices.
        advance = recovered["phase"][0, 100] - recovered["phase"][0, 40]
        torch.testing.assert_close(advance, torch.tensor(math.pi))
