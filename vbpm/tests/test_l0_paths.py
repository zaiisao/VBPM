import math

import numpy as np
import pytest
import torch

from vbpm.model import VBPM
from vbpm.tests.oracle import (bar_meters, beat_positions, oracle_draw, synthetic_song)

FPS = 50.0
FRAMES = 1500


@pytest.fixture(scope="module")
def model():
    return VBPM(input_dim=8)


def _window(bars, bpm, rubato=0.0, seed=0, start=3.0):
    beat_times, downbeat_times = synthetic_song(bars, bpm, FPS, rubato=rubato, seed=seed)
    positions = beat_positions(beat_times, downbeat_times)
    meters = bar_meters(positions)
    beat_frames = (beat_times - start) * FPS
    draw, crossings = oracle_draw(beat_frames, positions, meters, FRAMES)
    return draw, crossings, beat_frames


def _replay(model, draw):
    return model.draws_to_paths(draw, torch.ones(1, FRAMES))


def _check_replay(model, draw, crossings):
    path = _replay(model, draw)
    beats = np.nonzero(path["is_beat"][0].numpy())[0]
    downbeats = np.nonzero(path["is_downbeat"][0].numpy())[0]
    assert list(beats) == [c for c, _, _ in crossings]
    assert list(downbeats) == [c for c, _, d in crossings if d]
    crossing = path["crossing"][0].numpy()
    for c, true_frame, _ in crossings:
        assert abs(crossing[c] - true_frame) < 0.05
    return path


def test_oracle_replays_a_metronome_in_four(model):
    draw, crossings, _ = _window([4] * 40, 120.0)
    assert len(crossings) > 50
    _check_replay(model, draw, crossings)


def test_oracle_replays_rubato_across_a_meter_change(model):
    draw, crossings, _ = _window([3] * 12 + [4] * 20, 90.0, rubato=0.04, seed=1)
    _check_replay(model, draw, crossings)
    spacing = np.diff([c for c, _, d in crossings if d])
    assert len(set(np.round(spacing, -1))) > 1


def test_oracle_crossings_are_the_annotated_beats(model):
    draw, crossings, beat_frames = _window([4] * 40, 75.0, rubato=0.03, seed=2)
    inside = beat_frames[(beat_frames > 0) & (beat_frames < FRAMES - 1)]
    assert len(crossings) == len(inside)
    assert np.allclose([t for _, t, _ in crossings], inside)


def test_doubling_the_tempo_doubles_the_beats(model):
    draw, crossings, _ = _window([4] * 60, 60.0)
    doubled = {k: v.clone() for k, v in draw.items()}
    doubled["log_tempo"][:, 0] += math.log(2.0)
    doubled["log_tempo"][:, 1:] = 0.0
    single = {k: v.clone() for k, v in draw.items()}
    single["log_tempo"][:, 1:] = 0.0
    n_single = int(_replay(model, single)["is_beat"].sum())
    n_double = int(_replay(model, doubled)["is_beat"].sum())
    assert abs(n_double - 2 * n_single) <= 2


def _batch(draw, n):
    return {k: v.expand(n, *v.shape[1:]).clone() for k, v in draw.items()}


class _AimAtPath:
    def __init__(self, kappa):
        self.kappa = kappa

    def step(self, feats_k, pred, tempo_drift, meter):
        return {"phase": (feats_k[:, 0], torch.full_like(pred, self.kappa))}


def _steady_prior(kappa):
    return {"phase": torch.tensor(kappa), "log_meter_transition": torch.tensor([[0.0, -30.0],
                                                                              [-30.0, 0.0]])}


def test_uncorrected_phase_noise_drifts_as_a_random_walk(model):
    draw, _, _ = _window([4] * 40, 120.0)
    kappa = 383.0
    base = _replay(model, draw)["phi_path"]
    phi = model.draws_to_paths(_batch(draw, 256), torch.ones(256, FRAMES),
                               prior=_steady_prior(kappa))["phi_path"]
    frames = 200
    drift = (phi[:, frames] - base[0, frames]).std().item()
    expected = math.sqrt(frames / kappa)
    assert 0.8 * expected < drift < 1.2 * expected


def test_phase_step_aimed_at_the_path_keeps_drift_at_one_frame(model):
    draw, _, _ = _window([4] * 40, 120.0)
    kappa = 383.0
    base = _replay(model, draw)["phi_path"][0]
    feats = base[None, :, None].expand(256, FRAMES, 1)
    phi = model.draws_to_paths(_batch(draw, 256), torch.ones(256, FRAMES),
                               posterior=_AimAtPath(kappa), feats=feats)["phi_path"]
    drift = (phi[:, [200, 1000, FRAMES - 1]] - base[[200, 1000, FRAMES - 1]]).std(0)
    assert (drift < 1.2 / math.sqrt(kappa)).all(), drift


def test_only_crossing_frames_carry_tempo_gradient(model):
    draw, crossings, _ = _window([4] * 40, 120.0, rubato=0.02, seed=3)
    log_tempo = draw["log_tempo"].clone().requires_grad_(True)
    path = _replay(model, {**draw, "log_tempo": log_tempo})
    path["phi_path"].sum().backward()
    grad = log_tempo.grad[0]
    assert grad[0] != 0 and torch.isfinite(grad).all()
    live = {0} | {c for c, _, _ in crossings}
    idle = torch.tensor([t not in live for t in range(FRAMES)])
    assert (grad[idle] == 0).all()
    assert (grad[[c for c, _, _ in crossings[:-1]]] != 0).all()
    assert not path["is_beat"].requires_grad


def test_meter_change_keeps_the_beat_tempo(model):
    draw, crossings, _ = _window([3] * 12 + [4] * 20, 100.0)
    _check_replay(model, draw, crossings)
    changes = draw["log_tempo"][0, 1:].abs()
    assert changes.max() < 0.01
    assert (changes > 1e-4).sum() <= 2
