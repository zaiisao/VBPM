import math

import numpy as np
import pytest
import torch

from vbpm.model import VBPM
from vbpm.nets import EmissionModel
from vbpm.specs import EmissionSpec
from vbpm.util.oracle import (bar_meters, beat_positions, labels_from_beats, oracle_draw,
                               synthetic_song)

FPS = 50.0
FRAMES = 500
MARGIN = 0.05


def _song(seed):
    rng = np.random.default_rng(seed)
    meter = int(rng.choice([3, 4]))
    bpm = float(rng.uniform(70, 160))
    beat_times, downbeat_times = synthetic_song([meter] * 40, bpm, FPS,
                                                rubato=float(rng.uniform(0, 0.03)), seed=seed)
    start = float(rng.uniform(2.5, 4.0))
    positions = beat_positions(beat_times, downbeat_times)
    draw, _ = oracle_draw((beat_times - start) * FPS, positions, bar_meters(positions), FRAMES)
    cls = labels_from_beats(beat_times - start, downbeat_times - start, 0, FRAMES, FPS)
    return draw, cls, meter


def _replay(paths, draw):
    path = paths.draws_to_paths(draw, torch.ones(1, FRAMES))
    return path["phi_path"], path["meter_path"]


def _perturbations(draw, meter):
    shifted = {k: v.clone() for k, v in draw.items()}
    shifted["phase0"] += math.pi / meter
    doubled = {k: v.clone() for k, v in draw.items()}
    doubled["log_tempo"][:, 0] += math.log(2.0)
    halved = {k: v.clone() for k, v in draw.items()}
    halved["log_tempo"][:, 0] -= math.log(2.0)
    remetered = {k: v.clone() for k, v in draw.items()}
    remetered["meter"] = remetered["meter"].flip(-1)
    return {"half-beat shift": shifted, "tempo x2": doubled, "tempo x1/2": halved,
            "wrong meter": remetered}


def _stack(items):
    return [torch.cat(parts) for parts in zip(*items)]


def _train(emission, inputs, cls, steps=400):
    opt = torch.optim.Adam(emission.parameters(), lr=1e-3)
    mask = torch.ones(cls.shape)
    for _ in range(steps):
        loss = -emission.loglik(*inputs, cls, mask).mean() / FRAMES
        opt.zero_grad()
        loss.backward()
        opt.step()
    return emission


def _recon(emission, inputs, cls):
    with torch.no_grad():
        return float(emission.loglik(*inputs, cls, torch.ones(cls.shape)).mean()) / FRAMES


@pytest.fixture(scope="module")
def corpus():
    torch.manual_seed(0)
    paths = VBPM(input_dim=8)
    songs = [_song(seed) for seed in range(24)]
    train, held = songs[:18], songs[18:]
    train_inputs = _stack([_replay(paths, d) for d, _, _ in train])
    train_cls = torch.cat([c for _, c, _ in train])
    held_inputs = _stack([_replay(paths, d) for d, _, _ in held])
    held_cls = torch.cat([c for _, c, _ in held])
    perturbed = {}
    for name in _perturbations(*held[0][::2]):
        perturbed[name] = _stack([_replay(paths, _perturbations(d, m)[name]) for d, _, m in held])
    teacher = _train(EmissionModel(EmissionSpec(), (3, 4)), train_inputs, train_cls)
    shuffled = [torch.roll(x, 1, 0) for x in train_inputs]
    control = _train(EmissionModel(EmissionSpec(), (3, 4)), shuffled, train_cls)
    return {"teacher": teacher, "control": control, "held_inputs": held_inputs,
            "held_cls": held_cls, "perturbed": perturbed}


def test_teacher_forced_emission_reads_the_true_path(corpus):
    truth = _recon(corpus["teacher"], corpus["held_inputs"], corpus["held_cls"])
    control = _recon(corpus["control"], corpus["held_inputs"], corpus["held_cls"])
    assert truth > control + MARGIN, (truth, control)


@pytest.mark.parametrize("name", ["half-beat shift", "tempo x2", "tempo x1/2", "wrong meter"])
def test_teacher_forced_emission_prefers_truth_over(corpus, name):
    truth = _recon(corpus["teacher"], corpus["held_inputs"], corpus["held_cls"])
    wrong = _recon(corpus["teacher"], corpus["perturbed"][name], corpus["held_cls"])
    assert truth > wrong + MARGIN, (name, truth, wrong)
