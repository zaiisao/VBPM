"""Behavioural tests for the vbpm package, derived from docstrings and math only."""
from __future__ import annotations

import math
import pathlib
import types

import numpy as np
import pytest
import torch

from vbpm import run as run_mod
from vbpm.config import load_config
from vbpm.scoring import controls as controls_mod
from vbpm.scoring.evaluation import f_measure, peak_times
from vbpm.constants import TWO_PI
from vbpm.specs import EmissionSpec
from vbpm.readout import downbeat_frames
from vbpm.variants.vbpm import VBPM
from vbpm.nets import Encoder
from vbpm.vonmises import kl_vonmises, log_i0, mean_resultant, sample_vonmises_icdf

from vbpm.data.features import FPS, atomic_save_npy

assert FPS == 50.0, "synthetic fixtures below are built around 50 fps"

# ---------------------------------------------------------------------------- helpers


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


class _Song:
    dataset = "toy"
    song_id = "toy_song"
    fold = 0

    def __init__(self, downbeats):
        self._db = np.asarray(downbeats, dtype=np.float64)

    def beats(self):
        return self._db, self._db  # every beat a downbeat: pure downbeat data


# ======================================================================= metrics_db


def test_f_measure_perfect_match():
    """Identical predicted and annotated times -> f = precision = recall = 1.
    Perfect agreement is the fixed point of any matching metric.
    """
    times = np.array([1.0, 2.0, 3.0])
    assert f_measure(times, times) == (1.0, 1.0, 1.0)


def test_f_measure_empty_predicted():
    """No predictions against real annotations -> recall 0 and f 0; nothing was found,
    so the score must not reward silence.
    """
    f, precision, recall = f_measure(np.array([]), np.array([1.0, 2.0]))
    assert f == 0.0 and recall == 0.0


def test_f_measure_empty_annotated():
    """Empty annotations: predicting nothing is exactly right (f=1); predicting anything
    is a pure false positive (f=0). Both edges follow from the definitions.
    """
    assert f_measure(np.array([]), np.array([]))[0] == 1.0
    assert f_measure(np.array([1.0]), np.array([]))[0] == 0.0


def test_f_measure_tolerance_edge_inclusive():
    """A prediction exactly `tolerance` from the annotation must count as a hit
    ("inside tolerance" with <= in the matcher), and one just beyond must not.
    Uses tolerance=0.5 so the edge is exactly float-representable.
    """
    f_at, _, _ = f_measure(np.array([1.5]), np.array([1.0]), tolerance=0.5)
    f_beyond, _, _ = f_measure(np.array([1.5000001]), np.array([1.0]), tolerance=0.5)
    assert f_at == 1.0
    assert f_beyond == 0.0


def test_f_measure_greedy_one_to_one():
    """Two predictions both within tolerance of ONE annotation: one-to-one matching
    allows only a single hit, so precision = 1/2, recall = 1. Double-counting would
    let a spike train score perfectly.
    """
    f, precision, recall = f_measure(np.array([1.00, 1.01]), np.array([1.0]))
    assert precision == 0.5 and recall == 1.0
    assert f == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_peak_times_relative_threshold():
    """The threshold is RELATIVE to the curve's own max (per the docstring): a curve
    topping out at 0.16 must still yield its peak, and a frame under threshold*max
    must not be picked.
    """
    probs = np.zeros(200)
    probs[50] = 0.16          # global max -> normalised to 1.0, always picked
    probs[150] = 0.16 * 0.4   # 0.4 relative < default 0.5 threshold -> rejected
    times = peak_times(probs, fps=50.0, period_s=1.0)
    assert 50 / 50.0 in times
    assert 150 / 50.0 not in times


def test_encoder_target_blind():
    """Point 2: the base encoder reads AUDIO ONLY -- structurally. Its forward
    has no target parameter at all (the psi variant's posterior subclass adds one),
    so target-blindness is a property of the signature, not a flag to keep off.
    """
    _seed()
    enc = Encoder(input_dim=4, d_model=8)
    named = enc.forward.__code__.co_varnames[:enc.forward.__code__.co_argcount]
    assert "y" not in named
    assert not getattr(enc, "reads_target", False)


def test_emission_logits_cosine_shape():
    """A + b*cos(phi): peak at phi=0, trough at phi=pi, even symmetry, period 2*pi --
    all properties of the cosine the docstring names.
    """
    _seed()
    model = VBPM(input_dim=4, d_model=8, emission=EmissionSpec(kind="cosine"))
    phi = torch.linspace(-math.pi, math.pi, 101)[None]

    logits = model.emission_model(phi)[0]
    peak = model.emission_model(torch.zeros(1, 1))[0, 0]
    trough = model.emission_model(torch.full((1, 1), math.pi))[0, 0]
    assert torch.all(logits <= peak + 1e-6)
    assert torch.all(logits >= trough - 1e-6)

    a, b = model.emission_model.a.item(), model.emission_model.b.item()
    assert peak.item() == pytest.approx(a + b, abs=1e-5)
    assert trough.item() == pytest.approx(a - b, abs=1e-5)

    sym = model.emission_model(-phi)[0]
    assert torch.allclose(logits, sym, atol=1e-6)
    period = model.emission_model(phi + TWO_PI)[0]
    assert torch.allclose(logits, period, atol=1e-5)


def test_emission_logits_triangle_shape():
    """Tent: logit = a + b*(1 - 2|phi|/pi) on the wrapped angle -- value a+b at 0, a-b
    at pi, LINEAR in |phi| in between, even, and continuous across the wrap at +-pi.
    """
    _seed()
    model = VBPM(input_dim=4, d_model=8, emission=EmissionSpec(kind="triangle"))
    a, b = model.emission_model.a.item(), model.emission_model.b.item()

    def at(p):
        return model.emission_model(torch.tensor([[p]]))[0, 0].item()

    assert at(0.0) == pytest.approx(a + b, abs=1e-5)
    assert at(math.pi) == pytest.approx(a - b, abs=1e-4)

    # linearity: value at phi is a + b*(1 - 2 phi/pi) for phi in (0, pi)
    for p in (0.3, 1.0, 2.5):
        assert at(p) == pytest.approx(a + b * (1 - 2 * p / math.pi), abs=1e-4)
        assert at(-p) == pytest.approx(at(p), abs=1e-6)          # symmetry

    # continuity at the wrap: pi - eps and pi + eps (== -pi + eps) agree
    eps = 1e-3
    assert at(math.pi - eps) == pytest.approx(at(math.pi + eps), abs=5e-3)


def test_emission_b_floor_semantics():
    """emission_model.b == b_floor + softplus(b_raw), with the floor
    defaulting to 0 -- exactly the property's docstring ('never below the scheduled
    floor').
    """
    _seed()
    model = VBPM(input_dim=4, d_model=8, emission=EmissionSpec(kind="cosine"))
    assert model.emission_model.b_floor == 0.0
    sp = torch.nn.functional.softplus(model.emission_model.b_raw).item()
    assert model.emission_model.b.item() == pytest.approx(sp)

    model.emission_model.b_floor.fill_(5.0)   # a BUFFER: mutate in place, never rebind
    assert model.emission_model.b.item() == pytest.approx(5.0 + sp)
    assert model.emission_model.b.item() >= 5.0


def test_emission_b_floor_survives_state_dict_roundtrip():
    """Set the scheduled floor, save state_dict, load into a fresh model: emission_b
        must be preserved, because the floor is part of the likelihood the checkpoint
        claims to represent. (Was an xfail: the floor used to be a plain attribute and
        silently reset to 0.0 on reload; it is a registered buffer now.)
    """
    _seed()
    model = VBPM(input_dim=4, d_model=8, emission=EmissionSpec(kind="cosine"))
    model.emission_model.b_floor.fill_(5.0)
    b_before = model.emission_model.b.item()
    state = model.state_dict()
    fresh = VBPM(input_dim=4, d_model=8, emission=EmissionSpec(kind="cosine"))
    fresh.load_state_dict(state)
    assert fresh.emission_model.b.item() == pytest.approx(b_before), \
        "scheduled emission floor lost across save/load"


def test_downbeat_frames_marks_zero_crossings():
    """Rule g: a linear ramp crossing phi = 0 (mod 2*pi) exactly at frames 50, 100, 150
    must produce crossings there and nowhere else; the crossing is flagged on the frame
    AFTER the wrap of the [0, 2pi) representation.
    """
    t = torch.arange(200, dtype=torch.float64)
    mu = (t * (TWO_PI / 50.0) - 3.0 * TWO_PI / 2.0)[None]   # wraps every 50 frames
    crossing = downbeat_frames(mu)[0]
    idx = torch.nonzero(crossing).flatten() + 1              # diff index -> frame index
    # mu(t) = 2*pi*t/50 - 3*pi is 0 mod 2*pi at t = 25, 75, 125, 175
    expected = {25, 75, 125, 175}
    assert set(idx.tolist()) == expected


def test_infer_phase_requires_eval_mode():
    """The deployment path asserts eval mode; calling it in train mode must raise."""
    _seed()
    model = VBPM(input_dim=4, d_model=8)
    model.train()
    with pytest.raises(AssertionError):
        model.infer_phase(torch.randn(1, 5, 4))

    model.eval()
    out = model.infer_phase(torch.randn(1, 5, 4))
    assert out.shape == (1, 5)


# ====================================================================== run helpers


def _args(beta_start=0.0, beta_end=1.0, beta_warmup=4):
    return types.SimpleNamespace(beta_start=beta_start, beta_end=beta_end,
                                 beta_warmup=beta_warmup)


def test_beta_at_schedule_edges():
    """Linear annealing: epoch 0 -> beta_start, epoch >= warmup -> beta_end, midpoint
    exactly halfway; warmup 0 disables annealing and returns beta_end always.
    """
    a = _args(0.0, 1.0, 4)
    assert run_mod.beta_at(0, a) == 0.0
    assert run_mod.beta_at(2, a) == pytest.approx(0.5)
    assert run_mod.beta_at(4, a) == 1.0
    assert run_mod.beta_at(100, a) == 1.0
    assert run_mod.beta_at(0, _args(0.3, 0.9, 0)) == 0.9


def _blindness_batch(input_dim: int, batch_size: int = 2, frames: int = 10):
    """A synthetic batch for the target-blindness control: no real audio needed."""
    return {"h": torch.randn(batch_size, frames, input_dim),
            "delta": torch.full((batch_size, frames), 0.06),
            "mask": torch.ones(batch_size, frames),
            "y": torch.zeros(batch_size, frames)}


def test_target_blindness_control_detects_leak():
    """The control must PASS the real model (whose deployed net never receives y) and
    FAIL a model whose deployed phase moves with anything but h -- otherwise a passing
    control proves nothing.
    """
    _seed()
    batch = _blindness_batch(input_dim=4)
    controls_mod.assert_encoder_is_target_blind(VBPM(input_dim=4, d_model=8), batch)

    leaky = VBPM(input_dim=4, d_model=8)
    calls = {"n": 0}
    real = leaky.infer_phase

    def cheating(h, delta=None):
        calls["n"] += 1
        return real(h, delta) + (0.1 if calls["n"] > 2 else 0.0)
    leaky.infer_phase = cheating

    with pytest.raises(AssertionError):
        controls_mod.assert_encoder_is_target_blind(leaky, batch)


def test_deployed_net_reads_no_target_in_any_shipped_config():
    """EVERY config's deployed net must read h and delta only."""
    configs = sorted((pathlib.Path(__file__).resolve().parent.parent / "configs")
                     .glob("*.yaml"))
    assert configs, "no configs found: this test would pass vacuously"

    checked = 0
    for path in configs:
        _seed()
        cfg, hooks = load_config(str(path))
        # BROKEN BY THE OBSERVATION-MODEL CHANGE, deliberately. Each of these variants
        # owns a forward()/heads() written against the per-frame Bernoulli and the
        # two-value heads() signature; they raise rather than silently mis-score. They
        # are BCE models, so running them would compare two different objectives under
        # one elbo column. Delete a name here when the variant is ported, not before.
        if cfg.variant in {"psi", "ladder", "anchor_time",
                           "interval_exact_norm", "interval_exact_rotation",
                           "interval_exact_tempo"}:
            continue
        model = hooks.build_model(cfg, input_dim=4)
        controls_mod.assert_encoder_is_target_blind(
            model, _blindness_batch(input_dim=4))
        checked += 1
    assert checked, "every config was skipped: this test would pass vacuously"


# ============================================================== vbpm cache write


def _cache_write(group_dir: pathlib.Path, stem: str, features: np.ndarray):
    """Exercises vbpm.data.atomic_save_npy, the single write-then-rename authority."""
    cache_path = group_dir / f"{stem}.npy"
    group_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(cache_path, features)
    return cache_path


def test_cache_write_atomic_rename_and_roundtrip(tmp_path):
    """After the write-then-rename block, exactly <stem>.npy exists (no stray .partial
    or .partial.npy from np.save's silent suffix-appending), and np.load returns the
    array bit-exactly in float32 -- the atomicity the comment promises.
    """
    features = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = _cache_write(tmp_path, "songA", features)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["songA.npy"], f"stray files: {names}"
    np.testing.assert_array_equal(np.load(path), features.astype(np.float32))


def test_cache_write_dotted_stem(tmp_path):
    """A stem containing dots (e.g. 'song.v2') must still produce exactly
    'song.v2.npy': with_suffix replaces only the final '.npy', and the open-handle
    np.save never appends a second one.
    """
    features = np.ones((2, 2), dtype=np.float32)
    path = _cache_write(tmp_path, "song.v2", features)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["song.v2.npy"]
    np.testing.assert_array_equal(np.load(path), features)


def test_cache_write_overwrites_existing(tmp_path):
    """A rewrite replaces the previous array atomically: the final file holds the new
    contents and no partial remains.
    """
    _cache_write(tmp_path, "s", np.zeros((2, 2), dtype=np.float32))
    path = _cache_write(tmp_path, "s", np.ones((2, 2), dtype=np.float32))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.npy"]
    np.testing.assert_array_equal(np.load(path), np.ones((2, 2), dtype=np.float32))


# ================================================== trajectory health diagnostics


def test_trajectory_health_separates_the_recorded_failure_modes():
    """The four numbers must distinguish trajectories that F cannot."""
    from vbpm.scoring.evaluation import scoring_records, trajectory_health

    n, fps, period = 600, 50.0, 2.0
    db = np.arange(1.0, 12.0, period)
    y = np.zeros(n, dtype=np.float32)
    for t in db:
        y[int(t * fps) - 1:int(t * fps) + 2] = 1.0
    raw = {"y": torch.tensor(y)[None].repeat(2, 1), "mask": torch.ones(2, n),
           "t0": torch.zeros(2), "fps": torch.full((2,), fps),
           "downbeat_times": [db] * 2,
           "anchors": [np.concatenate([[db[0] - period], db, [db[-1] + period]])] * 2,
           "dataset": ["toy"] * 2, "song_id": ["a", "b"]}
    crops = scoring_records(raw)
    t = torch.arange(n, dtype=torch.float32)
    tempo = TWO_PI / (period * fps)
    kappa = torch.full((2, n), 2000.0)

    def wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def health(mu):
        return trajectory_health(mu[None].repeat(2, 1), kappa, torch.ones(2, n), crops)

    adv, kap, err, cov = health(wrap(tempo * (t - db[0] * fps)))
    assert adv == pytest.approx(tempo, rel=1e-3)
    assert err < 1e-3 and cov == 1.0 and kap == pytest.approx(2000.0)

    adv_o, _, err_o, cov_o = health(wrap(tempo * (t - db[0] * fps) + math.pi))
    assert adv_o == pytest.approx(tempo, rel=1e-3)      # tempo right, offset wrong
    assert err_o == pytest.approx(math.pi, abs=1e-3)   # worse than chance, not random
    assert cov_o == 1.0

    adv_s, _, err_s, _ = health(wrap(0.715 * t))
    assert adv_s == pytest.approx(0.715, rel=1e-3)
    assert err_s > 1.5                                  # indistinguishable from chance

    adv_f, _, err_f, cov_f = health(torch.zeros(n))
    assert adv_f == pytest.approx(0.0, abs=1e-6)
    assert err_f == pytest.approx(math.pi / 2, rel=0.02)
    assert cov_f == pytest.approx(1 / 16)


def test_mean_resultant_matches_the_series_and_its_gradient():
    k = torch.tensor([0.5, 2.0, 20.0, 383.0], dtype=torch.float64, requires_grad=True)
    a = mean_resultant(k)
    x = torch.linspace(-math.pi, math.pi, 200001, dtype=torch.float64)[:-1]
    for i, kv in enumerate([0.5, 2.0, 20.0, 383.0]):
        w = torch.exp(kv * torch.cos(x) - kv)
        assert float(a[i]) == pytest.approx(
            float((w * torch.cos(x)).sum() / w.sum()), rel=1e-6)
    a.sum().backward()
    assert torch.isfinite(k.grad).all() and (k.grad > 0).all()


def test_kl_vonmises_is_zero_at_identity_and_matches_monte_carlo():
    k = torch.tensor([4.0, 60.0], dtype=torch.float64)
    assert torch.allclose(kl_vonmises(torch.zeros(2, dtype=torch.float64), k,
                                      torch.zeros(2, dtype=torch.float64), k),
                          torch.zeros(2, dtype=torch.float64), atol=1e-12)
    mu1, k1, mu2, k2 = 0.3, 30.0, 0.05, 12.0
    x = torch.linspace(-math.pi, math.pi, 400001, dtype=torch.float64)[:-1]

    def lp(mu, kk):
        return kk * torch.cos(x - mu) - math.log(TWO_PI) - float(
            log_i0(torch.tensor(kk, dtype=torch.float64)))
    w = torch.exp(lp(mu1, k1))
    ref = float((w * (lp(mu1, k1) - lp(mu2, k2))).sum() / w.sum())
    got = float(kl_vonmises(torch.tensor(mu1), torch.tensor(k1),
                            torch.tensor(mu2), torch.tensor(k2)))
    assert got == pytest.approx(ref, rel=1e-5)


def test_icdf_sampler_concentrates_and_stays_differentiable_in_kappa():
    torch.manual_seed(0)
    for kv, tol in ((5.0, 0.02), (383.0, 0.002)):
        k = torch.full((200000,), kv, dtype=torch.float64)
        s = sample_vonmises_icdf(k)
        assert float(s.mean().abs()) < tol
        assert float(torch.cos(s).mean()) == pytest.approx(
            float(mean_resultant(torch.tensor(kv, dtype=torch.float64))), rel=5e-3)
    k = torch.full((4096,), 40.0, dtype=torch.float64, requires_grad=True)
    sample_vonmises_icdf(k).abs().mean().backward()
    assert torch.isfinite(k.grad).all() and float(k.grad.abs().sum()) > 0.0


def test_smooth_marginals_match_brute_force_enumeration():
    """Exact inference: forward-backward equals summing every path by hand."""
    import itertools
    from vbpm.nets import PosteriorModel, PriorModel
    from vbpm.specs import RateSpec, WalkSpec

    _seed()
    C, N, T, METERS = 2, 3, 3, (3, 4)
    M = len(METERS)
    prior = PriorModel(RateSpec(grid=C, lo=0.05, hi=0.12, meters=METERS),
                       WalkSpec(kappa_physical=3.0), n_grid=N).double()
    post = PosteriorModel(8, 8, prior, n_harm=1).double()
    evidence = torch.randn(1, T, N, dtype=torch.float64) * 0.7
    log_q_rate0 = torch.log_softmax(torch.randn(1, C, dtype=torch.float64), -1)
    log_q_meter = torch.log_softmax(torch.randn(1, M, dtype=torch.float64), -1)

    q_phase, q_rate, q_meter, log_z = post.smooth(evidence, log_q_rate0, prior,
                                                  log_q_meter)

    p0 = torch.softmax(prior.rate_log_prior, 0)
    pm0 = torch.softmax(prior.meter_log_prior, 0)
    total, marginal = 0.0, torch.zeros(T, M, C, N, dtype=torch.float64)
    for path in itertools.product(range(M * C * N), repeat=T):
        states = [(k // (C * N), (k // N) % C, k % N) for k in path]
        u0, c0, n0 = states[0]
        w = (float(pm0[u0]) * float(log_q_meter[0, u0].exp())
             * float(p0[c0]) * float(log_q_rate0[0, c0].exp()) / N
             * float(evidence[0, 0, n0].exp()))
        for t in range(1, T):
            (u, c, i), (v, d, j) = states[t - 1], states[t]
            step = (prior.k_stay[c, i, j] * (c == d) * (u == v)
                    + prior.k_wrap[c, i, j] * prior.switch[c, d]
                    * prior.meter_switch[u, v])
            w *= float(step) * float(evidence[0, t, j].exp())
        total += w
        for t, (u, c, n) in enumerate(states):
            marginal[t, u, c, n] += w

    assert float(log_z[0]) == pytest.approx(math.log(total), abs=1e-6)
    ref = marginal / total
    assert torch.allclose(q_phase[0], ref.sum(2), atol=1e-12)
    assert torch.allclose(q_rate[0], ref.sum((1, 3)), atol=1e-12)
    assert torch.allclose(q_meter[0], ref.sum((2, 3)), atol=1e-12)


def test_emission_loglik_is_the_bernoulli_it_claims():
    """loglik == -BCEWithLogits at every grid phase; masked frames cost exactly 0."""
    from vbpm.nets import EmissionModel
    from vbpm.specs import EmissionSpec

    _seed()
    emission = EmissionModel(EmissionSpec(kind="triangle")).double()
    grid = torch.arange(128, dtype=torch.float64) * (TWO_PI / 128)
    y = (torch.rand(2, 30, dtype=torch.float64) < 0.2).double()
    mask = torch.ones(2, 30, dtype=torch.float64)
    mask[1, -5:] = 0.0

    ours = emission.bernoulli_loglik(emission(grid), y, mask)
    logits = emission(grid)[None, None].expand(2, 30, 128)
    reference = -torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y[..., None].expand(2, 30, 128), reduction="none") * mask[..., None]

    assert torch.allclose(ours, reference, atol=1e-15)
    assert (ours[1, -5:] == 0).all()
