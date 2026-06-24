"""Unit tests for models/svt_core.py:
   sample_from_prior, sample_from_prior_pf, _systematic_resample.

Ground truth / invariants used (never tautologies):
  * Closed-form recursion: the deterministic phase-mean chain is reconstructed
    independently from the model's own per-frame correction/tempo tensors and
    compared to the returned phase_mu, exercising the exact wrap arithmetic.
  * Structural invariants: shapes, finiteness, range [0, 2pi), one-hot simplex,
    argmax consistency, sawtooth monotonicity between wraps.
  * Resampling invariants: indices in [0, N), length N, and concentration on a
    near-degenerate weight vector (a property of systematic resampling, derived
    from the inverse-CDF construction, NOT copied from a debug run).
"""

import sys
sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import numpy as np
import pytest
import torch

from models.svt_core import SVTModel, LOG_TEMPO_MIN, LOG_TEMPO_MAX, TWO_PI

torch.manual_seed(0)
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(K=4, D=16, audio_emission=False, **kw):
    torch.manual_seed(1234)
    m = SVTModel(
        hidden_dim=D, nhead=2, num_layers=1, num_meter_classes=K,
        input_dim=2, audio_emission=audio_emission, **kw,
    ).to(DEVICE)
    m.eval()
    return m


def _acts(B=1, T=40, input_dim=2):
    torch.manual_seed(7)
    return torch.randn(B, T, input_dim, device=DEVICE)


# ---------------------------------------------------------------------------
# sample_from_prior: keys / shapes / finiteness
# ---------------------------------------------------------------------------

def test_sample_from_prior_keys_shapes_finite():
    K, B, T = 5, 2, 37
    m = _make_model(K=K)
    act = _acts(B=B, T=T)
    out = m.sample_from_prior(act, temperature=0.3)

    expected = {"phase", "phase_mu", "log_tempo",
                "meter_soft", "meter_onehot", "beat_logits"}
    assert expected.issubset(set(out.keys())), out.keys()

    assert out["phase"].shape == (B, T)
    assert out["phase_mu"].shape == (B, T)
    assert out["log_tempo"].shape == (B, T)
    assert out["meter_soft"].shape == (B, T, K)
    assert out["meter_onehot"].shape == (B, T, K)
    assert out["beat_logits"].shape == (B, T, 2)

    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} has non-finite entries"


def test_sample_from_prior_phase_ranges():
    """phase and phase_mu are wrapped to [0, 2pi); log_tempo respects clamps."""
    m = _make_model(K=4)
    act = _acts(B=3, T=50)
    out = m.sample_from_prior(act, temperature=0.2)

    for key in ("phase", "phase_mu"):
        v = out[key]
        assert (v >= 0.0).all(), f"{key} below 0"
        assert (v < TWO_PI + 1e-5).all(), f"{key} >= 2pi"

    # Free-running tempo is clamped to the musical band each step.
    lt = out["log_tempo"]
    assert (lt >= LOG_TEMPO_MIN - 1e-5).all()
    assert (lt <= LOG_TEMPO_MAX + 1e-5).all()


def test_meter_onehot_is_valid_onehot_and_matches_soft_argmax():
    """meter_onehot is a true one-hot simplex and its argmax == soft argmax."""
    K = 6
    m = _make_model(K=K)
    out = m.sample_from_prior(_acts(B=2, T=30), temperature=0.5)
    oh = out["meter_onehot"]
    soft = out["meter_soft"]

    # exactly one 1 per frame, all entries in {0,1}
    assert torch.all((oh == 0) | (oh == 1)), "one-hot has non-binary entries"
    assert torch.allclose(oh.sum(-1), torch.ones_like(oh.sum(-1))), "rows not summing to 1"
    # one-hot location == argmax of the soft sample (per construction in source)
    assert torch.equal(oh.argmax(-1), soft.argmax(-1))

    # soft meter is a valid categorical (Gumbel-softmax simplex)
    assert torch.allclose(soft.sum(-1), torch.ones_like(soft.sum(-1)), atol=1e-4)
    assert (soft >= -1e-6).all()


# ---------------------------------------------------------------------------
# sample_from_prior: deterministic phase_mu chain == closed-form recursion
# ---------------------------------------------------------------------------

def test_phase_mu_matches_closed_form_recursion():
    """The returned phase_mu must equal the exact deterministic recursion
       phase_mu_t = wrap(phase_mu_{t-1} + exp(log_tempo_mu_{t-1}) + corr_t),
       reconstructed independently from the model's own correction/tempo heads.

       This is the ground-truth check that the mean chain is the bar-pointer
       sawtooth (no sample jitter leaks in)."""
    from models.svt_core import _softplus_pos

    K, B, T = 4, 1, 60
    m = _make_model(K=K, tempo_anchor_mode="none", tempo_reversion_alpha=0.0)
    act = _acts(B=B, T=T)
    out = m.sample_from_prior(act, temperature=0.1)
    got = out["phase_mu"][0]  # [T]

    # --- Recompute the deterministic chain independently from the heads. ---
    with torch.no_grad():
        h_prior = m.encode_prior(act)
        phase_corr_all, tempo_corr_all = m.prior_mean_corrections(h_prior)  # [B,T]
        h_global = h_prior.mean(dim=1)
        init_prior = m.init_prior_head(h_global)

        phase_mu = torch.remainder(init_prior["phase_mu"][0], TWO_PI)
        log_tempo_mu = init_prior["tempo_mu"][0].clamp(LOG_TEMPO_MIN, LOG_TEMPO_MAX)
        ref = [phase_mu.clone()]
        for t in range(1, T):
            tempo_lin = torch.exp(log_tempo_mu.clamp(max=10.0))
            phase_mu = torch.remainder(
                phase_mu + tempo_lin + phase_corr_all[0, t], TWO_PI,
            )
            # anchor_mode='none' & alpha=0 => pure walk: mu += corr, then clamp.
            log_tempo_mu = (log_tempo_mu + tempo_corr_all[0, t]).clamp(
                LOG_TEMPO_MIN, LOG_TEMPO_MAX,
            )
            ref.append(phase_mu.clone())
        ref = torch.stack(ref)

    assert torch.allclose(got, ref, atol=1e-5), \
        f"max abs diff {(got - ref).abs().max().item():.3e}"


def test_phase_mu_clean_sawtooth_monotone_between_wraps():
    """Freshly-initialised model: corr heads ~0 (final layer scaled 0.01, bias 0),
       so per-step increment ~= exp(log_tempo) > 0. The phase_mu chain must
       therefore advance (strictly increase modulo wraps): every step is either a
       positive advance or a downward wrap of magnitude < 2pi. A wrap (negative
       diff) must coincide with crossing 2pi, i.e. diff + 2pi must be a small
       positive advance < pi."""
    K, B, T = 4, 1, 200
    m = _make_model(K=K, tempo_anchor_mode="none", tempo_reversion_alpha=0.0)
    out = m.sample_from_prior(_acts(B=B, T=T), temperature=0.1)
    pm = out["phase_mu"][0]
    diffs = pm[1:] - pm[:-1]

    # un-wrap: an advance is diff (if >=0) or diff + 2pi (a wrap-around)
    adv = torch.where(diffs >= 0, diffs, diffs + TWO_PI)
    # All advances must be strictly positive (chain never stalls / goes backward).
    assert (adv > 0).all(), f"non-positive advance: min {adv.min().item():.3e}"
    # And bounded by one bar-pointer step (tempo ~< 1 rad/frame + corr ~0 << pi).
    assert (adv < math.pi).all(), f"advance too large: max {adv.max().item():.3e}"

    # The chain must actually wrap at least once over 200 frames (it's a sawtooth,
    # not a monotone line) -> there exists a negative raw diff.
    assert (diffs < 0).any(), "phase_mu never wrapped over 200 frames (not a sawtooth)"


def test_sample_from_prior_deterministic_chain_seed_independent():
    """phase_mu is the DETERMINISTIC mean chain: it must NOT depend on the RNG
       seed (only the stochastic 'phase' draw does). Distinguishes the two."""
    m = _make_model(K=4)
    act = _acts(B=1, T=40)
    torch.manual_seed(11)
    o1 = m.sample_from_prior(act, temperature=0.1)
    torch.manual_seed(999)
    o2 = m.sample_from_prior(act, temperature=0.1)
    # deterministic mean chain identical across seeds
    assert torch.allclose(o1["phase_mu"], o2["phase_mu"], atol=1e-6)
    # stochastic sample differs across seeds (sanity: RNG actually changed things)
    assert not torch.allclose(o1["phase"], o2["phase"], atol=1e-3)


# ---------------------------------------------------------------------------
# _systematic_resample
# ---------------------------------------------------------------------------

def test_systematic_resample_index_bounds_and_length():
    m = _make_model()
    for N in (1, 2, 8, 100):
        torch.manual_seed(N)
        w = torch.rand(N)
        w = w / w.sum()
        idx = m._systematic_resample(w)
        assert idx.shape == (N,), f"length mismatch for N={N}"
        assert idx.dtype in (torch.int64, torch.int32, torch.long)
        assert (idx >= 0).all() and (idx < N).all(), f"out-of-range idx for N={N}"


def test_systematic_resample_concentrates_on_heavy_particle():
    """Near-degenerate weights (all mass on one particle) => every drawn ancestor
       is that particle. This is an exact property of systematic resampling: with
       cumulative CDF flat everywhere except one unit jump, all N evenly spaced
       inverse-CDF queries land in that particle's interval."""
    N = 200
    heavy = 137
    w = torch.full((N,), 1e-9)
    w[heavy] = 1.0
    w = w / w.sum()
    m = _make_model()
    for trial in range(20):
        torch.manual_seed(trial)
        idx = m._systematic_resample(w)
        assert (idx == heavy).all(), \
            f"trial {trial}: {(idx != heavy).sum().item()} indices not on heavy particle"


def test_systematic_resample_two_heavy_split_roughly_proportional():
    """Mass split 0.75/0.25 between two particles => systematic resampling returns
       ~0.75 N / ~0.25 N copies (low-variance; exact count within +-1 due to the
       single uniform offset). Counts must be proportional, not arbitrary."""
    N = 400
    a, b = 10, 300
    w = torch.zeros(N)
    w[a] = 0.75
    w[b] = 0.25
    m = _make_model()
    for trial in range(10):
        torch.manual_seed(100 + trial)
        idx = m._systematic_resample(w)
        # only the two heavy particles may appear
        uniq = set(idx.unique().tolist())
        assert uniq.issubset({a, b}), f"unexpected ancestors {uniq}"
        na = int((idx == a).sum())
        # systematic resampling: count within +-1 of N * weight
        assert abs(na - 0.75 * N) <= 1, f"count {na} not ~{0.75*N}"


def test_systematic_resample_uniform_weights_is_near_permutation():
    """Uniform weights => systematic resampling visits each particle exactly once
       (the offsets fall one per CDF cell). Mean ancestor index ~ (N-1)/2."""
    N = 256
    w = torch.full((N,), 1.0 / N)
    m = _make_model()
    torch.manual_seed(3)
    idx = m._systematic_resample(w)
    counts = torch.bincount(idx, minlength=N)
    # each cell hit exactly once for exactly-uniform weights
    assert (counts == 1).all(), "uniform-weight resample is not a permutation"


# ---------------------------------------------------------------------------
# sample_from_prior_pf (Dir 1B)
# ---------------------------------------------------------------------------

def test_pf_requires_audio_emission():
    m = _make_model(audio_emission=False)
    with pytest.raises(AssertionError):
        m.sample_from_prior_pf(_acts(B=1, T=20), n_particles=16)


def test_pf_requires_B_eq_1():
    m = _make_model(audio_emission=True)
    with pytest.raises(AssertionError):
        m.sample_from_prior_pf(_acts(B=2, T=20), n_particles=16)


def test_pf_keys_shapes_finite():
    K, T = 5, 45
    m = _make_model(K=K, audio_emission=True)
    act = _acts(B=1, T=T)
    out = m.sample_from_prior_pf(
        act, n_particles=64, obs_sigma=0.3, temperature=0.1, ess_frac=0.5,
    )
    expected = {"phase", "phase_mu", "log_tempo", "meter_soft",
                "meter_onehot", "beat_logits", "beat_activation"}
    assert expected.issubset(set(out.keys())), out.keys()

    assert out["phase"].shape == (1, T)
    assert out["phase_mu"].shape == (1, T)
    assert out["log_tempo"].shape == (1, T)
    assert out["meter_soft"].shape == (1, T, K)
    assert out["meter_onehot"].shape == (1, T, K)
    assert out["beat_logits"].shape == (1, T, 2)
    assert out["beat_activation"].shape == (1, T)

    for k, v in out.items():
        assert torch.isfinite(v).all(), f"PF output {k} non-finite"


def test_pf_map_trajectory_in_range_and_clamped():
    """MAP phase wrapped to [0,2pi); MAP log_tempo within the musical clamp band;
       phase_mu == phase (PF MAP path is the read-out, per source)."""
    m = _make_model(K=4, audio_emission=True)
    out = m.sample_from_prior_pf(_acts(B=1, T=40), n_particles=64)
    ph = out["phase"]
    assert (ph >= 0).all() and (ph < TWO_PI + 1e-5).all()
    lt = out["log_tempo"]
    assert (lt >= LOG_TEMPO_MIN - 1e-5).all() and (lt <= LOG_TEMPO_MAX + 1e-5).all()
    # source sets phase_mu := phase for the PF read-out
    assert torch.equal(out["phase"], out["phase_mu"])


def test_pf_meter_onehot_matches_map_soft_argmax():
    K = 7
    m = _make_model(K=K, audio_emission=True)
    out = m.sample_from_prior_pf(_acts(B=1, T=35), n_particles=48)
    oh = out["meter_onehot"][0]
    soft = out["meter_soft"][0]
    assert torch.all((oh == 0) | (oh == 1))
    assert torch.allclose(oh.sum(-1), torch.ones_like(oh.sum(-1)))
    assert torch.equal(oh.argmax(-1), soft.argmax(-1))


def test_pf_beat_activation_is_weighted_probability():
    """beat_activation is a per-frame weighted fraction of particles wrapping in:
       it must lie in [0,1] (convex combo of {0,1} boundary indicators by a
       softmax weight that sums to 1). t=0 is never set => exactly 0."""
    m = _make_model(K=4, audio_emission=True)
    out = m.sample_from_prior_pf(_acts(B=1, T=50), n_particles=80)
    ba = out["beat_activation"][0]
    assert (ba >= -1e-6).all(), f"beat_activation < 0 (min {ba.min().item()})"
    assert (ba <= 1.0 + 1e-5).all(), f"beat_activation > 1 (max {ba.max().item()})"
    assert ba[0].abs().item() < 1e-7, "frame 0 beat_activation should be 0"


def test_pf_runs_with_anchor_modes():
    """PF should run across OU-anchor config variants without NaN/shape errors --
       catches config-dependent regressions in the free-running rollout."""
    for kw in (
        dict(tempo_anchor_mode="ema", tempo_reversion_alpha=0.1, audio_emission=True),
        dict(tempo_anchor_mode="latent", tempo_reversion_alpha=0.1, audio_emission=True),
        dict(tempo_anchor_mode="global", tempo_reversion_alpha=0.1, audio_emission=True),
    ):
        m = _make_model(K=4, **kw)
        out = m.sample_from_prior_pf(_acts(B=1, T=30), n_particles=32)
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{kw} -> {k} non-finite"


@pytest.mark.xfail(reason="BUG: sample_from_prior_pf is incompatible with "
                          "bar_phase=True -- the PF loop never builds a bar-phase "
                          "trajectory and its final _decode() omits bar_phase=, so "
                          "_decode asserts (bar_phase is not None). sample_from_prior "
                          "handles bar_phase correctly; the PF path does not.",
                   strict=True, raises=AssertionError)
def test_pf_bar_phase_is_broken():
    """Documents the bar_phase+PF incompatibility. sample_from_prior works with
       bar_phase=True (see test_sample_from_prior_bar_phase_keys); the PF variant
       crashes in _decode. Marked xfail(strict) so it flags if/when the source is
       fixed (the xfail will then XPASS and fail the suite, prompting removal)."""
    m = _make_model(K=4, bar_phase=True, audio_emission=True)
    m.sample_from_prior_pf(_acts(B=1, T=30), n_particles=32)


def test_sample_from_prior_bar_phase_keys():
    """bar_phase=True adds bar_phase / bar_phase_mu trajectories, both wrapped."""
    m = _make_model(K=4, bar_phase=True)
    out = m.sample_from_prior(_acts(B=1, T=40), temperature=0.1)
    assert "bar_phase" in out and "bar_phase_mu" in out
    for key in ("bar_phase", "bar_phase_mu"):
        v = out[key]
        assert v.shape == (1, 40)
        assert torch.isfinite(v).all()
        assert (v >= 0).all() and (v < TWO_PI + 1e-5).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
