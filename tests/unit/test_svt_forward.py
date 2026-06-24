"""Unit tests for models.svt_core.SVTModel.forward.

Ground truth used here:
  * Structural invariants (shapes, dict keys) read off the real source.
  * No-NaN / no-Inf finiteness invariants.
  * Probability-simplex invariant for the Gumbel-Softmax meter sample
    (rows sum to 1, all >= 0) — closed-form property of softmax.
  * Phase samples wrapped to [0, 2*pi) — the code applies torch.remainder.
  * Reparameterization => finite gradients must reach BOTH encoders and the
    decoder (finite-difference-free structural gradient-flow check; the
    von Mises implicit-reparam backward is the only path for d/d(phase_kappa)).
  * Seed determinism (same seed => identical output; different seed => differs)
    — a Monte-Carlo / RNG-contract invariant, NOT a hard-coded value.
  * Temperature monotonicity for Gumbel-Softmax: higher tau => softer (higher
    entropy) meter samples in expectation — an analytic property of the
    relaxation, estimated by Monte-Carlo averaging over the batch/time axis.

These assert against invariants and math, never "code == code".
"""

import sys

sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import pytest
import torch

from models.svt_core import SVTModel, TWO_PI

DEVICE = "cpu"
HID = 32
K = 8
B, T, IN = 2, 16, 2


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_model(seed: int = 0, **kw) -> SVTModel:
    torch.manual_seed(seed)
    m = SVTModel(
        hidden_dim=HID,
        nhead=4,
        num_layers=2,
        num_meter_classes=K,
        input_dim=IN,
        **kw,
    ).to(DEVICE)
    m.eval()  # deterministic dropout-free; sampling RNG still active
    return m


def _make_batch(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    activations = torch.rand(B, T, IN, generator=g)
    # plausible sparse beat / downbeat targets (0/1)
    beat = (torch.rand(B, T, generator=g) > 0.7).float()
    down = (beat.bool() & (torch.rand(B, T, generator=g) > 0.5)).float()
    return activations, beat, down


def _run(model, seed=1234, **fwd_kw):
    """Forward with a fixed global RNG seed for reproducible sampling."""
    activations, beat, down = _make_batch()
    torch.manual_seed(seed)
    return model(activations, beat_targets=beat, downbeat_targets=down, **fwd_kw)


# ---------------------------------------------------------------------------
# Shapes / keys
# ---------------------------------------------------------------------------

def test_output_top_level_keys():
    out = _run(_make_model())
    for key in ("beat_logits", "posterior", "prior", "samples"):
        assert key in out, f"missing top-level key {key!r}"


def test_beat_logits_shape():
    out = _run(_make_model())
    assert out["beat_logits"].shape == (B, T, 2)


def test_posterior_keys_and_shapes():
    post = _run(_make_model())["posterior"]
    assert post["meter_logits"].shape == (B, T, K)
    for k in ("phase_mu", "phase_log_kappa", "tempo_mu", "tempo_log_sigma"):
        assert post[k].shape == (B, T), f"posterior[{k!r}] shape {post[k].shape}"


def test_prior_keys_and_shapes():
    prior = _run(_make_model())["prior"]
    assert prior["meter_logits"].shape == (B, T, K)
    for k in ("phase_mu", "phase_kappa", "tempo_mu", "tempo_sigma"):
        assert prior[k].shape == (B, T), f"prior[{k!r}] shape {prior[k].shape}"


def test_samples_keys_and_shapes():
    samples = _run(_make_model())["samples"]
    assert samples["phase"].shape == (B, T)
    assert samples["log_tempo"].shape == (B, T)
    assert samples["meter_soft"].shape == (B, T, K)


# ---------------------------------------------------------------------------
# Finiteness (no NaN / Inf anywhere)
# ---------------------------------------------------------------------------

def _assert_all_finite(out):
    bad = []
    for name in ("beat_logits",):
        t = out[name]
        if not torch.isfinite(t).all():
            bad.append(name)
    for group in ("posterior", "prior", "samples"):
        for k, v in out[group].items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                bad.append(f"{group}.{k}")
    return bad


def test_all_outputs_finite():
    out = _run(_make_model())
    bad = _assert_all_finite(out)
    assert not bad, f"non-finite values in: {bad}"


def test_all_outputs_finite_extreme_temperature_low():
    out = _run(_make_model(), temperature=0.05)
    bad = _assert_all_finite(out)
    assert not bad, f"non-finite at low temperature in: {bad}"


def test_all_outputs_finite_extreme_temperature_high():
    out = _run(_make_model(), temperature=5.0)
    bad = _assert_all_finite(out)
    assert not bad, f"non-finite at high temperature in: {bad}"


def test_finite_with_no_targets():
    # forward must synthesize zero targets internally; still finite.
    activations, _, _ = _make_batch()
    model = _make_model()
    torch.manual_seed(7)
    out = model(activations)  # no beat/downbeat targets
    bad = _assert_all_finite(out)
    assert not bad, f"non-finite with default (None) targets: {bad}"


# ---------------------------------------------------------------------------
# Invariants: simplex, phase wrapping, positivity
# ---------------------------------------------------------------------------

def test_meter_soft_is_on_simplex():
    samples = _run(_make_model())["samples"]
    ms = samples["meter_soft"]
    assert (ms >= -1e-6).all(), "meter_soft has negative entries"
    sums = ms.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        f"meter_soft rows do not sum to 1 (max dev {(sums-1).abs().max().item():.2e})"


def test_phase_samples_wrapped():
    phase = _run(_make_model())["samples"]["phase"]
    assert (phase >= 0.0).all() and (phase < TWO_PI + 1e-5).all(), \
        f"phase out of [0,2pi): min {phase.min().item()} max {phase.max().item()}"


def test_prior_phase_mu_wrapped_for_transition_steps():
    # The PRIOR transition recursion (t>=1) wraps via torch.remainder(.,2pi),
    # so those means must lie in [0, 2pi). (The t=0 init slot uses the absolute
    # pi*tanh parameterization in (-pi, pi); the posterior uses pi*tanh at every
    # step by default, so neither is contractually wrapped — only the t>=1 prior.)
    out = _run(_make_model())
    mu = out["prior"]["phase_mu"][:, 1:]
    assert (mu >= -1e-5).all() and (mu < TWO_PI + 1e-4).all(), \
        f"prior.phase_mu (t>=1) out of [0,2pi): min {mu.min()} max {mu.max()}"


def test_phase_mu_means_are_valid_angles():
    # All phase_mu values are von Mises mean directions interpreted mod 2pi;
    # by construction (pi*tanh or remainder) they stay within (-pi, 2pi).
    out = _run(_make_model())
    for grp in ("prior", "posterior"):
        mu = out[grp]["phase_mu"]
        assert (mu > -math.pi - 1e-4).all() and (mu < TWO_PI + 1e-4).all(), \
            f"{grp}.phase_mu outside (-pi, 2pi): min {mu.min()} max {mu.max()}"


def test_prior_concentration_positive():
    prior = _run(_make_model())["prior"]
    assert (prior["phase_kappa"] > 0).all(), "prior phase_kappa must be > 0"
    assert (prior["tempo_sigma"] > 0).all(), "prior tempo_sigma must be > 0"


def test_prior_tempo_mu_within_musical_clamp():
    # forward clamps the carried prior log-tempo to [LOG_TEMPO_MIN, LOG_TEMPO_MAX]
    # for t>=1. The t=0 init slot is unclamped, so test t>=1.
    from models.svt_core import LOG_TEMPO_MIN, LOG_TEMPO_MAX
    prior = _run(_make_model())["prior"]
    tm = prior["tempo_mu"][:, 1:]
    assert (tm >= LOG_TEMPO_MIN - 1e-5).all() and (tm <= LOG_TEMPO_MAX + 1e-5).all(), \
        f"prior tempo_mu (t>=1) outside musical clamp: [{tm.min()}, {tm.max()}]"


# ---------------------------------------------------------------------------
# Gradients: reparameterization must reach encoders AND decoder
# ---------------------------------------------------------------------------

def _named_modules_have_grad(model, prefix):
    """Return (n_params_with_grad, n_params_total) for params whose name
    starts with one of the given prefixes."""
    n_grad = n_tot = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(pre) for pre in prefix):
            n_tot += 1
            if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
                n_grad += 1
    return n_grad, n_tot


def test_gradients_reach_encoders_and_decoder():
    model = _make_model()
    model.train()  # ensure all params active
    activations, beat, down = _make_batch()
    torch.manual_seed(99)
    out = model(activations, beat_targets=beat, downbeat_targets=down)
    loss = out["beat_logits"].pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss), "loss not finite"

    # Posterior encoder: gradient flows via the latent samples + reparam.
    n_post, tot_post = _named_modules_have_grad(model, ("post_proj", "post_encoder"))
    assert n_post > 0, f"NO gradient reached posterior encoder (0/{tot_post})"

    # Prior encoder: the decoder reads h_prior (decoder_use_h_prior=True) AND
    # prior-mean corrections feed the sampled phase, so grads must reach it.
    n_prior, tot_prior = _named_modules_have_grad(
        model, ("prior_input_proj", "prior_encoder")
    )
    assert n_prior > 0, f"NO gradient reached prior encoder (0/{tot_prior})"

    # Decoder.
    n_dec, tot_dec = _named_modules_have_grad(model, ("emission_decoder",))
    assert n_dec == tot_dec and tot_dec > 0, \
        f"decoder params missing grad: {n_dec}/{tot_dec}"


def test_gradient_flows_through_phase_kappa_head():
    """The von Mises implicit-reparam BACKWARD is the only path for a gradient
    to reach the posterior phase-kappa head from beat_logits (phase sample ->
    cos/sin -> decoder). If the backward were broken (e.g. returned None/0 for
    kappa), this head would get no gradient. Confirms the implicit reparam path.
    """
    model = _make_model()
    model.train()
    activations, beat, down = _make_batch()
    torch.manual_seed(5)
    out = model(activations, beat_targets=beat, downbeat_targets=down)
    out["beat_logits"].pow(2).mean().backward()

    # The fused posterior head produces phase_kappa at slot K+1; its weight
    # row should receive a finite, structurally-present gradient.
    head_w = model.trans_post_head.fused_head.weight
    assert head_w.grad is not None, "posterior fused head got NO gradient"
    assert torch.isfinite(head_w.grad).all(), "posterior fused head grad has NaN/Inf"
    # Specifically the kappa output row (slot K+1) must be nonzero — proves the
    # von Mises dz/dkappa backward delivered a real gradient.
    kappa_row_grad = head_w.grad[K + 1]
    assert kappa_row_grad.abs().sum() > 0, \
        "phase_kappa head row got ZERO gradient (implicit-reparam backward broken?)"


def test_gradients_finite_everywhere():
    model = _make_model()
    model.train()
    activations, beat, down = _make_batch()
    torch.manual_seed(11)
    out = model(activations, beat_targets=beat, downbeat_targets=down)
    out["beat_logits"].pow(2).mean().backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients in params: {bad}"


# ---------------------------------------------------------------------------
# Determinism / RNG contract
# ---------------------------------------------------------------------------

def test_seed_determinism_same_seed_identical():
    model = _make_model(seed=3)
    out1 = _run(model, seed=2024)
    out2 = _run(model, seed=2024)
    for grp in ("posterior", "prior", "samples"):
        for k in out1[grp]:
            a, b = out1[grp][k], out2[grp][k]
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), f"{grp}.{k} not reproducible under same seed"
    assert torch.equal(out1["beat_logits"], out2["beat_logits"])


def test_different_seed_changes_samples():
    model = _make_model(seed=3)
    out1 = _run(model, seed=1)
    out2 = _run(model, seed=2)
    # Sampling-driven outputs should differ for different RNG seeds.
    assert not torch.equal(out1["samples"]["phase"], out2["samples"]["phase"]), \
        "phase samples identical across different seeds (sampling not stochastic?)"


# ---------------------------------------------------------------------------
# Temperature behavior (analytic Gumbel-Softmax property, Monte-Carlo estimate)
# ---------------------------------------------------------------------------

def _mean_meter_entropy(model, temperature, n_reps=8, seed0=100):
    """Monte-Carlo estimate of the mean entropy of the meter_soft simplex
    samples at a given Gumbel-Softmax temperature."""
    activations, beat, down = _make_batch()
    ents = []
    for r in range(n_reps):
        torch.manual_seed(seed0 + r)
        ms = model(activations, beat_targets=beat, downbeat_targets=down,
                   temperature=temperature)["samples"]["meter_soft"]
        p = ms.clamp_min(1e-9)
        p = p / p.sum(-1, keepdim=True)
        ent = -(p * p.log()).sum(-1)  # [B,T]
        ents.append(ent.mean().item())
    return sum(ents) / len(ents)


def test_temperature_increases_softness():
    """Higher Gumbel-Softmax tau => softer samples => higher mean entropy.
    This is an analytic property of the relaxation, not a copied number.
    """
    model = _make_model(seed=42)
    ent_low = _mean_meter_entropy(model, temperature=0.1)
    ent_high = _mean_meter_entropy(model, temperature=3.0)
    assert ent_high > ent_low + 1e-3, (
        f"entropy did not increase with temperature: "
        f"low(tau=0.1)={ent_low:.4f}  high(tau=3.0)={ent_high:.4f}"
    )


# ---------------------------------------------------------------------------
# Optional-feature smoke tests (still structural invariants, not tautologies)
# ---------------------------------------------------------------------------

def test_bar_phase_outputs_present_and_finite():
    model = _make_model(seed=8, bar_phase=True)
    out = _run(model)
    s = out["samples"]
    assert "bar_phase" in s and s["bar_phase"].shape == (B, T)
    bp = s["bar_phase"]
    assert (bp >= 0).all() and (bp < TWO_PI + 1e-5).all(), "bar_phase not wrapped"
    assert torch.isfinite(out["beat_logits"]).all()
    assert "barphase_mu" in out["prior"] and out["prior"]["barphase_mu"].shape == (B, T)


def test_audio_emission_recon_shape_and_finite():
    model = _make_model(seed=9, audio_emission=True)
    out = _run(model)
    rec = out["audio_recon"]
    assert rec is not None and rec.shape == (B, T, IN), \
        f"audio_recon shape {None if rec is None else rec.shape}"
    assert torch.isfinite(rec).all(), "audio_recon non-finite"


def test_meter_ste_produces_hard_onehot():
    """meter_ste=True => hard straight-through Gumbel: forward sample is a
    one-hot vector (exactly one 1, rest 0) on the simplex."""
    model = _make_model(seed=10, meter_ste=True)
    ms = _run(model)["samples"]["meter_soft"]
    sums = ms.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "STE rows not summing to 1"
    maxv = ms.max(-1).values
    # one-hot => the max entry is exactly 1 in the forward pass
    assert torch.allclose(maxv, torch.ones_like(maxv), atol=1e-4), \
        f"meter_ste forward not one-hot (max entry {maxv.min().item():.4f})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
