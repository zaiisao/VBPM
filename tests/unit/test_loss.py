"""Unit tests for models/loss.py :: compute_elbo_loss.

Ground truth strategy:
- BCE reconstructed independently with F.binary_cross_entropy_with_logits.
- Each KL component reconstructed from the closed-form KL in models.distributions
  (these are themselves separately tested; here we verify loss.py wires them up
  and applies free-bits / mean-reduction correctly).
- The `total` is reconstructed from its documented composition (bce + beta*sum(kl)
  + weighted sup terms) using inputs crafted so each term is independently known.
- Supervision terms reconstructed from their closed-form circular / squared loss.
- Structural invariants: all KL >= 0, finiteness, free-bits floor, gradient flow.
"""

import math
import sys

sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import pytest
import torch
import torch.nn.functional as F

from models.loss import compute_elbo_loss
from models.distributions import (
    categorical_kl,
    von_mises_kl,
    lognormal_kl,
)

torch.manual_seed(0)

# Tolerance for float32 comparisons.
ATOL = 1e-5
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Builders for valid posterior / prior dicts
# ---------------------------------------------------------------------------

def make_posterior(B=2, T=5, K=4, seed=1, barphase=True):
    g = torch.Generator().manual_seed(seed)
    post = {
        "meter_logits": torch.randn(B, T, K, generator=g),
        "phase_mu": torch.randn(B, T, generator=g),
        # log_kappa => kappa moderate, away from extremes
        "phase_log_kappa": torch.randn(B, T, generator=g) * 0.5,
        "tempo_mu": torch.randn(B, T, generator=g),
        "tempo_log_sigma": torch.randn(B, T, generator=g) * 0.3 - 0.5,
    }
    if barphase:
        post["barphase_mu"] = torch.randn(B, T, generator=g)
        post["barphase_log_kappa"] = torch.randn(B, T, generator=g) * 0.5
    return post


def make_prior(B=2, T=5, K=4, seed=2, barphase=True):
    g = torch.Generator().manual_seed(seed)
    prior = {
        "meter_logits": torch.randn(B, T, K, generator=g),
        "phase_mu": torch.randn(B, T, generator=g),
        "phase_kappa": torch.rand(B, T, generator=g) * 3.0 + 0.5,
        "tempo_mu": torch.randn(B, T, generator=g),
        "tempo_sigma": torch.rand(B, T, generator=g) * 0.5 + 0.2,
    }
    if barphase:
        prior["barphase_mu"] = torch.randn(B, T, generator=g)
        prior["barphase_kappa"] = torch.rand(B, T, generator=g) * 3.0 + 0.5
    return prior


def make_targets(B=2, T=5, seed=3):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, T, generator=g) > 0.5).float()


def make_logits(B=2, T=5, seed=4):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 2, generator=g)


# Independent free-bits-aware KL reduction, mirroring the documented behaviour.
def reduce_kl(kl, fb):
    if fb > 0.0:
        return kl.mean(dim=-1).clamp(min=fb).mean()
    return kl.mean()


# ---------------------------------------------------------------------------
# Return contract
# ---------------------------------------------------------------------------

def test_return_shape_and_keys():
    post, prior = make_posterior(), make_prior()
    logits, targets = make_logits(), make_targets()
    total, comp = compute_elbo_loss(logits, targets, post, prior)

    assert isinstance(total, torch.Tensor)
    assert total.shape == ()  # scalar
    assert isinstance(comp, dict)

    expected = {
        "bce", "kl_meter", "kl_phase", "kl_tempo", "kl_taubar", "kl_barphase",
        "taubar_sup", "meter_sup", "phase_sup", "barphase_sup", "tempo_density",
    }
    assert expected.issubset(comp.keys()), comp.keys()
    for k, v in comp.items():
        assert torch.isfinite(v).all(), f"{k} not finite: {v}"
        assert v.shape == (), f"{k} not scalar: {v.shape}"


def test_all_kl_components_nonnegative():
    # Run over several random seeds: KL is a divergence -> must be >= 0.
    for s in range(8):
        post = make_posterior(seed=10 + s)
        prior = make_prior(seed=100 + s)
        logits, targets = make_logits(seed=s), make_targets(seed=s)
        _, comp = compute_elbo_loss(logits, targets, post, prior)
        for key in ("kl_meter", "kl_phase", "kl_tempo", "kl_taubar", "kl_barphase"):
            assert comp[key].item() >= -1e-6, f"{key}={comp[key].item()} negative (seed {s})"


# ---------------------------------------------------------------------------
# BCE reconstruction vs F.binary_cross_entropy_with_logits ground truth
# ---------------------------------------------------------------------------

def test_bce_beat_only_matches_manual():
    post, prior = make_posterior(), make_prior()
    logits = make_logits()
    targets = make_targets()
    _, comp = compute_elbo_loss(logits, targets, post, prior)

    expected = F.binary_cross_entropy_with_logits(
        logits[:, :, 0], targets, reduction="mean"
    )
    assert torch.allclose(comp["bce"], expected, atol=ATOL, rtol=RTOL), (
        comp["bce"].item(), expected.item()
    )


def test_bce_with_downbeat_is_sum_of_two():
    post, prior = make_posterior(), make_prior()
    logits = make_logits()
    targets = make_targets()
    db = make_targets(seed=99)
    _, comp = compute_elbo_loss(logits, targets, post, prior, downbeat_targets=db)

    rb = F.binary_cross_entropy_with_logits(logits[:, :, 0], targets, reduction="mean")
    rd = F.binary_cross_entropy_with_logits(logits[:, :, 1], db, reduction="mean")
    assert torch.allclose(comp["bce"], rb + rd, atol=ATOL, rtol=RTOL)


def test_bce_pos_weight_matches_manual():
    post, prior = make_posterior(), make_prior()
    logits = make_logits()
    targets = make_targets()
    db = make_targets(seed=99)
    pw, pw_db = 3.0, 5.0
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        downbeat_targets=db, pos_weight=pw, pos_weight_db=pw_db,
    )
    rb = F.binary_cross_entropy_with_logits(
        logits[:, :, 0], targets, pos_weight=torch.tensor(pw), reduction="mean"
    )
    rd = F.binary_cross_entropy_with_logits(
        logits[:, :, 1], db, pos_weight=torch.tensor(pw_db), reduction="mean"
    )
    assert torch.allclose(comp["bce"], rb + rd, atol=ATOL, rtol=RTOL)


def test_bce_pos_weight_db_defaults_to_pos_weight():
    # When pos_weight_db is None, the downbeat channel should use pos_weight.
    post, prior = make_posterior(), make_prior()
    logits = make_logits()
    targets = make_targets()
    db = make_targets(seed=99)
    pw = 4.0
    _, comp = compute_elbo_loss(
        logits, targets, post, prior, downbeat_targets=db, pos_weight=pw,
    )
    rb = F.binary_cross_entropy_with_logits(
        logits[:, :, 0], targets, pos_weight=torch.tensor(pw), reduction="mean"
    )
    rd = F.binary_cross_entropy_with_logits(
        logits[:, :, 1], db, pos_weight=torch.tensor(pw), reduction="mean"
    )
    assert torch.allclose(comp["bce"], rb + rd, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# KL components reconstruct from closed-form distributions
# ---------------------------------------------------------------------------

def test_kl_components_match_distribution_formulas_no_freebits():
    post, prior = make_posterior(), make_prior()
    logits, targets = make_logits(), make_targets()
    _, comp = compute_elbo_loss(logits, targets, post, prior)

    kl_m = categorical_kl(post["meter_logits"], prior["meter_logits"]).mean()
    kl_phi = von_mises_kl(
        post["phase_mu"], post["phase_log_kappa"].exp(),
        prior["phase_mu"], prior["phase_kappa"],
    ).mean()
    kl_bp = von_mises_kl(
        post["barphase_mu"], post["barphase_log_kappa"].exp(),
        prior["barphase_mu"], prior["barphase_kappa"],
    ).mean()
    kl_t = lognormal_kl(
        post["tempo_mu"], post["tempo_log_sigma"].exp(),
        prior["tempo_mu"], prior["tempo_sigma"],
    ).mean()

    assert torch.allclose(comp["kl_meter"], kl_m, atol=ATOL, rtol=RTOL)
    assert torch.allclose(comp["kl_phase"], kl_phi, atol=ATOL, rtol=RTOL)
    assert torch.allclose(comp["kl_barphase"], kl_bp, atol=ATOL, rtol=RTOL)
    assert torch.allclose(comp["kl_tempo"], kl_t, atol=ATOL, rtol=RTOL)


def test_kl_barphase_zero_when_absent():
    # No barphase keys in posterior/prior -> kl_barphase must be exactly 0.
    post = make_posterior(barphase=False)
    prior = make_prior(barphase=False)
    logits, targets = make_logits(), make_targets()
    _, comp = compute_elbo_loss(logits, targets, post, prior)
    assert comp["kl_barphase"].item() == 0.0


def test_kl_taubar_zero_when_no_tempo_bar():
    post, prior = make_posterior(), make_prior()
    logits, targets = make_logits(), make_targets()
    _, comp = compute_elbo_loss(logits, targets, post, prior)
    assert comp["kl_taubar"].item() == 0.0


def test_kl_taubar_matches_lognormal():
    B = 2
    post, prior = make_posterior(B=B), make_prior(B=B)
    logits, targets = make_logits(B=B), make_targets(B=B)
    g = torch.Generator().manual_seed(7)
    tempo_bar = {
        "mu_q": torch.randn(B, generator=g),
        "sigma_q": torch.rand(B, generator=g) * 0.5 + 0.2,
        "mu_p": torch.randn(B, generator=g),
        "sigma_p": torch.rand(B, generator=g) * 0.5 + 0.2,
    }
    _, comp = compute_elbo_loss(logits, targets, post, prior, tempo_bar=tempo_bar)
    expect = lognormal_kl(
        tempo_bar["mu_q"], tempo_bar["sigma_q"],
        tempo_bar["mu_p"], tempo_bar["sigma_p"],
    ).mean()
    assert torch.allclose(comp["kl_taubar"], expect, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Free-bits behaviour
# ---------------------------------------------------------------------------

def test_free_bits_phase_floor_when_raw_kl_tiny():
    # Make posterior phase == prior phase => raw KL ~ 0, below the 0.3 floor.
    B, T = 2, 5
    post = make_posterior(B=B, T=T)
    prior = make_prior(B=B, T=T)
    # Match phase q and p exactly so von Mises KL is ~0.
    prior["phase_mu"] = post["phase_mu"].clone()
    prior["phase_kappa"] = post["phase_log_kappa"].exp().clone()

    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)

    fb = 0.3
    _, comp = compute_elbo_loss(
        logits, targets, post, prior, free_bits_phase=fb,
    )
    # raw KL nearly zero, but reported clamps to >= fb (per-batch mean then mean).
    raw = von_mises_kl(
        post["phase_mu"], post["phase_log_kappa"].exp(),
        prior["phase_mu"], prior["phase_kappa"],
    )
    assert raw.mean().item() < fb, raw.mean().item()
    assert comp["kl_phase"].item() >= fb - 1e-6, comp["kl_phase"].item()
    # And it should equal the clamped reduction precisely.
    expect = reduce_kl(raw, fb)
    assert torch.allclose(comp["kl_phase"], expect, atol=ATOL, rtol=RTOL)


def test_free_bits_does_not_inflate_large_kl():
    # When raw per-row KL already exceeds the floor, free-bits is a no-op.
    post, prior = make_posterior(), make_prior()
    logits, targets = make_logits(), make_targets()
    raw = von_mises_kl(
        post["phase_mu"], post["phase_log_kappa"].exp(),
        prior["phase_mu"], prior["phase_kappa"],
    )
    fb = 1e-4  # tiny floor, definitely below per-row means
    assert raw.mean(dim=-1).min().item() > fb
    _, comp = compute_elbo_loss(logits, targets, post, prior, free_bits_phase=fb)
    assert torch.allclose(comp["kl_phase"], reduce_kl(raw, fb), atol=ATOL, rtol=RTOL)
    # And equals the plain mean since floor is inactive.
    assert torch.allclose(comp["kl_phase"], raw.mean(), atol=1e-4, rtol=1e-3)


def test_global_free_bits_applies_to_all_when_specific_none():
    # free_bits global default should be used for each latent lacking a specific one.
    B, T = 2, 5
    post = make_posterior(B=B, T=T)
    prior = make_prior(B=B, T=T)
    # Collapse phase KL to ~0 to make the floor bite.
    prior["phase_mu"] = post["phase_mu"].clone()
    prior["phase_kappa"] = post["phase_log_kappa"].exp().clone()
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)

    fb = 0.5
    _, comp = compute_elbo_loss(logits, targets, post, prior, free_bits=fb)
    assert comp["kl_phase"].item() >= fb - 1e-6


# ---------------------------------------------------------------------------
# Total reconstruction
# ---------------------------------------------------------------------------

def test_total_equals_bce_plus_beta_sum_kl():
    # Pure ELBO (no sup, no density). total = bce + beta*sum(kl).
    B = 2
    post, prior = make_posterior(B=B), make_prior(B=B)
    logits, targets = make_logits(B=B), make_targets(B=B)
    g = torch.Generator().manual_seed(11)
    tempo_bar = {
        "mu_q": torch.randn(B, generator=g),
        "sigma_q": torch.rand(B, generator=g) * 0.4 + 0.2,
        "mu_p": torch.randn(B, generator=g),
        "sigma_p": torch.rand(B, generator=g) * 0.4 + 0.2,
    }
    beta = 0.7
    total, comp = compute_elbo_loss(
        logits, targets, post, prior, beta=beta, tempo_bar=tempo_bar,
    )
    kl_sum = (
        comp["kl_meter"] + comp["kl_phase"] + comp["kl_tempo"]
        + comp["kl_taubar"] + comp["kl_barphase"]
    )
    expect = comp["bce"] + beta * kl_sum
    assert torch.allclose(total, expect, atol=ATOL, rtol=RTOL), (total.item(), expect.item())


def test_total_includes_weighted_sup_terms():
    # Add all supervision terms with distinct weights; verify total decomposition.
    B, T, K = 2, 6, 4
    post, prior = make_posterior(B=B, T=T, K=K), make_prior(B=B, T=T, K=K)
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)
    g = torch.Generator().manual_seed(13)

    tempo_bar = {
        "mu_q": torch.randn(B, generator=g),
        "sigma_q": torch.rand(B, generator=g) * 0.4 + 0.2,
        "mu_p": torch.randn(B, generator=g),
        "sigma_p": torch.rand(B, generator=g) * 0.4 + 0.2,
    }
    barphase_targets = torch.randn(B, T, generator=g)
    phase_targets = torch.randn(B, T, generator=g)
    meter_targets = torch.randint(0, K, (B, T), generator=g)

    w = dict(
        barphase_sup_weight=0.3,
        taubar_sup_weight=0.4,
        meter_sup_weight=0.5,
        phase_sup_weight=0.6,
        tempo_density_weight=0.25,
    )
    beta = 1.0
    total, comp = compute_elbo_loss(
        logits, targets, post, prior, beta=beta,
        tempo_bar=tempo_bar,
        barphase_targets=barphase_targets,
        phase_targets=phase_targets,
        meter_targets=meter_targets,
        **w,
    )
    kl_sum = (
        comp["kl_meter"] + comp["kl_phase"] + comp["kl_tempo"]
        + comp["kl_taubar"] + comp["kl_barphase"]
    )
    expect = (
        comp["bce"]
        + beta * kl_sum
        + w["tempo_density_weight"] * comp["tempo_density"]
        + w["taubar_sup_weight"] * comp["taubar_sup"]
        + w["meter_sup_weight"] * comp["meter_sup"]
        + w["phase_sup_weight"] * comp["phase_sup"]
        + w["barphase_sup_weight"] * comp["barphase_sup"]
    )
    assert torch.allclose(total, expect, atol=1e-4, rtol=1e-3), (total.item(), expect.item())


# ---------------------------------------------------------------------------
# Supervision-term closed forms
# ---------------------------------------------------------------------------

def test_barphase_sup_circular_loss():
    B, T = 2, 5
    post, prior = make_posterior(B=B, T=T), make_prior(B=B, T=T)
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)
    g = torch.Generator().manual_seed(21)
    bt = torch.randn(B, T, generator=g)
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        barphase_targets=bt, barphase_sup_weight=1.0,
    )
    expect = (1.0 - torch.cos(prior["barphase_mu"] - bt)).mean()
    assert torch.allclose(comp["barphase_sup"], expect, atol=ATOL, rtol=RTOL)


def test_barphase_sup_accepts_3d_targets():
    B, T = 2, 5
    post, prior = make_posterior(B=B, T=T), make_prior(B=B, T=T)
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)
    g = torch.Generator().manual_seed(22)
    bt = torch.randn(B, T, 1, generator=g)
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        barphase_targets=bt, barphase_sup_weight=1.0,
    )
    expect = (1.0 - torch.cos(prior["barphase_mu"] - bt.squeeze(-1))).mean()
    assert torch.allclose(comp["barphase_sup"], expect, atol=ATOL, rtol=RTOL)


def test_phase_sup_circular_loss():
    B, T = 2, 5
    post, prior = make_posterior(B=B, T=T), make_prior(B=B, T=T)
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)
    g = torch.Generator().manual_seed(23)
    pt = torch.randn(B, T, generator=g)
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        phase_targets=pt, phase_sup_weight=1.0,
    )
    expect = (1.0 - torch.cos(prior["phase_mu"] - pt)).mean()
    assert torch.allclose(comp["phase_sup"], expect, atol=ATOL, rtol=RTOL)


def test_phase_sup_bounds_zero_to_two():
    # 1 - cos(.) is in [0, 2]; identical alignment -> exactly 0.
    B, T = 2, 5
    post, prior = make_posterior(B=B, T=T), make_prior(B=B, T=T)
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)
    pt = prior["phase_mu"].clone()
    _, comp = compute_elbo_loss(
        logits, targets, post, prior, phase_targets=pt, phase_sup_weight=1.0,
    )
    assert abs(comp["phase_sup"].item()) < 1e-6
    # Anti-aligned (pi offset) -> exactly 2.
    pt2 = prior["phase_mu"] + math.pi
    _, comp2 = compute_elbo_loss(
        logits, targets, post, prior, phase_targets=pt2, phase_sup_weight=1.0,
    )
    assert abs(comp2["phase_sup"].item() - 2.0) < 1e-5


def test_taubar_sup_squared_log_tempo():
    # taubar_sup = mean((mu_q - log(2pi*N/T))^2).
    B, T = 2, 8
    post, prior = make_posterior(B=B, T=T), make_prior(B=B, T=T)
    logits = make_logits(B=B, T=T)
    # Deterministic targets with a known beat count per row.
    targets = torch.zeros(B, T)
    targets[0, [0, 2, 4, 6]] = 1.0  # 4 beats
    targets[1, [1, 5]] = 1.0        # 2 beats
    g = torch.Generator().manual_seed(31)
    tempo_bar = {
        "mu_q": torch.randn(B, generator=g),
        "sigma_q": torch.rand(B, generator=g) * 0.4 + 0.2,
        "mu_p": torch.randn(B, generator=g),
        "sigma_p": torch.rand(B, generator=g) * 0.4 + 0.2,
    }
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        tempo_bar=tempo_bar, taubar_sup_weight=1.0,
    )
    two_pi = 2.0 * math.pi
    n_beats = targets.sum(dim=1).clamp(min=1.0)
    tgt = torch.log(two_pi * n_beats / T)
    expect = ((tempo_bar["mu_q"] - tgt) ** 2).mean()
    assert torch.allclose(comp["taubar_sup"], expect, atol=ATOL, rtol=RTOL)


def test_meter_sup_cross_entropy_index_targets():
    B, T, K = 2, 6, 4
    post, prior = make_posterior(B=B, T=T, K=K), make_prior(B=B, T=T, K=K)
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)
    g = torch.Generator().manual_seed(41)
    meter_targets = torch.randint(0, K, (B, T), generator=g)
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        meter_targets=meter_targets, meter_sup_weight=1.0,
    )
    expect = F.cross_entropy(
        post["meter_logits"].reshape(-1, K), meter_targets.reshape(-1)
    )
    assert torch.allclose(comp["meter_sup"], expect, atol=ATOL, rtol=RTOL)


def test_meter_sup_one_hot_targets():
    # One-hot (3D) meter targets should be argmaxed and give same CE.
    B, T, K = 2, 6, 4
    post, prior = make_posterior(B=B, T=T, K=K), make_prior(B=B, T=T, K=K)
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)
    g = torch.Generator().manual_seed(42)
    idx = torch.randint(0, K, (B, T), generator=g)
    one_hot = F.one_hot(idx, K).float()
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        meter_targets=one_hot, meter_sup_weight=1.0,
    )
    expect = F.cross_entropy(post["meter_logits"].reshape(-1, K), idx.reshape(-1))
    assert torch.allclose(comp["meter_sup"], expect, atol=ATOL, rtol=RTOL)


def test_tempo_density_squared_loss():
    B, T = 2, 8
    post, prior = make_posterior(B=B, T=T), make_prior(B=B, T=T)
    logits = make_logits(B=B, T=T)
    targets = torch.zeros(B, T)
    targets[0, [0, 2, 4, 6]] = 1.0
    targets[1, [1, 5]] = 1.0
    _, comp = compute_elbo_loss(
        logits, targets, post, prior, tempo_density_weight=1.0,
    )
    two_pi = 2.0 * math.pi
    n_beats = targets.sum(dim=1).clamp(min=1.0)
    tgt = torch.log(two_pi * n_beats / T)
    pred = prior["tempo_mu"].mean(dim=1)
    expect = ((pred - tgt) ** 2).mean()
    assert torch.allclose(comp["tempo_density"], expect, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Sup terms are gated off (== 0) when weight is 0 or targets are None
# ---------------------------------------------------------------------------

def test_sup_terms_zero_when_weight_zero():
    B, T, K = 2, 6, 4
    post, prior = make_posterior(B=B, T=T, K=K), make_prior(B=B, T=T, K=K)
    logits, targets = make_logits(B=B, T=T), make_targets(B=B, T=T)
    g = torch.Generator().manual_seed(51)
    # Provide all targets but with zero weights.
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        barphase_targets=torch.randn(B, T, generator=g), barphase_sup_weight=0.0,
        phase_targets=torch.randn(B, T, generator=g), phase_sup_weight=0.0,
        meter_targets=torch.randint(0, K, (B, T), generator=g), meter_sup_weight=0.0,
    )
    for key in ("barphase_sup", "phase_sup", "meter_sup", "taubar_sup", "tempo_density"):
        assert comp[key].item() == 0.0, f"{key}={comp[key].item()}"


def test_sup_terms_zero_when_targets_none():
    post, prior = make_posterior(), make_prior()
    logits, targets = make_logits(), make_targets()
    # weights > 0 but targets None -> should still be 0 (guarded).
    _, comp = compute_elbo_loss(
        logits, targets, post, prior,
        barphase_sup_weight=1.0, phase_sup_weight=1.0, meter_sup_weight=1.0,
        taubar_sup_weight=1.0,
    )
    for key in ("barphase_sup", "phase_sup", "meter_sup", "taubar_sup"):
        assert comp[key].item() == 0.0, f"{key}={comp[key].item()}"


# ---------------------------------------------------------------------------
# beta scaling: KL terms scale linearly, bce unaffected
# ---------------------------------------------------------------------------

def test_beta_scales_kl_only():
    B = 2
    post, prior = make_posterior(B=B), make_prior(B=B)
    logits, targets = make_logits(B=B), make_targets(B=B)
    t0, c0 = compute_elbo_loss(logits, targets, post, prior, beta=0.0)
    t1, c1 = compute_elbo_loss(logits, targets, post, prior, beta=1.0)
    t2, c2 = compute_elbo_loss(logits, targets, post, prior, beta=2.0)
    # bce identical across beta.
    assert torch.allclose(c0["bce"], c1["bce"]) and torch.allclose(c1["bce"], c2["bce"])
    # at beta=0, total == bce (no sup terms).
    assert torch.allclose(t0, c0["bce"], atol=ATOL, rtol=RTOL)
    # total grows linearly in beta: (t2 - t1) == (t1 - t0).
    assert torch.allclose(t2 - t1, t1 - t0, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Gradient flow: total must produce gradients on learnable inputs
# ---------------------------------------------------------------------------

def test_gradients_flow_to_logits_and_posterior():
    B, T, K = 2, 5, 4
    post = make_posterior(B=B, T=T, K=K)
    prior = make_prior(B=B, T=T, K=K)
    logits = make_logits(B=B, T=T).requires_grad_(True)
    targets = make_targets(B=B, T=T)
    for v in post.values():
        v.requires_grad_(True)

    total, _ = compute_elbo_loss(logits, targets, post, prior)
    total.backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum() > 0).item()
    for k, v in post.items():
        assert v.grad is not None, f"no grad for posterior[{k}]"
        assert torch.isfinite(v.grad).all(), f"non-finite grad for posterior[{k}]"


def test_components_are_detached():
    # The returned components dict must be detached (no grad graph).
    post, prior = make_posterior(), make_prior()
    logits = make_logits().requires_grad_(True)
    targets = make_targets()
    _, comp = compute_elbo_loss(logits, targets, post, prior)
    for k, v in comp.items():
        assert not v.requires_grad, f"{k} still requires grad"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
