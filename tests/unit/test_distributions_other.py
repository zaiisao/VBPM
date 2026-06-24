"""Unit tests for models/distributions.py — lognormal / gumbel / categorical.

Ground truth sources:
- Closed-form Gaussian KL (validated against scipy MC + analytic identities).
- Monte-Carlo estimates of analytic quantities (E[log z], Var[log z], E[z]).
- Finite-difference / autograd for reparameterization gradients.
- Structural invariants: simplex (rows sum to 1, >= 0), one-hot, entropy
  monotonicity in temperature, KL >= 0, KL(p||p) == 0, no-NaN.
"""

import sys

sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import numpy as np
import pytest
import torch

from models.distributions import (
    categorical_kl,
    gumbel_softmax_sample,
    lognormal_kl,
    lognormal_sample,
    lognormal_sample_logspace,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_kl_numpy(mu_q, sig_q, mu_p, sig_p):
    """Analytic KL(N(mu_q,sig_q^2)||N(mu_p,sig_p^2)) computed independently in numpy."""
    return (
        np.log(sig_p / sig_q)
        + (sig_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sig_p ** 2)
        - 0.5
    )


def _categorical_kl_numpy(logits_q, logits_p):
    """Independent numpy implementation of KL(Cat(q)||Cat(p)) from logits."""
    def log_softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

    log_q = log_softmax(logits_q)
    log_p = log_softmax(logits_p)
    q = np.exp(log_q)
    return (q * (log_q - log_p)).sum(axis=-1)


def _entropy(probs):
    """Mean Shannon entropy over a batch of probability rows."""
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# lognormal_sample_logspace — empirical moments of the LOG-tempo
# ---------------------------------------------------------------------------

def test_lognormal_logspace_empirical_mean_var():
    """log-space sample ~ N(mu, sigma^2): empirical mean/var match params."""
    torch.manual_seed(123)
    mu = torch.tensor(0.7)
    sigma = torch.tensor(1.3)
    N = 400_000
    mu_b = mu.expand(N)
    sig_b = sigma.expand(N)
    s = lognormal_sample_logspace(mu_b, sig_b)
    assert torch.isfinite(s).all()
    # SE of mean ~ sigma/sqrt(N) ~ 0.002; allow generous tolerance.
    assert abs(s.mean().item() - mu.item()) < 0.02
    # SE of variance ~ sigma^2 * sqrt(2/N); allow generous tolerance.
    assert abs(s.var(unbiased=True).item() - sigma.item() ** 2) < 0.03


def test_lognormal_sample_positive_and_logmoments():
    """Original-space sample is positive; its log matches N(mu, sigma^2)."""
    torch.manual_seed(7)
    mu = torch.tensor(-0.4)
    sigma = torch.tensor(0.9)
    N = 400_000
    s = lognormal_sample(mu.expand(N), sigma.expand(N))
    assert torch.isfinite(s).all()
    assert (s > 0).all(), "LogNormal sample must be strictly positive"
    logs = s.log()
    assert abs(logs.mean().item() - mu.item()) < 0.02
    assert abs(logs.var(unbiased=True).item() - sigma.item() ** 2) < 0.03


def test_lognormal_sample_mean_matches_analytic():
    """E[z] for z~LogNormal == exp(mu + sigma^2/2)."""
    torch.manual_seed(11)
    mu = torch.tensor(0.2)
    sigma = torch.tensor(0.5)
    N = 1_000_000
    s = lognormal_sample(mu.expand(N), sigma.expand(N))
    analytic = math.exp(mu.item() + 0.5 * sigma.item() ** 2)
    # MC SE of the mean for LogNormal is non-trivial; allow ~1.5% relative error.
    assert abs(s.mean().item() - analytic) / analytic < 0.02


# ---------------------------------------------------------------------------
# lognormal reparameterization gradients (autograd)
# ---------------------------------------------------------------------------

def test_lognormal_logspace_reparam_grad_dmu():
    """d/dmu E[mu + sigma*eps] == 1 exactly (per-sample reparam grad)."""
    torch.manual_seed(3)
    mu = torch.tensor([0.1, 1.0, -2.0], requires_grad=True)
    sigma = torch.tensor([0.5, 1.2, 0.3])
    s = lognormal_sample_logspace(mu, sigma)
    s.sum().backward()
    # For each element dz_i/dmu_i = 1 regardless of the (fixed) eps draw.
    assert torch.allclose(mu.grad, torch.ones_like(mu), atol=1e-6)


def test_lognormal_logspace_reparam_grad_dsigma_equals_eps():
    """d s / d sigma == eps; recover eps = (s - mu)/sigma and check grad."""
    torch.manual_seed(5)
    mu = torch.tensor([0.0, 0.5])
    sigma = torch.tensor([0.8, 1.5], requires_grad=True)
    s = lognormal_sample_logspace(mu, sigma)
    eps = ((s - mu) / sigma).detach()
    s.sum().backward()
    assert torch.allclose(sigma.grad, eps, atol=1e-6)


def test_lognormal_origspace_reparam_grad_dmu_expectation():
    """For z=exp(mu+sigma*eps): d/dmu E[z] == E[z] == exp(mu+sigma^2/2) (MC)."""
    torch.manual_seed(9)
    mu = torch.tensor(0.3, requires_grad=True)
    sigma = torch.tensor(0.6)
    N = 1_000_000
    s = lognormal_sample(mu.expand(N), sigma.expand(N))
    loss = s.mean()
    loss.backward()
    analytic = math.exp(mu.item() + 0.5 * sigma.item() ** 2)
    # d E[z]/dmu = E[z] for lognormal.
    assert abs(mu.grad.item() - analytic) / analytic < 0.02


# ---------------------------------------------------------------------------
# lognormal_kl — analytic vs numpy, nonnegativity, self-KL == 0
# ---------------------------------------------------------------------------

def test_lognormal_kl_matches_numpy_gaussian():
    """lognormal_kl == closed-form Gaussian KL computed independently in numpy."""
    torch.manual_seed(1)
    mu_q = torch.randn(64)
    sig_q = torch.rand(64) * 2.0 + 0.1
    mu_p = torch.randn(64)
    sig_p = torch.rand(64) * 2.0 + 0.1
    out = lognormal_kl(mu_q, sig_q, mu_p, sig_p)
    ref = _gaussian_kl_numpy(
        mu_q.numpy(), sig_q.numpy(), mu_p.numpy(), sig_p.numpy()
    )
    assert torch.isfinite(out).all()
    assert np.allclose(out.numpy(), ref, atol=1e-5)


def test_lognormal_kl_self_zero():
    """KL(p||p) == 0."""
    torch.manual_seed(2)
    mu = torch.randn(50)
    sig = torch.rand(50) + 0.2
    out = lognormal_kl(mu, sig, mu.clone(), sig.clone())
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_lognormal_kl_nonnegative():
    """KL >= 0 for arbitrary parameter pairs."""
    torch.manual_seed(4)
    mu_q = torch.randn(200)
    sig_q = torch.rand(200) * 3.0 + 0.05
    mu_p = torch.randn(200)
    sig_p = torch.rand(200) * 3.0 + 0.05
    out = lognormal_kl(mu_q, sig_q, mu_p, sig_p)
    assert (out >= -1e-6).all(), f"min KL = {out.min().item()}"


def test_lognormal_kl_matches_mc_estimate():
    """Closed-form KL matches a Monte-Carlo estimate E_q[log q - log p] (log-space)."""
    torch.manual_seed(6)
    mu_q, sig_q = 0.5, 1.1
    mu_p, sig_p = -0.3, 0.7
    N = 2_000_000
    x = mu_q + sig_q * torch.randn(N)

    def log_normal_pdf(x, m, s):
        return -0.5 * math.log(2 * math.pi) - math.log(s) - 0.5 * ((x - m) / s) ** 2

    mc = (log_normal_pdf(x, mu_q, sig_q) - log_normal_pdf(x, mu_p, sig_p)).mean()
    closed = lognormal_kl(
        torch.tensor(mu_q), torch.tensor(sig_q),
        torch.tensor(mu_p), torch.tensor(sig_p),
    )
    assert abs(mc.item() - closed.item()) < 0.01


# ---------------------------------------------------------------------------
# categorical_kl — analytic vs numpy, nonnegativity, self-KL == 0, MC
# ---------------------------------------------------------------------------

def test_categorical_kl_matches_numpy():
    """categorical_kl from logits == independent numpy reference."""
    torch.manual_seed(8)
    lq = torch.randn(32, 5)
    lp = torch.randn(32, 5)
    out = categorical_kl(lq, lp)
    ref = _categorical_kl_numpy(lq.numpy(), lp.numpy())
    assert out.shape == (32,)
    assert torch.isfinite(out).all()
    assert np.allclose(out.numpy(), ref, atol=1e-5)


def test_categorical_kl_self_zero():
    """KL(q||q) == 0; also invariant to an additive constant in the prior logits."""
    torch.manual_seed(10)
    lq = torch.randn(16, 4)
    out = categorical_kl(lq, lq.clone())
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    # Logit shift invariance: softmax(lp) == softmax(lp + c).
    shifted = categorical_kl(lq, lq + 3.7)
    assert torch.allclose(shifted, torch.zeros_like(shifted), atol=1e-5)


def test_categorical_kl_nonnegative():
    """KL >= 0 for arbitrary logit pairs."""
    torch.manual_seed(12)
    lq = torch.randn(500, 7) * 2.0
    lp = torch.randn(500, 7) * 2.0
    out = categorical_kl(lq, lp)
    assert (out >= -1e-6).all(), f"min KL = {out.min().item()}"


def test_categorical_kl_matches_mc():
    """KL matches MC estimate sum_k q_k (log q_k - log p_k) sampled from q."""
    torch.manual_seed(13)
    lq = torch.randn(3)
    lp = torch.randn(3)
    q = torch.softmax(lq, dim=-1)
    p = torch.softmax(lp, dim=-1)
    N = 2_000_000
    idx = torch.multinomial(q, N, replacement=True)
    mc = (q[idx].log() - p[idx].log()).mean()
    closed = categorical_kl(lq, lp)
    assert abs(mc.item() - closed.item()) < 0.005


def test_categorical_kl_known_uniform_value():
    """KL(delta_0 || Uniform_K) == log K for a (near) one-hot posterior."""
    K = 4
    lq = torch.tensor([100.0, -100.0, -100.0, -100.0])  # essentially delta on class 0
    lp = torch.zeros(K)  # uniform
    out = categorical_kl(lq, lp)
    assert abs(out.item() - math.log(K)) < 1e-4


# ---------------------------------------------------------------------------
# gumbel_softmax_sample — simplex / one-hot / temperature monotonicity / NaN
# ---------------------------------------------------------------------------

def test_gumbel_soft_is_valid_simplex():
    """Soft Gumbel-Softmax rows sum to 1 and are nonnegative."""
    torch.manual_seed(14)
    logits = torch.randn(256, 6)
    soft = gumbel_softmax_sample(logits, temperature=0.5, hard=False)
    assert soft.shape == logits.shape
    assert torch.isfinite(soft).all()
    assert (soft >= 0).all()
    assert torch.allclose(soft.sum(dim=-1), torch.ones(256), atol=1e-5)


def test_gumbel_hard_is_one_hot():
    """hard=True yields exact one-hot rows (single 1, rest 0)."""
    torch.manual_seed(15)
    logits = torch.randn(256, 5)
    hard = gumbel_softmax_sample(logits, temperature=0.7, hard=True)
    assert torch.isfinite(hard).all()
    # Exactly one entry == 1 per row, all values in {0,1}.
    row_sums = hard.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(256), atol=1e-6)
    is_binary = ((hard == 0) | (hard == 1)).all()
    assert is_binary, "hard sample must be strictly 0/1"
    assert (hard.max(dim=-1).values == 1).all()


def test_gumbel_lower_temp_is_sharper():
    """Lower temperature -> lower mean entropy (sharper distribution)."""
    torch.manual_seed(16)
    logits = torch.randn(20000, 4)
    soft_cold = gumbel_softmax_sample(logits, temperature=0.1, hard=False)
    soft_hot = gumbel_softmax_sample(logits, temperature=5.0, hard=False)
    h_cold = _entropy(soft_cold)
    h_hot = _entropy(soft_hot)
    assert torch.isfinite(soft_cold).all() and torch.isfinite(soft_hot).all()
    assert h_cold.item() < h_hot.item(), (
        f"cold entropy {h_cold.item()} should be < hot {h_hot.item()}"
    )


def test_gumbel_hard_marginal_matches_softmax():
    """Mean of hard one-hot samples approximates softmax(logits) probabilities."""
    torch.manual_seed(17)
    logits = torch.tensor([2.0, 0.0, -1.0, 1.0])
    p = torch.softmax(logits, dim=-1)
    N = 200000
    batch = logits.expand(N, -1)
    hard = gumbel_softmax_sample(batch, temperature=1.0, hard=True)
    emp = hard.mean(dim=0)
    # MC SE ~ sqrt(p(1-p)/N) ~ 1e-3; allow generous tolerance.
    assert torch.allclose(emp, p, atol=0.01), f"emp={emp}, p={p}"


def test_gumbel_soft_grad_flows():
    """Soft Gumbel-Softmax is differentiable w.r.t. logits (no NaN grad)."""
    torch.manual_seed(18)
    logits = torch.randn(10, 4, requires_grad=True)
    soft = gumbel_softmax_sample(logits, temperature=0.5, hard=False)
    soft.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gumbel_hard_straight_through_grad():
    """hard=True still passes a finite (straight-through) gradient to logits."""
    torch.manual_seed(19)
    logits = torch.randn(10, 4, requires_grad=True)
    hard = gumbel_softmax_sample(logits, temperature=0.5, hard=True)
    (hard * torch.arange(4.0)).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# Edge cases / no-NaN under extremes
# ---------------------------------------------------------------------------

def test_lognormal_kl_tiny_sigma_no_nan():
    """Tiny sigma (below the 1e-6 floor) must not produce NaN/Inf."""
    mu_q = torch.zeros(8)
    sig_q = torch.zeros(8)  # clamped to 1e-6 internally
    mu_p = torch.zeros(8)
    sig_p = torch.ones(8)
    out = lognormal_kl(mu_q, sig_q, mu_p, sig_p)
    assert torch.isfinite(out).all()


def test_categorical_kl_extreme_logits_no_nan():
    """Large logits do not overflow log_softmax."""
    lq = torch.tensor([[1e4, -1e4, 0.0]])
    lp = torch.tensor([[0.0, 0.0, 0.0]])
    out = categorical_kl(lq, lp)
    assert torch.isfinite(out).all()
    assert (out >= -1e-6).all()


def test_gumbel_extreme_temperature_no_nan():
    """Very small/large temperatures still give finite valid simplex rows."""
    torch.manual_seed(20)
    logits = torch.randn(128, 5)
    for tau in (0.05, 100.0):
        soft = gumbel_softmax_sample(logits, temperature=tau, hard=False)
        assert torch.isfinite(soft).all(), f"NaN at tau={tau}"
        assert torch.allclose(soft.sum(dim=-1), torch.ones(128), atol=1e-4)
        assert (soft >= 0).all()
