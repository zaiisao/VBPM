"""Unit tests for von Mises utilities in models/distributions.py.

Ground truth sources:
- scipy.special.i0 / i1            -> resultant length A(k)=I1(k)/I0(k)
- numpy.random.vonmises            -> independent reference sampler
- scipy.stats.vonmises.cdf         -> CDF ground truth
- Monte-Carlo estimate of E_q[log q - log p] -> KL ground truth
- finite-difference / analytic A'(k) -> backward gradient ground truth

All tests run on CPU with small tensors.
"""

import sys
sys.path.insert(0, "/home/sogang/jaehoon/CHART")

import math

import numpy as np
import pytest
import torch
from scipy import special as sp_special
from scipy import stats as sp_stats

from models.distributions import (
    von_mises_sample,
    von_mises_kl,
    _von_mises_cdf_series,
    _von_mises_cdf_normal,
    _A,
    _log_i0,
)

torch.manual_seed(0)
np.random.seed(0)
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Helper: circular statistics
# ---------------------------------------------------------------------------
def circular_mean(samples: np.ndarray) -> float:
    """Mean angle of circular samples."""
    return float(np.angle(np.mean(np.exp(1j * samples))))


def resultant_length(samples: np.ndarray) -> float:
    """Mean resultant length R of circular samples."""
    return float(np.abs(np.mean(np.exp(1j * samples))))


def analytic_A(kappa: float) -> float:
    """A(k) = I1(k)/I0(k) via scipy Bessel functions (ground truth)."""
    return float(sp_special.i1(kappa) / sp_special.i0(kappa))


# ---------------------------------------------------------------------------
# FORWARD: resultant length R(k) ~ I1(k)/I0(k)
# This is the key regression test for the kappa-ignoring acceptance bug.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.05, 0.3, 1.0, 3.0, 8.0, 20.0])
def test_forward_resultant_length_matches_bessel(kappa):
    N = 200_000
    mu = torch.zeros(N, dtype=DTYPE)
    k = torch.full((N,), kappa, dtype=DTYPE)
    samples = von_mises_sample(mu, k).detach().numpy()

    assert np.all(np.isfinite(samples)), "non-finite samples"

    R_emp = resultant_length(samples)
    R_true = analytic_A(kappa)

    # MC std error of R is ~ 1/sqrt(N); allow a generous absolute tolerance.
    tol = 0.012 + 3.0 / math.sqrt(N)
    assert abs(R_emp - R_true) < tol, (
        f"kappa={kappa}: empirical R={R_emp:.4f} vs Bessel A={R_true:.4f} "
        f"(diff {abs(R_emp - R_true):.4f} > tol {tol:.4f}). "
        "Resultant length should track concentration."
    )


# ---------------------------------------------------------------------------
# FORWARD: circular mean ~ mu (test a nonzero mu so it is not trivial)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.3, 1.0, 3.0, 8.0])
@pytest.mark.parametrize("mu_val", [0.0, 1.0, -2.0])
def test_forward_circular_mean_matches_mu(kappa, mu_val):
    N = 200_000
    mu = torch.full((N,), mu_val, dtype=DTYPE)
    k = torch.full((N,), kappa, dtype=DTYPE)
    samples = von_mises_sample(mu, k).detach().numpy()

    mean_emp = circular_mean(samples)
    # wrap difference to (-pi, pi]
    diff = math.atan2(math.sin(mean_emp - mu_val), math.cos(mean_emp - mu_val))

    # circular SE ~ 1/sqrt(N*R); looser for small kappa.
    R_true = analytic_A(kappa)
    se = 1.0 / math.sqrt(N * max(R_true, 0.05))
    tol = 0.02 + 6.0 * se
    assert abs(diff) < tol, (
        f"kappa={kappa}, mu={mu_val}: circular mean {mean_emp:.4f} "
        f"off by {abs(diff):.4f} (tol {tol:.4f})"
    )


# ---------------------------------------------------------------------------
# FORWARD vs numpy.random.vonmises: distribution match via histogram / KS-like
# Compare empirical cos & sin moments (E[cos]=A, E[sin]=0) cross-checked.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.5, 2.0, 5.0])
def test_forward_moments_match_numpy_reference(kappa):
    N = 300_000
    mu_val = 0.7
    mu = torch.full((N,), mu_val, dtype=DTYPE)
    k = torch.full((N,), kappa, dtype=DTYPE)
    ours = von_mises_sample(mu, k).detach().numpy()
    ref = np.random.vonmises(mu_val, kappa, size=N)

    # Compare E[cos(z)] and E[sin(z)] between our sampler and numpy's.
    for fn in (np.cos, np.sin):
        m_ours = float(np.mean(fn(ours)))
        m_ref = float(np.mean(fn(ref)))
        se = 2.0 / math.sqrt(N)
        assert abs(m_ours - m_ref) < 6.0 * se + 0.01, (
            f"kappa={kappa} {fn.__name__}: ours {m_ours:.4f} vs numpy {m_ref:.4f}"
        )


# ---------------------------------------------------------------------------
# FORWARD: samples lie on the circle (finite), mu shifts samples rigidly.
# ---------------------------------------------------------------------------
def test_forward_mu_translation_invariance():
    # With identical RNG state, sampling at mu and mu+delta should differ by delta
    # exactly, because the algorithm computes z independently of mu then adds mu.
    N = 5000
    k = torch.full((N,), 2.0, dtype=DTYPE)

    torch.manual_seed(1234)
    s0 = von_mises_sample(torch.zeros(N, dtype=DTYPE), k).detach()
    torch.manual_seed(1234)
    delta = 0.9
    s1 = von_mises_sample(torch.full((N,), delta, dtype=DTYPE), k).detach()

    assert torch.allclose(s1 - s0, torch.full((N,), delta, dtype=DTYPE), atol=1e-9), (
        "mu must shift samples rigidly (sample = mu + z)"
    )


# ---------------------------------------------------------------------------
# BACKWARD: dz/dmu == 1  (closed form)
# ---------------------------------------------------------------------------
def test_backward_dz_dmu_is_one():
    N = 2000
    mu = torch.zeros(N, dtype=DTYPE, requires_grad=True)
    k = torch.full((N,), 3.0, dtype=DTYPE)
    s = von_mises_sample(mu, k)
    s.sum().backward()
    assert torch.allclose(mu.grad, torch.ones(N, dtype=DTYPE), atol=1e-9), (
        f"dz/dmu should be exactly 1, got mean {mu.grad.mean().item()}"
    )


# ---------------------------------------------------------------------------
# BACKWARD: autograd d/dk E[cos z] vs analytic A'(k) = 1 - A/k - A^2
# This is the implicit-reparam backward correctness test.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.3, 1.0, 3.0, 8.0])
def test_backward_dEcos_dkappa_matches_analytic(kappa):
    # E_vM[cos z] = A(k). d/dk A(k) = 1 - A(k)/k - A(k)^2.
    # Reparameterized estimator: grad of mean(cos(sample)) w.r.t kappa -> A'(k).
    N = 400_000
    mu = torch.zeros(N, dtype=DTYPE)
    k = torch.full((N,), kappa, dtype=DTYPE, requires_grad=True)
    s = von_mises_sample(mu, k)
    obj = torch.cos(s).mean()
    (grad_k,) = torch.autograd.grad(obj, k)
    g_emp = float(grad_k.sum())  # sum over the broadcast since each element 1/N

    A = analytic_A(kappa)
    Aprime = 1.0 - A / kappa - A * A

    # MC noise in the gradient is larger than in the mean; allow generous tol.
    tol = 0.03 + 0.05 * abs(Aprime)
    assert abs(g_emp - Aprime) < tol, (
        f"kappa={kappa}: autograd dE[cos]/dk={g_emp:.4f} vs analytic "
        f"A'(k)={Aprime:.4f} (diff {abs(g_emp - Aprime):.4f} > tol {tol:.4f})"
    )


# ---------------------------------------------------------------------------
# BACKWARD: gradients are finite (no NaN/Inf) across kappa incl. extremes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [1e-4, 0.01, 0.5, 10.5, 50.0, 300.0, 700.0])
def test_backward_gradients_finite(kappa):
    N = 1000
    mu = torch.zeros(N, dtype=DTYPE, requires_grad=True)
    k = torch.full((N,), kappa, dtype=DTYPE, requires_grad=True)
    s = von_mises_sample(mu, k)
    (torch.cos(s).sum() + torch.sin(s).sum()).backward()
    assert torch.all(torch.isfinite(mu.grad)), f"mu.grad non-finite at kappa={kappa}"
    assert torch.all(torch.isfinite(k.grad)), f"kappa.grad non-finite at kappa={kappa}"


# ---------------------------------------------------------------------------
# KL: analytic von_mises_kl vs Monte-Carlo E_q[log q - log p].
# ---------------------------------------------------------------------------
def _vm_logpdf_np(z, mu, kappa):
    """log von Mises pdf (numpy), normalized via scipy i0."""
    return kappa * np.cos(z - mu) - np.log(2.0 * np.pi * sp_special.i0(kappa))


@pytest.mark.parametrize(
    "mu_q,kq,mu_p,kp",
    [
        (0.0, 2.0, 0.0, 1.0),
        (0.5, 3.0, -0.4, 2.0),
        (1.2, 5.0, 0.3, 1.5),
        (0.0, 0.5, 0.0, 0.3),
        (-1.0, 4.0, 1.0, 4.0),
    ],
)
def test_kl_matches_monte_carlo(mu_q, kq, mu_p, kp):
    # Analytic KL from the implementation.
    kl_analytic = float(
        von_mises_kl(
            torch.tensor(mu_q, dtype=DTYPE),
            torch.tensor(kq, dtype=DTYPE),
            torch.tensor(mu_p, dtype=DTYPE),
            torch.tensor(kp, dtype=DTYPE),
        )
    )
    # MC ground truth: draw from q via numpy reference, average log q - log p.
    N = 1_000_000
    zq = np.random.vonmises(mu_q, kq, size=N)
    lq = _vm_logpdf_np(zq, mu_q, kq)
    lp = _vm_logpdf_np(zq, mu_p, kp)
    kl_mc = float(np.mean(lq - lp))
    se = float(np.std(lq - lp) / math.sqrt(N))

    assert kl_analytic >= -1e-9, f"KL must be >= 0, got {kl_analytic}"
    assert abs(kl_analytic - kl_mc) < 6.0 * se + 0.01, (
        f"analytic KL {kl_analytic:.4f} vs MC {kl_mc:.4f} (6SE={6*se:.4f})"
    )


def test_kl_zero_for_identical_distributions():
    mu = torch.tensor([0.0, 1.0, -2.0], dtype=DTYPE)
    k = torch.tensor([0.5, 3.0, 10.0], dtype=DTYPE)
    kl = von_mises_kl(mu, k, mu.clone(), k.clone())
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-10), (
        f"KL(p||p) must be 0, got {kl}"
    )


def test_kl_nonnegative_random():
    # Gibbs: KL >= 0 for arbitrary parameter pairs.
    g = torch.Generator().manual_seed(7)
    mu_q = (torch.rand(64, generator=g, dtype=DTYPE) - 0.5) * 2 * math.pi
    mu_p = (torch.rand(64, generator=g, dtype=DTYPE) - 0.5) * 2 * math.pi
    kq = torch.rand(64, generator=g, dtype=DTYPE) * 20 + 0.01
    kp = torch.rand(64, generator=g, dtype=DTYPE) * 20 + 0.01
    kl = von_mises_kl(mu_q, kq, mu_p, kp)
    assert torch.all(kl >= -1e-8), f"KL negativity: min {kl.min().item()}"
    assert torch.all(torch.isfinite(kl))


# ---------------------------------------------------------------------------
# CDF series vs scipy.stats.vonmises.cdf  (kappa < 10.5 regime)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.1, 0.5, 1.0, 3.0, 8.0, 10.0])
def test_cdf_series_matches_scipy(kappa):
    # scipy vonmises.cdf is defined on (-pi, pi) centered at 0; the
    # implementation centers at mu=0 and returns CDF measured from -pi.
    z = torch.linspace(-math.pi + 1e-3, math.pi - 1e-3, 50, dtype=DTYPE)
    k = torch.full_like(z, kappa)
    cdf, _ = _von_mises_cdf_series(z, k)
    cdf = cdf.detach().numpy()
    ref = sp_stats.vonmises.cdf(z.numpy(), kappa)
    assert np.all(np.isfinite(cdf))
    assert np.max(np.abs(cdf - ref)) < 1e-4, (
        f"kappa={kappa}: max |series-scipy| = {np.max(np.abs(cdf - ref)):.2e}"
    )


# ---------------------------------------------------------------------------
# CDF normal-approx vs scipy.stats.vonmises.cdf  (kappa >= 10.5 regime)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [11.0, 20.0, 50.0, 200.0])
def test_cdf_normal_matches_scipy(kappa):
    # Normal approximation is accurate near the bulk; test over +-3 sigma.
    sigma = 1.0 / math.sqrt(kappa)
    half = min(math.pi - 1e-3, 4.0 * sigma)
    z = torch.linspace(-half, half, 50, dtype=DTYPE)
    k = torch.full_like(z, kappa)
    cdf = _von_mises_cdf_normal(z, k).detach().numpy()
    ref = sp_stats.vonmises.cdf(z.numpy(), kappa)
    assert np.all(np.isfinite(cdf))
    assert np.max(np.abs(cdf - ref)) < 1e-3, (
        f"kappa={kappa}: max |normal-scipy| = {np.max(np.abs(cdf - ref)):.2e}"
    )


# ---------------------------------------------------------------------------
# CDF series dcdf/dkappa vs finite difference (forward-mode tangent check)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.5, 2.0, 6.0])
def test_cdf_series_dkappa_matches_finite_difference(kappa):
    z = torch.linspace(-2.0, 2.0, 30, dtype=DTYPE)
    h = 1e-6
    k0 = torch.full_like(z, kappa)
    cdf_plus, _ = _von_mises_cdf_series(z, k0 + h)
    cdf_minus, _ = _von_mises_cdf_series(z, k0 - h)
    fd = ((cdf_plus - cdf_minus) / (2 * h)).detach().numpy()
    _, dcdf = _von_mises_cdf_series(z, k0)
    dcdf = dcdf.detach().numpy()
    assert np.max(np.abs(dcdf - fd)) < 1e-4, (
        f"kappa={kappa}: analytic dcdf/dk vs FD max diff "
        f"{np.max(np.abs(dcdf - fd)):.2e}"
    )


# ---------------------------------------------------------------------------
# CDF monotonicity and range invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.2, 2.0, 9.0])
def test_cdf_series_monotone_and_bounded(kappa):
    z = torch.linspace(-math.pi + 1e-3, math.pi - 1e-3, 200, dtype=DTYPE)
    k = torch.full_like(z, kappa)
    cdf, _ = _von_mises_cdf_series(z, k)
    cdf = cdf.detach()
    assert torch.all(cdf >= 0.0) and torch.all(cdf <= 1.0)
    diffs = cdf[1:] - cdf[:-1]
    assert torch.all(diffs >= -1e-9), "CDF must be non-decreasing in z"


# ---------------------------------------------------------------------------
# Bessel helper sanity: _A(k) and _log_i0(k) vs scipy (these underpin the KL).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [0.01, 0.5, 1.0, 5.0, 20.0, 100.0])
def test_A_and_logi0_match_scipy(kappa):
    k = torch.tensor(kappa, dtype=DTYPE)
    A_ours = float(_A(k))
    A_ref = analytic_A(kappa)
    assert abs(A_ours - A_ref) < 1e-9, f"_A({kappa}) {A_ours} vs {A_ref}"

    logi0_ours = float(_log_i0(k))
    logi0_ref = float(np.log(sp_special.i0(kappa)))
    assert abs(logi0_ours - logi0_ref) < 1e-7, (
        f"_log_i0({kappa}) {logi0_ours} vs {logi0_ref}"
    )


# ---------------------------------------------------------------------------
# No-NaN at extreme kappa in forward sampler.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [1e-6, 1e-3, 700.0, 1000.0])
def test_forward_no_nan_extreme_kappa(kappa):
    N = 5000
    mu = torch.full((N,), 0.3, dtype=DTYPE)
    k = torch.full((N,), kappa, dtype=DTYPE)
    s = von_mises_sample(mu, k).detach()
    assert torch.all(torch.isfinite(s)), f"non-finite samples at kappa={kappa}"
    # All samples within [mu-pi, mu+pi].
    assert torch.all((s - 0.3).abs() <= math.pi + 1e-6)


# ---------------------------------------------------------------------------
# Large-kappa concentration: samples cluster tightly around mu.
# ---------------------------------------------------------------------------
def test_large_kappa_concentrates():
    N = 50_000
    mu_val = 1.0
    for kappa in (50.0, 200.0):
        mu = torch.full((N,), mu_val, dtype=DTYPE)
        k = torch.full((N,), kappa, dtype=DTYPE)
        s = von_mises_sample(mu, k).detach().numpy()
        # vM variance ~ 1/kappa for large kappa. Circular variance 1 - A(k).
        circ_var = 1.0 - resultant_length(s - mu_val + mu_val)  # use raw
        circ_var = 1.0 - resultant_length(s)
        expected = 1.0 - analytic_A(kappa)
        assert abs(circ_var - expected) < 0.01, (
            f"kappa={kappa}: circ var {circ_var:.4f} vs {expected:.4f}"
        )
