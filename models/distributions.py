"""Distribution utilities for the variational bar pointer model.

Provides closed-form KL divergences and reparameterized samplers for:
- Categorical (meter) with Gumbel-Softmax relaxation
- von Mises (phase) with Best-Fisher rejection sampling + implicit reparam
- Log-Normal (tempo) via Gaussian reparameterization in log-space

The von Mises backward pass follows the implicit reparameterization gradient
formulation from Figurnov et al. (2018), ported from the TensorFlow Probability
reference implementation (Hill 1977, Algorithm 518).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_KAPPA_MAX = 700.0  # clamp to avoid Bessel overflow
_KAPPA_MIN = 1e-6
TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Bessel helpers (numerically stable via exponentially-scaled versions)
# ---------------------------------------------------------------------------

def _log_i0(kappa: Tensor) -> Tensor:
    """Compute log I_0(kappa) in a numerically stable way."""
    return torch.log(torch.special.i0e(kappa)) + kappa.abs()


def _A(kappa: Tensor) -> Tensor:
    """Mean resultant length A(kappa) = I_1(kappa) / I_0(kappa)."""
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)


def _cosxm1(x: Tensor) -> Tensor:
    """Compute cos(x) - 1 in a numerically stable way: -2 * sin^2(x/2).

    Avoids catastrophic cancellation when x is near 0.
    """
    return -2.0 * torch.sin(x / 2.0) ** 2


# ---------------------------------------------------------------------------
# Categorical (meter)
# ---------------------------------------------------------------------------

def categorical_kl(logits_q: Tensor, logits_p: Tensor) -> Tensor:
    """KL(Cat(q) || Cat(p)) from unnormalized logits.

    Args:
        logits_q: Posterior logits, shape ``(..., K)``.
        logits_p: Prior logits, shape ``(..., K)``.

    Returns:
        KL divergence per element, shape ``(...)``.
    """
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    q = log_q.exp()
    return (q * (log_q - log_p)).sum(dim=-1)


def gumbel_softmax_sample(
    logits: Tensor,
    temperature: float,
    hard: bool = False,
) -> Tensor:
    """Sample from Gumbel-Softmax relaxation.

    Args:
        logits: Unnormalized log-probabilities, shape ``(..., K)``.
        temperature: Softmax temperature (tau > 0).
        hard: If True, use straight-through estimator.

    Returns:
        Soft (or hard) sample, shape ``(..., K)``.
    """
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


# ---------------------------------------------------------------------------
# Von Mises (phase) — CDF helpers for implicit reparameterization backward
# ---------------------------------------------------------------------------

def _von_mises_cdf_series(
    z: Tensor, kappa: Tensor, n_terms: int = 20,
) -> tuple[Tensor, Tensor]:
    """Von Mises CDF and dcdf/dkappa via Fourier series with forward-mode tangent.

    Implements Hill (1977) Algorithm 518, adapted from TensorFlow Probability.
    Accurate for kappa < ~10.5.

    Args:
        z: Sample point (centered at mu=0), shape ``(...)``.
        kappa: Concentration parameter, shape ``(...)``.
        n_terms: Number of Fourier terms (default 20).

    Returns:
        Tuple of (cdf, dcdf_dkappa), each shape ``(...)``.
    """
    # Backward recurrence from n=n_terms down to n=1
    rn = torch.zeros_like(kappa)
    drn = torch.zeros_like(kappa)
    vn = torch.zeros_like(kappa)
    dvn = torch.zeros_like(kappa)

    for i in range(n_terms, 0, -1):
        n = float(i)
        denominator = 2.0 * n / kappa + rn
        ddenominator = -2.0 * n / (kappa ** 2) + drn
        rn = 1.0 / denominator
        drn = -ddenominator / (denominator ** 2)

        multiplier = torch.sin(n * z) / n + vn
        vn = rn * multiplier
        dvn = drn * multiplier + rn * dvn

    cdf = 0.5 + z / TWO_PI + vn / math.pi
    dcdf = dvn / math.pi

    # Clip CDF to [0, 1] and zero gradient where clipped (TFP convention)
    valid = (cdf >= 0.0) & (cdf <= 1.0)
    cdf = cdf.clamp(0.0, 1.0)
    dcdf = dcdf * valid.float()

    return cdf, dcdf


def _von_mises_cdf_normal(z: Tensor, kappa: Tensor) -> Tensor:
    """Von Mises CDF via corrected normal approximation (Hill 1977).

    Accurate for kappa >= ~10.5. Adapted from TensorFlow Probability.
    Must be called under torch.enable_grad() if gradient w.r.t. kappa is needed.

    Args:
        z: Sample point (centered at mu=0), shape ``(...)``.
        kappa: Concentration parameter, shape ``(...)``.

    Returns:
        CDF values, shape ``(...)``.
    """
    s = math.sqrt(2.0 / math.pi) / torch.special.i0e(kappa) * torch.sin(0.5 * z)
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    c = 24.0 * kappa
    c1 = 56.0

    xi = s - s3 / (
        (c - 2.0 * s2 - 16.0) / 3.0
        - (s4 + 1.75 * s2 + 83.5) / (c - c1 - s2 + 3.0)
    ) ** 2

    return 0.5 * (1.0 + torch.erf(xi * math.sqrt(0.5)))


def _inv_prob(z: Tensor, kappa: Tensor) -> Tensor:
    """Compute 1/p(z|0, kappa) stably using exponentially-scaled Bessel.

    Identity: 1/p = 2*pi * I_0(kappa) * exp(-kappa*cos(z))
            = 2*pi * i0e(kappa) * exp(kappa) * exp(-kappa*cos(z))
            = 2*pi * i0e(kappa) * exp(-kappa*(cos(z)-1))
            = 2*pi * i0e(kappa) * exp(-kappa*cosxm1(z))

    The exp(kappa) terms cancel, preventing overflow.
    """
    return torch.exp(-kappa * _cosxm1(z)) * TWO_PI * torch.special.i0e(kappa)


# ---------------------------------------------------------------------------
# Von Mises sampler with implicit reparameterization
# ---------------------------------------------------------------------------

class _VonMisesSampleFn(torch.autograd.Function):
    """Von Mises sampler with implicit reparameterization gradients.

    Forward: Best-Fisher rejection algorithm.
    Backward: implicit reparameterization (Figurnov et al. 2018),
        ported from TensorFlow Probability.
        dz/dmu = 1
        dz/dkappa = -dF(z|kappa)/dkappa / p(z|0,kappa)
    """

    @staticmethod
    def forward(ctx, mu: Tensor, kappa: Tensor) -> Tensor:  # type: ignore[override]
        kappa_safe = kappa.clamp(min=_KAPPA_MIN, max=_KAPPA_MAX)

        # Best-Fisher rejection algorithm setup.
        # For small kappa, the exact rho computation is numerically unstable
        # (tau - sqrt(2*tau) ≈ 0, causing 0/0). Use the approximation from TFP.
        _SMALL_CUTOFF = 0.02
        tau = 1.0 + torch.sqrt(1.0 + 4.0 * kappa_safe ** 2)
        rho = (tau - torch.sqrt(2.0 * tau)) / (2.0 * kappa_safe)
        r_exact = (1.0 + rho ** 2) / (2.0 * rho)
        r_approx = 1.0 / kappa_safe + kappa_safe
        r = torch.where(kappa_safe < _SMALL_CUTOFF, r_approx, r_exact)

        shape = mu.shape
        device = mu.device
        dtype = mu.dtype

        done = torch.zeros(shape, dtype=torch.bool, device=device)
        f_accepted = torch.zeros(shape, dtype=dtype, device=device)
        max_iter = 1000

        for _ in range(max_iter):
            if done.all():
                break
            u1 = torch.rand(shape, dtype=dtype, device=device)
            u2 = torch.rand(shape, dtype=dtype, device=device)
            c = torch.cos(math.pi * u1)
            f = (1.0 + r * c) / (r + c)
            # Guard log(f) for f <= 0 (can happen when r is large / kappa small)
            f_pos = f.clamp(min=1e-30)
            accept = (kappa_safe * (r - f) + torch.log(f_pos) - torch.log(r)) >= torch.log(u2)
            accept = accept & (f > 0)  # reject negative f
            newly_accepted = accept & ~done
            f_accepted = torch.where(newly_accepted, f, f_accepted)
            done = done | accept

        # For elements that were never accepted (very small kappa → near-uniform),
        # sample uniformly on [-pi, pi] via acos(uniform(-1,1)).
        uniform_f = 2.0 * torch.rand(shape, dtype=dtype, device=device) - 1.0
        f_accepted = torch.where(done, f_accepted, uniform_f)
        u3 = torch.rand(shape, dtype=dtype, device=device)
        z = torch.where(u3 > 0.5, torch.acos(f_accepted), -torch.acos(f_accepted))
        sample = mu + z

        ctx.save_for_backward(z, kappa_safe)
        return sample

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        z, kappa = ctx.saved_tensors

        # dz/dmu = 1
        grad_mu = grad_output

        # --- Compute dcdf/dkappa via two branches ---
        # Series branch (kappa < 10.5): manual forward-mode tangent
        _, dcdf_series = _von_mises_cdf_series(z, kappa)

        # Normal branch (kappa >= 10.5): PyTorch autograd
        with torch.enable_grad():
            kappa_g = kappa.detach().requires_grad_(True)
            cdf_normal = _von_mises_cdf_normal(z.detach(), kappa_g)
            (dcdf_normal,) = torch.autograd.grad(
                cdf_normal, kappa_g,
                grad_outputs=torch.ones_like(cdf_normal),
                create_graph=False,
            )

        # Select branch
        small = kappa < 10.5
        dcdf_dkappa = torch.where(small, dcdf_series, dcdf_normal)

        # Stable 1/p(z|0,kappa)
        inv_p = _inv_prob(z, kappa)

        # dz/dkappa = -dF/dkappa / p(z) = -dcdf_dkappa * inv_p
        # Guard 0 * inf = nan: when CDF is saturated (dcdf=0), gradient is 0.
        # Also guard rare inf from tiny*huge in extreme tails.
        dz_dkappa = torch.where(
            dcdf_dkappa == 0,
            torch.zeros_like(dcdf_dkappa),
            -dcdf_dkappa * inv_p,
        )
        dz_dkappa = torch.nan_to_num(dz_dkappa, nan=0.0, posinf=0.0, neginf=0.0)
        grad_kappa = grad_output * dz_dkappa

        return grad_mu, grad_kappa


def von_mises_sample(mu: Tensor, kappa: Tensor) -> Tensor:
    """Draw reparameterized samples from vM(mu, kappa).

    Args:
        mu: Mean direction, shape ``(...)``.
        kappa: Concentration parameter, shape ``(...)``.

    Returns:
        Samples on the circle, shape ``(...)``.
    """
    return _VonMisesSampleFn.apply(mu, kappa)


def von_mises_kl(
    mu_q: Tensor,
    kappa_q: Tensor,
    mu_p: Tensor,
    kappa_p: Tensor,
) -> Tensor:
    """KL(vM(mu_q, kappa_q) || vM(mu_p, kappa_p)).

    Args:
        mu_q: Posterior mean direction.
        kappa_q: Posterior concentration.
        mu_p: Prior mean direction.
        kappa_p: Prior concentration.

    Returns:
        KL divergence, same shape as inputs.
    """
    kappa_q = kappa_q.clamp(min=_KAPPA_MIN, max=_KAPPA_MAX)
    kappa_p = kappa_p.clamp(min=_KAPPA_MIN, max=_KAPPA_MAX)

    return (
        _log_i0(kappa_p) - _log_i0(kappa_q)
        + _A(kappa_q) * (kappa_q - kappa_p * torch.cos(mu_q - mu_p))
    )


# ---------------------------------------------------------------------------
# Log-Normal (tempo) — implemented as Gaussian KL in log-space
# ---------------------------------------------------------------------------

def lognormal_kl(
    mu_q: Tensor,
    sigma_q: Tensor,
    mu_p: Tensor,
    sigma_p: Tensor,
) -> Tensor:
    """KL(LogN(mu_q, sigma_q^2) || LogN(mu_p, sigma_p^2)).

    All parameters are in log-space (i.e. mu and sigma of the underlying
    Gaussian), so this reduces to the standard Gaussian KL.

    Args:
        mu_q: Posterior mean in log-space.
        sigma_q: Posterior std in log-space (> 0).
        mu_p: Prior mean in log-space.
        sigma_p: Prior std in log-space (> 0).

    Returns:
        KL divergence, same shape as inputs.
    """
    # Numerical floor only: prevents log(0) / div-by-zero. Symmetric so neither
    # q nor p is implicitly biased. Behavior is fully controlled by Softplus(NN).
    sigma_q = sigma_q.clamp(min=1e-6)
    sigma_p = sigma_p.clamp(min=1e-6)

    return (
        torch.log(sigma_p / sigma_q)
        + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2)
        - 0.5
    )


def lognormal_sample(mu: Tensor, sigma: Tensor) -> Tensor:
    """Reparameterized sample from LogNormal(mu, sigma^2).

    Returns the sample in the *original* (positive) space: exp(mu + sigma * eps).

    Args:
        mu: Mean in log-space.
        sigma: Std in log-space (> 0).

    Returns:
        Positive-valued sample, same shape as inputs.
    """
    eps = torch.randn_like(mu)
    return torch.exp(mu + sigma * eps)


def lognormal_sample_logspace(mu: Tensor, sigma: Tensor) -> Tensor:
    """Reparameterized sample, returned in log-space: mu + sigma * eps.

    Args:
        mu: Mean in log-space.
        sigma: Std in log-space (> 0).

    Returns:
        Sample in log-space, same shape as inputs.
    """
    eps = torch.randn_like(mu)
    return mu + sigma * eps
