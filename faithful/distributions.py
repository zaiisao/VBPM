"""Distributions, samplers and closed-form KLs for the faithful bar-pointer VAE.

Every function here is a direct, *batched* port of ``notebooks/build_elbo_notebook.py``
(sections 1-3), which is itself a line-by-line transcription of *ELBO for DBN* §5.1-5.3
and Algorithm 2. The functions are shape-generic: they operate elementwise over
arbitrary leading dimensions, so the same code serves the toy notebook ``[]`` /
single-sequence case and the batched ``[B]`` / ``[B, T]`` training case.

Numerical safety here is limited to what the *math* requires (exponentially-scaled
Bessel functions; a Bessel-overflow guard inside the von Mises sampler). There are
NO structural clamps on kappa, sigma or log-tempo -- those would be bandages.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Bessel helpers and the (zero-mean) von Mises density / CDF  (notebook §1)
# ---------------------------------------------------------------------------
def log_i0(kappa: torch.Tensor) -> torch.Tensor:
    """log I0(kappa), stable via the exponentially-scaled Bessel i0e."""
    return torch.log(torch.special.i0e(kappa)) + kappa


def A_kappa(kappa: torch.Tensor) -> torch.Tensor:
    """Mean resultant length A(kappa) = I1(kappa) / I0(kappa)."""
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)


def von_mises_pdf(z: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Density of a ZERO-MEAN von Mises at angle z."""
    return torch.exp(kappa * (torch.cos(z) - 1.0)) / (TWO_PI * torch.special.i0e(kappa))


def von_mises_cdf(z: torch.Tensor, kappa: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
    """F(z | 0, kappa) = \\int_{-pi}^{z} p(t | 0, kappa) dt, via the trapezoid rule.

    Works for z, kappa of the same (arbitrary) shape; integrates along a new last axis.
    """
    lower = -math.pi
    frac = torch.linspace(0.0, 1.0, n_steps, device=z.device, dtype=z.dtype)  # [n]
    z_e = z.unsqueeze(-1)
    k_e = kappa.unsqueeze(-1)
    t = lower + frac * (z_e - lower)                                  # integration grid
    pdf = torch.exp(k_e * (torch.cos(t) - 1.0)) / (TWO_PI * torch.special.i0e(k_e))
    weights = torch.ones_like(pdf)
    weights[..., 0] = 0.5
    weights[..., -1] = 0.5
    step = (z_e - lower) / (n_steps - 1)
    return (pdf * weights).sum(-1) * step.squeeze(-1)


# ---------------------------------------------------------------------------
# Algorithm 2: von Mises Best-Fisher sampler + implicit reparameterisation
# ---------------------------------------------------------------------------
def best_fisher_rejection(kappa: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
    """Algorithm 2, forward pass (lines 1-16). Samples z ~ vM(0, kappa) elementwise."""
    shape = kappa.shape
    k = kappa.reshape(-1).clamp(min=1e-3)

    tau = 1.0 + torch.sqrt(1.0 + 4.0 * k * k)        # line 1
    rho = (tau - torch.sqrt(2.0 * tau)) / (2.0 * k)  # line 2
    r = (1.0 + rho * rho) / (2.0 * rho)              # line 3

    z = torch.zeros_like(k)
    accepted = torch.zeros_like(k, dtype=torch.bool)
    for _ in range(max_iter):
        u1 = torch.rand_like(k)
        u2 = torch.rand_like(k)
        u3 = torch.rand_like(k)
        c = torch.cos(math.pi * u1)                  # line 6
        f = (1.0 + r * c) / (r + c)                  # line 7
        accept = (k * (r - f) + torch.log(f) - torch.log(r)) >= torch.log(u2)   # line 9
        sign = torch.where(u3 > 0.5, 1.0, -1.0)      # lines 11-15
        angle = sign * torch.acos(torch.clamp(f, -1.0, 1.0))
        newly = accept & (~accepted)
        z = torch.where(newly, angle, z)
        accepted = accepted | accept
        if bool(accepted.all()):
            break
    return z.reshape(shape)


class VonMisesSample(torch.autograd.Function):
    """Algorithm 2 as an autograd Function: forward = rejection sampler,
    backward = implicit reparameterisation gradients (Figurnov et al. 2018)."""

    @staticmethod
    def forward(ctx, mu, kappa):
        z = best_fisher_rejection(kappa)             # z ~ vM(0, kappa)
        phi = mu + z                                 # line 16
        ctx.save_for_backward(z, kappa)
        return phi

    @staticmethod
    def backward(ctx, grad_phi):
        z, kappa = ctx.saved_tensors
        # dF(z|kappa)/dkappa via autograd on the CDF (line 20)
        with torch.enable_grad():
            k = kappa.detach().clone().requires_grad_(True)
            cdf = von_mises_cdf(z.detach(), k)
            (dF_dk,) = torch.autograd.grad(cdf.sum(), k)
        p = von_mises_pdf(z, kappa)                  # line 19
        dphi_dkappa = -dF_dk / (p + 1e-12)           # line 21
        grad_mu = grad_phi * 1.0                     # line 22 (dphi/dmu = 1)
        grad_kappa = grad_phi * dphi_dkappa          # line 23
        return grad_mu, grad_kappa


def sample_von_mises(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    return VonMisesSample.apply(mu, kappa)


# ---------------------------------------------------------------------------
# Gumbel-Softmax relaxation of the Categorical meter  (notebook §4, paper §5.1)
# ---------------------------------------------------------------------------
def gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    return F.softmax((logits + g) / temperature, dim=-1)


# ---------------------------------------------------------------------------
# Closed-form KL divergences  (paper §5.1-5.3, notebook §3)
# ---------------------------------------------------------------------------
def kl_categorical(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """KL( Cat(q) || Cat(p) ); inputs are LOG-probabilities, summed over the K classes."""
    q = log_q.exp()
    return (q * (log_q - log_p)).sum(-1)


def kl_von_mises(mu_q, kappa_q, mu_p, kappa_p) -> torch.Tensor:
    return (log_i0(kappa_p) - log_i0(kappa_q)
            + A_kappa(kappa_q) * (kappa_q - kappa_p * torch.cos(mu_q - mu_p)))


def kl_log_normal(mu_q, sigma_q, mu_p, sigma_p) -> torch.Tensor:
    """Log-Normal KL reduces to the Gaussian KL in log-space (§5.3)."""
    return (torch.log(sigma_p / sigma_q)
            + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2) - 0.5)
