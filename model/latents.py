"""The three bar-pointer latent variables, their sampling, and their KL divergences.

Per frame the model carries three latents:
  * meter        m         ~ Categorical   (which beats-per-bar hypothesis)
  * bar phase    phi       ~ von Mises      (angular position within the bar, wraps every bar)
  * log tempo    log_tempo ~ Normal         (log of the bar-phase advance per frame; LogNormal on tempo)

This module is pure math (sampling + KL); the generative/inference dynamics live in bar_pointer_vae.py.
The von Mises KL uses exponentially-scaled modified Bessel functions (i0e/i1e) for numerical stability
at large concentration.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# von Mises (bar phase)
# ---------------------------------------------------------------------------

def sample_von_mises(mean_angle: torch.Tensor, concentration: torch.Tensor) -> torch.Tensor:
    """Best-Fisher rejection-free reparameterized-ish sampler for the von Mises distribution.

    Returns an angle in [0, 2*pi). Uses the standard Best & Fisher (1979) acceptance scheme; the small
    fixed iteration count is adequate for the concentrations seen here and keeps the op batched/cheap.
    """
    concentration = concentration.clamp(min=1e-4)
    tau = 1.0 + torch.sqrt(1.0 + 4.0 * concentration ** 2)
    rho = (tau - torch.sqrt(2.0 * tau)) / (2.0 * concentration)
    r = (1.0 + rho ** 2) / (2.0 * rho)
    angle = torch.zeros_like(mean_angle)
    accepted = torch.zeros_like(mean_angle, dtype=torch.bool)
    for _ in range(8):  # a few proposal rounds; near-certain acceptance for our concentration range
        uniform1 = torch.rand_like(mean_angle)
        uniform2 = torch.rand_like(mean_angle)
        uniform3 = torch.rand_like(mean_angle)
        z = torch.cos(math.pi * uniform1)
        f = (1.0 + r * z) / (r + z)
        c = concentration * (r - f)
        accept_now = (c * (2.0 - c) - uniform2 > 0) | (torch.log(c / uniform2 + 1e-12) + 1.0 - c >= 0)
        proposal = torch.sign(uniform3 - 0.5) * torch.acos(f.clamp(-1.0, 1.0))
        newly = accept_now & (~accepted)
        angle = torch.where(newly, proposal, angle)
        accepted = accepted | accept_now
    # Straight-through: the sampled offset is treated as detached noise so gradients flow only through the
    # mean. This avoids backprop through the rejection sampler's acos (whose gradient is infinite at +/-1
    # and otherwise produces NaNs); the concentration still receives its gradient via the KL term.
    return (mean_angle + angle.detach()) % TWO_PI


def kl_von_mises(mean_q: torch.Tensor, concentration_q: torch.Tensor,
                 mean_p: torch.Tensor, concentration_p: torch.Tensor) -> torch.Tensor:
    """KL( vM(mean_q, conc_q) || vM(mean_p, conc_p) ), summed over the batch's last dim if any."""
    concentration_q = concentration_q.clamp(min=1e-4)
    concentration_p = concentration_p.clamp(min=1e-4)
    # A(k) = I1(k)/I0(k); with scaled Bessel i1e/i0e the exp(k) factors cancel.
    bessel_ratio_q = torch.special.i1e(concentration_q) / torch.special.i0e(concentration_q)
    # log( I0(conc_p) / I0(conc_q) ) = (conc_p - conc_q) + log( i0e(conc_p) / i0e(conc_q) )
    log_normalizer_ratio = (concentration_p - concentration_q) + torch.log(
        torch.special.i0e(concentration_p) / torch.special.i0e(concentration_q)
    )
    return log_normalizer_ratio + bessel_ratio_q * (
        concentration_q - concentration_p * torch.cos(mean_q - mean_p)
    )


# ---------------------------------------------------------------------------
# Normal (log tempo)
# ---------------------------------------------------------------------------

def sample_normal(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Reparameterized Normal sample."""
    return mean + std * torch.randn_like(mean)


def kl_normal(mean_q: torch.Tensor, std_q: torch.Tensor,
              mean_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
    """KL( Normal(mean_q, std_q) || Normal(mean_p, std_p) )."""
    std_q = std_q.clamp(min=1e-6)
    std_p = std_p.clamp(min=1e-6)
    return (
        torch.log(std_p / std_q)
        + (std_q ** 2 + (mean_q - mean_p) ** 2) / (2.0 * std_p ** 2)
        - 0.5
    )


# ---------------------------------------------------------------------------
# Categorical (meter)
# ---------------------------------------------------------------------------

def sample_gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Differentiable Categorical relaxation (Gumbel-softmax). Returns a soft one-hot [..., num_meters]."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    return F.softmax((logits + gumbel_noise) / temperature, dim=-1)


def kl_categorical(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """KL( Categorical(log_q) || Categorical(log_p) ), given log-probabilities. Summed over categories."""
    return (log_q.exp() * (log_q - log_p)).sum(dim=-1)
