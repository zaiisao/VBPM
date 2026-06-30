"""The three bar-pointer latent variables: sampling and KL divergences.

Per frame the model carries three latents:
  * meter        m         ~ Categorical   (which beats-per-bar hypothesis)
  * bar phase    phi       ~ von Mises      (angular position within the bar, wraps every bar)
  * log tempo    log_tempo ~ Normal         (log of the bar-phase advance per frame; LogNormal on tempo)

We delegate to ``torch.distributions`` / ``torch.nn.functional`` for everything they provide -- the
Normal/Categorical KLs, Gumbel-softmax, and the von Mises *sampler* (a hand-rolled von Mises sampler bit
us before). torch's ``VonMises`` has no reparameterized gradient (``has_rsample`` is False), so we supply
one via IMPLICIT REPARAMETERIZATION (Figurnov, Mohamed & Mnih, NeurIPS 2018, arXiv:1805.08498): the
gradient through the sample is dphi/dmu = 1 and dphi/dkappa = -(dF/dkappa)/pdf, where F is the von Mises
CDF. This gives a correct gradient through the *concentration* (a straight-through sample would zero it).
The von Mises CDF and KL are the only closed forms torch lacks; both are verified in tests/check_von_mises.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, VonMises, kl_divergence

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# von Mises (bar phase): zero-mean density/CDF + implicit-reparameterization sampler
# ---------------------------------------------------------------------------

def _zero_mean_von_mises_pdf(angle: torch.Tensor, concentration: torch.Tensor) -> torch.Tensor:
    """Density of a zero-mean von Mises at ``angle`` (i0e keeps it stable at large concentration)."""
    return torch.exp(concentration * (torch.cos(angle) - 1.0)) / (TWO_PI * torch.special.i0e(concentration))


def _zero_mean_von_mises_cdf(angle: torch.Tensor, concentration: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
    """F(angle | 0, concentration) = int_{-pi}^{angle} pdf, by the trapezoid rule (torch has no vM CDF).

    Integrates along a new last axis; ``angle`` and ``concentration`` broadcast to the same shape.
    """
    lower = -math.pi
    fraction = torch.linspace(0.0, 1.0, n_steps, device=angle.device, dtype=angle.dtype)
    angle_grid = lower + fraction * (angle.unsqueeze(-1) - lower)                 # [..., n_steps]
    concentration_e = concentration.unsqueeze(-1)
    pdf = torch.exp(concentration_e * (torch.cos(angle_grid) - 1.0)) / (TWO_PI * torch.special.i0e(concentration_e))
    trapezoid_weights = torch.ones_like(pdf)
    trapezoid_weights[..., 0] = 0.5
    trapezoid_weights[..., -1] = 0.5
    step = (angle.unsqueeze(-1) - lower) / (n_steps - 1)
    return (pdf * trapezoid_weights).sum(-1) * step.squeeze(-1)


class _VonMisesImplicitReparam(torch.autograd.Function):
    """Forward: draw phi = mean + z with z ~ vonMises(0, concentration) from torch's trusted sampler.
    Backward: implicit reparameterization gradient (Figurnov et al. 2018)."""

    @staticmethod
    def forward(ctx, mean_angle, concentration):
        standardized_sample = VonMises(torch.zeros_like(concentration), concentration).sample()  # z ~ vM(0, kappa)
        ctx.save_for_backward(standardized_sample, concentration)
        return mean_angle + standardized_sample

    @staticmethod
    def backward(ctx, grad_phi):
        standardized_sample, concentration = ctx.saved_tensors
        # dF(z|kappa)/dkappa by autograd through the CDF; pdf is dF/dz. Implicit function theorem then
        # gives dz/dkappa = -(dF/dkappa)/(dF/dz) = -(dF/dkappa)/pdf.
        with torch.enable_grad():
            kappa = concentration.detach().clone().requires_grad_(True)
            cdf = _zero_mean_von_mises_cdf(standardized_sample.detach(), kappa)
            (d_cdf_d_kappa,) = torch.autograd.grad(cdf.sum(), kappa)
        pdf = _zero_mean_von_mises_pdf(standardized_sample, concentration)
        d_phi_d_kappa = -d_cdf_d_kappa / (pdf + 1e-12)
        return grad_phi, grad_phi * d_phi_d_kappa            # dphi/dmu = 1 ; dphi/dkappa = above


def sample_von_mises(mean_angle: torch.Tensor, concentration: torch.Tensor) -> torch.Tensor:
    """Reparameterized von Mises sample in [0, 2*pi) with implicit-reparam gradients through mean AND conc."""
    concentration = concentration.clamp(min=1e-4)
    return _VonMisesImplicitReparam.apply(mean_angle, concentration) % TWO_PI


def kl_von_mises(mean_q: torch.Tensor, concentration_q: torch.Tensor,
                 mean_p: torch.Tensor, concentration_p: torch.Tensor) -> torch.Tensor:
    """KL( vM(mean_q, conc_q) || vM(mean_p, conc_p) ) -- closed form (torch does not register this pair).

    KL = log(I0(conc_p)/I0(conc_q)) + A(conc_q) * (conc_q - conc_p*cos(mean_q - mean_p)),  A(k)=I1(k)/I0(k).
    i0e/i1e keep the exp(k) factors cancelled for stability. Verified against VonMises.log_prob in tests/.
    """
    concentration_q = concentration_q.clamp(min=1e-4)
    concentration_p = concentration_p.clamp(min=1e-4)
    bessel_ratio_q = torch.special.i1e(concentration_q) / torch.special.i0e(concentration_q)  # A(conc_q)
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
    """Reparameterized Normal sample (torch's rsample)."""
    return Normal(mean, std.clamp(min=1e-6)).rsample()


def kl_normal(mean_q: torch.Tensor, std_q: torch.Tensor,
              mean_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
    """KL( Normal(mean_q, std_q) || Normal(mean_p, std_p) ) via torch's registered KL."""
    return kl_divergence(Normal(mean_q, std_q.clamp(min=1e-6)), Normal(mean_p, std_p.clamp(min=1e-6)))


# ---------------------------------------------------------------------------
# Categorical (meter)
# ---------------------------------------------------------------------------

def sample_gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Differentiable Categorical relaxation -- torch's Gumbel-softmax. Returns a soft one-hot."""
    return F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)


def kl_categorical(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """KL( Categorical(log_q) || Categorical(log_p) ) via torch's registered KL.

    Inputs are (normalized) log-probabilities; passing them as ``logits`` is exact because softmax of a
    normalized log-prob vector returns the original probabilities.
    """
    return kl_divergence(Categorical(logits=log_q), Categorical(logits=log_p))
