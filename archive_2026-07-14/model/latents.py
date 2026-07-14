"""Latent distribution families of the VBPM: wrapped Cauchy (bar phase), Laplace (log-tempo),
Categorical (meter). Sampling + closed-form KLs only.

Laplace and Categorical KLs are hand-rolled closed forms verified against
``torch.distributions.kl_divergence`` (notebook verification cells); the wrapped Cauchy has no torch
implementation, so its KL, log-density, and reparameterized sampler live here and are verified by
dense quadrature in ``tests/check_wrapped_cauchy.py``. The von Mises family was archived 2026-07-10
(deviation-era archive) -- wrapped Cauchy replaced it: heavy tails match the measured tempo/microtiming
residual laws, sampling is a plain inverse-CDF transform (no implicit-reparameterization trick), and
the closed forms need no Bessel functions.
"""
import math

import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


# ---- Laplace (log-tempo increments; heavy-tailed random walk) --------------------------------

def kl_between_laplaces(loc_q, scale_q, loc_p, scale_p):
    # Closed-form KL between two Laplace laws -- the tempo term of the heavy-tailed model.
    delta = (loc_q - loc_p).abs()
    return (torch.log(scale_p / scale_q)
            + (scale_q * torch.exp(-delta / scale_q) + delta) / scale_p
            - 1.0)


def kl_between_normals(mean_q, std_q, mean_p, std_p):
    # The Gaussian baseline the model departs from -- kept for the light-tailed lineage comparison.
    return (torch.log(std_p / std_q)
            + (std_q ** 2 + (mean_q - mean_p) ** 2) / (2.0 * std_p ** 2)
            - 0.5)


# ---- Categorical (meter) ---------------------------------------------------------------------

def kl_between_categoricals(log_probs_q, log_probs_p):
    # Inputs are normalized log-probabilities over the last axis.
    return (log_probs_q.exp() * (log_probs_q - log_probs_p)).sum(dim=-1)


# ---- wrapped Cauchy (bar phase) --------------------------------------------------------------

def concentration_to_rho(concentration):
    # Map a positive concentration c to the wrapped Cauchy mean resultant length rho = c/(1+c)
    # in (0, 1): c -> 0 gives a uniform phase, c -> inf a point mass (the deterministic limit).
    concentration = concentration.clamp(min=1e-4)
    return (concentration / (1.0 + concentration)).clamp(max=1.0 - 1e-4)


def kl_between_wrapped_cauchy(mean_q, concentration_q, mean_p, concentration_p):
    # Closed-form KL between two wrapped Cauchy laws -- the phase term of the model.
    rho_q = concentration_to_rho(concentration_q)
    rho_p = concentration_to_rho(concentration_p)
    numerator = 1.0 + (rho_q * rho_p) ** 2 - 2.0 * rho_q * rho_p * torch.cos(mean_q - mean_p)
    denominator = (1.0 - rho_q ** 2) * (1.0 - rho_p ** 2)
    return torch.log(numerator / denominator)


def wrapped_cauchy_log_prob(angle, mean, concentration):
    # log density of WC(mean, rho): (1 / 2pi) (1 - rho^2) / (1 + rho^2 - 2 rho cos(angle - mean)).
    rho = concentration_to_rho(concentration)
    return (torch.log(1.0 - rho ** 2) - math.log(TWO_PI)
            - torch.log(1.0 + rho ** 2 - 2.0 * rho * torch.cos(angle - mean)))


def sample_wrapped_cauchy(mean_angle, concentration):
    # Reparameterized wrapped Cauchy sample in [0, 2pi): wrap a linear Cauchy(0, gamma),
    # gamma = -log(rho), rho = c/(1+c). A direct differentiable transform of Uniform noise, so
    # plain autograd gives the reparameterization gradient through BOTH mean and concentration.
    rho = concentration_to_rho(concentration)
    gamma = -torch.log(rho)
    u = torch.rand_like(mean_angle).clamp(1e-6, 1.0 - 1e-6)
    cauchy = gamma * torch.tan(math.pi * (u - 0.5))              # Cauchy(0, gamma) via inverse CDF
    return (mean_angle + cauchy) % TWO_PI


# ---- head activations (positive-scale maps shared by posterior, prior, and filter) -----------

def tempo_std_from_score(raw_score):
    # Laplace scale of the log-tempo increment: softplus + floor.
    return F.softplus(raw_score) + 1e-3


def phase_concentration_from_score(raw_score):
    # Wrapped Cauchy concentration (rho = c/(1+c)): softplus + floor.
    return F.softplus(raw_score) + 0.01
