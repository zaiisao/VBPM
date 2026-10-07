"""Circular Gaussian interval masses and Bernoulli clock probabilities."""

import math

import torch
from torch.nn import functional as F


def log_interval_mass(lower, upper, width):
    """Compute stable Gaussian log probability in an interval."""
    reflect = lower + upper > 0
    lo = torch.where(reflect, -upper, lower) / width
    hi = torch.where(reflect, -lower, upper) / width
    log_hi = torch.special.log_ndtr(hi)
    log_lo = torch.special.log_ndtr(lo)
    difference = (log_lo - log_hi).clamp_max(-torch.finfo(torch.float64).eps)
    return log_hi + torch.log(-torch.expm1(difference))


def clock_log_masses(phase, velocity, centers, raw_width):
    """Log probability of each beat landmark falling within each frame."""
    # Each frame spans half of the preceding and following phase steps.
    previous_step = torch.cat((velocity[..., :1], velocity), dim=-1)
    next_step = torch.cat((velocity, velocity[..., -1:]), dim=-1)
    previous_step = previous_step.clamp_min(1e-4).double()
    next_step = next_step.clamp_min(1e-4).double()

    phase_offset = phase[..., None] - centers
    phase_offset = torch.atan2(phase_offset.sin(), phase_offset.cos()).double()
    lower = phase_offset - previous_step[..., None] / 2
    upper = phase_offset + next_step[..., None] / 2

    # Widths are ordered [beat, downbeat]; landmark zero is the downbeat.
    min_width = 0.001
    width_fraction = (-F.softplus(-raw_width.double())).exp()
    beat_width, downbeat_width = min_width + (math.pi - min_width) * width_fraction
    landmark_widths = torch.stack((downbeat_width, beat_width, beat_width, beat_width))

    # Split intervals crossing the circular boundary into their wrapped pieces.
    central_mass = log_interval_mass(
        lower.clamp_min(-math.pi), upper.clamp_max(math.pi), landmark_widths
    )
    left_wrapped_mass = log_interval_mass(
        lower + 2 * math.pi, torch.full_like(upper, math.pi), landmark_widths
    )
    right_wrapped_mass = log_interval_mass(
        torch.full_like(lower, -math.pi), upper - 2 * math.pi, landmark_widths
    )
    left_wrapped_mass = left_wrapped_mass.masked_fill(lower >= -math.pi, -torch.inf)
    right_wrapped_mass = right_wrapped_mass.masked_fill(upper <= math.pi, -torch.inf)
    interval_mass = torch.logsumexp(
        torch.stack((central_mass, left_wrapped_mass, right_wrapped_mass)), dim=0
    )

    # Normalize the Gaussian over the circular domain [-pi, pi].
    log_normalizer = torch.erf(math.pi / (math.sqrt(2) * landmark_widths)).log()
    return interval_mass - log_normalizer


def clock_log_probs(log_mass):
    """Convert landmark masses to non-beat, beat, and downbeat log probabilities."""
    output_dtype = log_mass.dtype
    log_mass = log_mass.clamp_max(-1e-12)
    log_missing = torch.log(-torch.expm1(log_mass))
    log_no_beat = log_missing[..., 1:].sum(-1)
    log_beat = torch.log(-torch.expm1(log_no_beat.clamp_max(-1e-20)))
    # For remote kernels, exp underflow would make the complement zero;
    # the small-probability union approaches the sum of its masses.
    log_beat = torch.where(log_no_beat > -1e-20, torch.logsumexp(log_mass[..., 1:], -1), log_beat)
    logits = torch.stack(
        (log_missing[..., 0] + log_no_beat, log_missing[..., 0] + log_beat, log_mass[..., 0]),
        -1,
    )
    # Tiny observation-error floor supplies finite categorical likelihoods.
    # It is fixed and never used as a latent supervision or count target.
    error = 1e-8
    floor = torch.full_like(logits, math.log(error / 3))
    return torch.logaddexp(logits + math.log1p(-error), floor).to(output_dtype)
