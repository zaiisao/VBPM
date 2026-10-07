"""Shared normalized clock and frame-bin observation decoders."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from diagnostics.synthetic_ladder import SparseDecoder


def log_interval_mass(lower, upper, width):
    """Compute stable Gaussian log probability in an interval."""
    reflect = lower + upper > 0
    lo = torch.where(reflect, -upper, lower) / width
    hi = torch.where(reflect, -lower, upper) / width
    log_hi = torch.special.log_ndtr(hi)
    log_lo = torch.special.log_ndtr(lo)
    difference = (log_lo - log_hi).clamp_max(-torch.finfo(torch.float64).eps)
    return log_hi + torch.log(-torch.expm1(difference))


class ClockMassDecoder(nn.Module):
    """Decode normalized metrical event masses into categorical probabilities."""

    def __init__(self):
        super().__init__()
        self.register_buffer("centers", torch.arange(4) * (math.pi / 2), persistent=False)
        fraction = (1.0 - 0.001) / (math.pi - 0.001)
        self.raw_width = nn.Parameter(torch.full((2,), math.log(fraction / (1 - fraction))))

    def log_masses(self, phase, velocity):
        """Return metrical event log masses in each frame interval."""
        previous = torch.cat((velocity[..., :1], velocity), -1).clamp_min(1e-4)
        following = torch.cat((velocity, velocity[..., -1:]), -1).clamp_min(1e-4)
        distance = phase[..., None] - self.centers
        distance = torch.atan2(distance.sin(), distance.cos()).double()
        lower = distance - 0.5 * previous[..., None].double()
        upper = distance + 0.5 * following[..., None].double()
        sigma = 0.001 + (math.pi - 0.001) * (-F.softplus(-self.raw_width.double())).exp()
        width = torch.stack((sigma[1], sigma[0], sigma[0], sigma[0]))
        central = log_interval_mass(lower.clamp_min(-math.pi), upper.clamp_max(math.pi), width)
        below = log_interval_mass(lower + 2 * math.pi, torch.full_like(upper, math.pi), width)
        above = log_interval_mass(torch.full_like(lower, -math.pi), upper - 2 * math.pi, width)
        below = torch.where(lower < -math.pi, below, torch.full_like(below, -torch.inf))
        above = torch.where(upper > math.pi, above, torch.full_like(above, -torch.inf))
        log_mass = torch.logsumexp(torch.stack((central, below, above)), 0)
        log_mass = log_mass - torch.erf(math.pi / (math.sqrt(2) * width)).log()
        return log_mass

    def forward(self, phase, velocity):
        """Evaluate the network on its input tensors."""
        log_mass = self.log_masses(phase, velocity)
        log_beat = torch.logsumexp(log_mass[..., 1:], -1)
        log_down = log_mass[..., 0]
        log_total = torch.logaddexp(log_beat, log_down)
        # Remote kernels can have log mass far below floating-point exp range.
        # The small-intensity event probability approaches the intensity itself.
        stable_log_event = torch.log(-torch.expm1(-log_total.clamp_min(-20).exp()))
        log_event = torch.where(log_total < -20, log_total, stable_log_event)
        return torch.stack(
            (-log_total.exp(), log_event + log_beat - log_total, log_event + log_down - log_total),
            -1,
        ).to(phase.dtype)


class BernoulliClockDecoder(ClockMassDecoder):
    """Independent landmark-bin probabilities, merging coincident labels.

    A downbeat takes precedence over coincident ordinary beats. Unlike the
    Poisson link, one concentrated landmark can have event probability near
    one. Normalized timing mass and structural centers stay unchanged.
    """

    def forward(self, phase, velocity):
        """Evaluate the network on its input tensors."""
        log_mass = self.log_masses(phase, velocity).clamp_max(-1e-12)
        log_missing = torch.log(-torch.expm1(log_mass))
        log_no_beat = log_missing[..., 1:].sum(-1)
        log_beat = torch.log(-torch.expm1(log_no_beat.clamp_max(-1e-20)))
        # For remote kernels, exp underflow would make the complement zero;
        # the small-probability union approaches the sum of its masses.
        log_beat = torch.where(
            log_no_beat > -1e-20, torch.logsumexp(log_mass[..., 1:], -1), log_beat
        )
        logits = torch.stack(
            (log_missing[..., 0] + log_no_beat, log_missing[..., 0] + log_beat, log_mass[..., 0]),
            -1,
        )
        # Tiny observation-error floor supplies finite categorical likelihoods.
        # It is fixed and never used as a latent supervision or count target.
        error = 1e-8
        floor = torch.full_like(logits, math.log(error / 3))
        return torch.logaddexp(logits + math.log1p(-error), floor).to(phase.dtype)


class FrameBinDecoder(SparseDecoder):
    """Integrate event probability over frame intervals."""

    def forward(self, phase, velocity):
        """Evaluate the network on its input tensors."""
        previous = torch.cat((velocity[..., :1], velocity), -1).clamp_min(1e-4)
        following = torch.cat((velocity, velocity[..., -1:]), -1).clamp_min(1e-4)
        distance = phase[..., None] - self.centers
        distance = torch.atan2(distance.sin(), distance.cos())
        after = distance >= 0
        local = torch.where(after, previous[..., None], following[..., None])
        other = torch.where(after, following[..., None], previous[..., None])
        # Double precision keeps normal-CDF tail gradients accurate. Reflection
        # puts remote event centers in the negative tail, avoiding cancellation
        # between two CDF values that both round to one.
        offset = (distance.abs() / local).double()
        width = (0.02 + F.softplus(self.raw_width)).double()
        upper = (0.5 - offset) / width
        lower = (-0.5 * (other / local).double() - offset) / width
        log_upper = torch.special.log_ndtr(upper)
        log_lower = torch.special.log_ndtr(lower)
        difference = (log_lower - log_upper).clamp_max(-torch.finfo(torch.float64).eps)
        log_mass = log_upper + torch.log(-torch.expm1(difference))
        event = self.height + log_mass.to(phase.dtype)
        return torch.stack(
            (torch.zeros_like(phase), torch.logsumexp(event[..., 1:], -1), event[..., 0]), -1
        )

    @torch.no_grad()
    def preserve_point_peak_heights(self):
        """Prediction-only initialization when transferring a point decoder."""
        width = 0.02 + F.softplus(self.raw_width)
        central_mass = torch.erf(0.5 / (math.sqrt(2) * width))
        self.height.sub_(central_mass.log())


class AngularFrameBinDecoder(FrameBinDecoder):
    """Integrate trainable angular uncertainty over physical frame intervals.

    Unlike a fixed timing width, angular uncertainty does not expand with
    predicted tempo. The 0.04 conversion initializes the inherited width
    parameters in radians; it is a fixed coordinate scale, not a tempo target.
    """

    def forward(self, phase, velocity):
        """Evaluate the network on its input tensors."""
        previous = torch.cat((velocity[..., :1], velocity), -1).clamp_min(1e-4)
        following = torch.cat((velocity, velocity[..., -1:]), -1).clamp_min(1e-4)
        distance = phase[..., None] - self.centers
        distance = torch.atan2(distance.sin(), distance.cos())
        local = torch.where(distance >= 0, previous[..., None], following[..., None])
        other = torch.where(distance >= 0, following[..., None], previous[..., None])
        width = ((0.02 + F.softplus(self.raw_width)) * 0.04).double()
        upper = (0.5 * local - distance.abs()).double() / width
        lower = (-0.5 * other - distance.abs()).double() / width
        log_upper = torch.special.log_ndtr(upper)
        log_lower = torch.special.log_ndtr(lower)
        difference = (log_lower - log_upper).clamp_max(-torch.finfo(torch.float64).eps)
        log_mass = log_upper + torch.log(-torch.expm1(difference))
        event = self.height + log_mass.to(phase.dtype)
        return torch.stack(
            (torch.zeros_like(phase), torch.logsumexp(event[..., 1:], -1), event[..., 0]), -1
        )
