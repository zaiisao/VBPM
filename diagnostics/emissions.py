"""Historical frame decoders and imports for diagnostic experiments."""

import math

import torch
from torch.nn import functional as F

from diagnostics.synthetic_ladder import SparseDecoder
from torch import nn
from vbpm.util.clock_likelihood import clock_log_masses, clock_log_probs
from vbpm.util.clock_likelihood import log_interval_mass as log_interval_mass


class ClockEmission(nn.Module):
    """Trainable p(y | z) from Bernoulli metrical clock masses.

    Four circular landmarks produce beat/downbeat probabilities in each frame.
    Learned timing widths control their spread; downbeats take precedence over
    ordinary beats at coincident landmarks.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("centers", torch.arange(4) * (math.pi / 2), persistent=False)
        fraction = (1.0 - 0.001) / (math.pi - 0.001)
        self.raw_width = nn.Parameter(torch.full((2,), math.log(fraction / (1 - fraction))))

    def log_masses(self, phase, velocity):
        """Log probability of each beat landmark falling within each frame."""
        return clock_log_masses(phase, velocity, self.centers, self.raw_width)

    def forward(self, phase, velocity):
        """Return non-beat, beat, and downbeat log probabilities."""
        return clock_log_probs(self.log_masses(phase, velocity)).to(phase.dtype)


class ClockMassDecoder(ClockEmission):
    """Historical Poisson-link decoder used only by diagnostic experiments."""

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
    """Diagnostic compatibility name for the production Bernoulli emission."""

    forward = ClockEmission.forward


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
