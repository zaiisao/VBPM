"""Phase/velocity transition prior, reparameterized sampling, and emission."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .util.vonmises import sample_vonmises_icdf


class LatentState(NamedTuple):
    """Bar phase and angular velocity for a batch of states."""

    phase: torch.Tensor
    velocity: torch.Tensor  # radians per second

    def features(self):
        """Encode phase continuously across the circular boundary."""
        return torch.stack((self.phase.cos(), self.phase.sin(), self.velocity), dim=-1)


class LatentParameters(NamedTuple):
    """Von Mises phase and Gaussian velocity parameters."""

    phase_mean: torch.Tensor
    phase_concentration: torch.Tensor
    velocity_mean: torch.Tensor
    velocity_log_std: torch.Tensor


class PriorModel(nn.Module):
    """p(z_t | z_{t-1}, h_t), with fixed meter four."""

    def __init__(self, input_dim, frame_period, d_model=128):
        super().__init__()
        self.frame_period = frame_period
        self.context = nn.Sequential(nn.Linear(input_dim + 3, d_model), nn.ReLU())
        self.velocity_head = nn.Linear(d_model, 2)
        self.concentration_head = nn.Linear(d_model, 1)

    def forward(self, audio, previous, *, initial=False):
        """Predict transition parameters from audio and the previous state."""
        context = self.context(torch.cat((audio, previous.features()), dim=-1))
        velocity_mean, velocity_log_std = self.velocity_head(context).unbind(-1)
        phase_mean = previous.phase + previous.velocity * self.frame_period
        concentration = F.softplus(self.concentration_head(context).squeeze(-1)) + 1.0
        if initial:
            phase_mean = torch.zeros_like(phase_mean)
            concentration = torch.zeros_like(concentration)
        return LatentParameters(phase_mean, concentration, velocity_mean, velocity_log_std)


class PosteriorModel(nn.Module):
    """q(z_t | z_{t-1}, h, labels), with bidirectional sequence context."""

    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.encoder = nn.GRU(input_dim + 3, d_model, batch_first=True, bidirectional=True)
        self.context = nn.Sequential(nn.Linear(2 * d_model + 3, d_model), nn.ReLU())
        self.phase_head = nn.Linear(d_model, 3)
        self.velocity_head = nn.Linear(d_model, 2)

    def encode(self, h, labels):
        """Read the complete audio and N/B/D label sequences."""
        label_features = F.one_hot(labels, num_classes=3).to(h.dtype)
        context, _ = self.encoder(torch.cat((h, label_features), dim=-1))
        return context

    def forward(self, context, previous):
        """Condition each posterior factor on the previous sampled state."""
        hidden = self.context(torch.cat((context, previous.features()), dim=-1))
        phase_cos, phase_sin, raw_concentration = self.phase_head(hidden).unbind(-1)
        phase_mean = torch.atan2(phase_sin, phase_cos)
        concentration = F.softplus(raw_concentration) + 1.0
        velocity_mean, velocity_log_std = self.velocity_head(hidden).unbind(-1)
        return LatentParameters(phase_mean, concentration, velocity_mean, velocity_log_std)


class LatentSampler(nn.Module):
    """Draw phase and velocity from the transition prior at every frame."""

    def forward(self, parameters, *, initial=False, sample=True):
        """Use distribution means for deterministic inference."""
        velocity = parameters.velocity_mean
        if sample:
            velocity = velocity + parameters.velocity_log_std.exp() * torch.randn_like(velocity)
        phase = parameters.phase_mean
        if sample:
            phase = (
                (torch.rand_like(phase) * 2 - 1) * torch.pi
                if initial
                else phase + sample_vonmises_icdf(parameters.phase_concentration)
            )
        return LatentState(phase, velocity)


class EmissionModel(nn.Module):
    """p(y_t | z_t): non-beat, beat, and downbeat logits."""

    def __init__(self, d_model=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, 3))

    def forward(self, state):
        """Predict three label logits from phase and velocity only."""
        return self.network(state.features())
