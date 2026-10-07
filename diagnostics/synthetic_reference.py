"""Shared synthetic cues and dense phase decoders for the GSNN ladder."""

import math

import torch
from torch import nn

KNOWN_V = 0.35
PHASE_KAPPA = 40.0


def emission(phase):
    """Fixed physical origin and eight distinguishable circular channels."""
    centers = torch.arange(8, device=phase.device, dtype=phase.dtype) * 2 * math.pi / 8
    probability = 0.05 + 0.9 * torch.sigmoid(4 * torch.cos(phase[..., None] - centers))
    return torch.logit(probability)


def make_data(stage, count, frames, seed):
    # Each stage reuses the same base draws. Velocity laws intentionally differ.
    """Generate synthetic audio cues and observations."""
    rng = torch.Generator().manual_seed(seed)
    initial_phase = 2 * math.pi * torch.rand(count, generator=rng) - math.pi
    initial_velocity = 0.2 + 0.3 * torch.rand(count, generator=rng)
    trend = torch.randint(3, (count,), generator=rng) - 1
    innovations = 0.002 * torch.randn(count, frames - 2, generator=rng)
    input_noise = 0.08 * torch.randn(count, frames, 2, generator=rng)
    observation_uniform = torch.rand(count, frames, 8, generator=rng)
    if stage == "phase_only":
        velocity = torch.full((count, frames - 1), KNOWN_V)
    elif stage == "constant_tempo":
        velocity = initial_velocity[:, None].expand(-1, frames - 1).clone()
    elif stage == "changing_tempo":
        increments = 0.003 * trend[:, None] + innovations
        velocity = torch.cat(
            (initial_velocity[:, None], initial_velocity[:, None] + increments.cumsum(1)), 1
        )
    else:
        raise ValueError(stage)
    phase = torch.cat((initial_phase[:, None], initial_phase[:, None] + velocity.cumsum(1)), 1)
    x = torch.stack((phase.cos(), phase.sin()), -1) + input_noise
    y = (observation_uniform < emission(phase).sigmoid()).float()
    return dict(x=x, y=y, phase=phase, velocity=velocity, trend=trend)


class DistributionNetwork(nn.Module):
    """Encode synthetic cues into phase and Gaussian tempo parameters."""

    def __init__(self, input_size, hidden):
        super().__init__()
        self.backbone = nn.GRU(input_size, hidden, batch_first=True, bidirectional=True)
        width = 2 * hidden + input_size
        self.phase_head = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.initial_velocity = nn.Linear(width, 2)
        self.velocity_increment = nn.Linear(width, 2)
        with torch.no_grad():
            self.phase_head[-1].weight.mul_(0.1)
            self.phase_head[-1].bias.copy_(torch.tensor([1.0, 0.0]))
            self.initial_velocity.weight.zero_()
            self.initial_velocity.bias.copy_(torch.tensor([0.0, -4.0]))
            self.velocity_increment.weight.zero_()
            self.velocity_increment.bias.copy_(torch.tensor([0.0, -6.0]))

    def forward(self, inputs):
        """Evaluate the network on its input tensors."""
        h, _ = self.backbone(inputs)
        features = torch.cat((inputs, h), -1)
        direction = self.phase_head(features[:, 0])
        # No atan2(0,0): the denominator receives a tiny positive regularizer.
        phase = torch.atan2(direction[:, 1], direction[:, 0] + 1e-8)
        initial = self.initial_velocity(features[:, 0])
        increments = self.velocity_increment(features[:, 1:-1])
        return dict(
            phase=phase,
            velocity_mean=KNOWN_V + initial[:, 0],
            velocity_log_std=initial[:, 1].clamp(-7, -1),
            increment_mean=increments[..., 0],
            increment_log_std=increments[..., 1].clamp(-9, -3),
        )


def noise_bank(steps, batch, frames, seed):
    # Concentration is FIXED, so drawing centered Von Mises offsets needs no
    # implicit concentration gradient. Rotation has an exact pathwise gradient.
    """Generate reproducible phase and velocity noise for training."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        offsets = torch.distributions.VonMises(torch.tensor(0.0), torch.tensor(PHASE_KAPPA)).sample(
            (steps, 2, batch)
        )
    rng = torch.Generator().manual_seed(seed + 1)
    return dict(
        phase=offsets,
        initial=torch.randn(steps, 2, batch, generator=rng),
        increments=torch.randn(steps, 2, batch, frames - 2, generator=rng),
    )


def extract_noise(bank, step):
    """Select the two Monte Carlo noise branches for a training step."""
    return [{key: value[step, branch] for key, value in bank.items()} for branch in range(2)]


def circular_error(estimate, truth):
    """Return mean wrapped phase error in degrees."""
    delta = estimate - truth
    return torch.atan2(delta.sin(), delta.cos()).abs().mean() * 180 / math.pi


def correlation(a, b):
    """Return correlation when both inputs have nonzero variance."""
    aa, bb = a.flatten() - a.mean(), b.flatten() - b.mean()
    denom = (aa.square().sum() * bb.square().sum()).sqrt()
    return None if float(denom) < 1e-9 else float((aa * bb).sum() / denom)


class CircularBernoulliDecoder(nn.Module):
    """Trainable circular neural decoder, initialized to the old probabilities.

    The projection's weights/biases and both probability bounds are free.
    Ordered sigmoid bounds keep every output a valid Bernoulli probability.
    """

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 8)
        self.lower_logits = nn.Parameter(torch.full((8,), math.log(0.05 / 0.95)))
        self.span_logits = nn.Parameter(torch.full((8,), math.log(0.9 / 0.05)))
        centers = torch.arange(8) * 2 * math.pi / 8
        with torch.no_grad():
            self.projection.weight.copy_(4 * torch.stack((centers.cos(), centers.sin()), 1))
            self.projection.bias.zero_()

    def forward(self, phase):
        """Evaluate the network on its input tensors."""
        features = torch.stack((phase.cos(), phase.sin()), -1)
        activation = self.projection(features).sigmoid()
        lower = self.lower_logits.sigmoid()
        amplitude = (1 - lower) * self.span_logits.sigmoid()
        probability = lower + amplitude * activation
        return torch.logit(probability.clamp(1e-6, 1 - 1e-6))


class MLPDecoder(nn.Module):
    """Fully learned logits from circular phase features; random initialization."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 8)
        )

    def forward(self, phase):
        """Evaluate the network on its input tensors."""
        return self.network(torch.stack((phase.cos(), phase.sin()), -1))
