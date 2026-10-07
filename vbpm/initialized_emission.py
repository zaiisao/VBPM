"""Trainable Transformer emission initialized in bar-phase coordinates.

A trainable Fourier output branch supplies an initial beat-aware function. The
Transformer is a trainable residual, initially zero at its output. Every branch
is optimized jointly with the prior under the same sampled reconstruction loss.
There are no frozen emission coefficients and no annotation inputs.
"""

from __future__ import annotations
import torch
from torch import nn
from .nets import EmissionModel


class InitializedEmission(nn.Module):
    """Provide an initialized trainable observation likelihood."""

    def __init__(self, spec, meters, input_dim, beat_concentration=64.0, down_concentration=512.0):
        super().__init__()
        self.transformer = EmissionModel(spec, meters, input_dim)
        self.phase_output = nn.Linear(8 + len(meters), 3)
        with torch.no_grad():
            self.transformer.out.weight.zero_()
            self.transformer.out.bias.zero_()
            self.phase_output.weight.zero_()
            self.phase_output.bias.zero_()
            # Eight phase features: cos/sin of harmonics 1,...,4. Initially,
            # beats recur at the four quarter phases and downbeats at phase 0.
            # Both sine/cosine weights and all offsets remain fully trainable.
            self.phase_output.weight[1, 6] = beat_concentration
            self.phase_output.weight[1, 0] = -2.0
            self.phase_output.bias[1] = 4.0 - beat_concentration
            self.phase_output.weight[2, 0] = down_concentration
            self.phase_output.bias[2] = 4.0 - down_concentration

    def forward(self, phi, velocity, meter, mask, h):
        """Evaluate the network on its input tensors."""
        harmonics = phi[..., None] * torch.arange(1, 5, device=phi.device, dtype=phi.dtype)
        phase_features = torch.stack([harmonics.cos(), harmonics.sin()], -1).flatten(-2)
        features = torch.cat([phase_features, meter], -1)
        return self.phase_output(features) + self.transformer(phi, velocity, meter, mask, h)

    def loglik(self, phi, velocity, meter, labels, mask, h):
        """Return summed observation log likelihood."""
        logp = self(phi, velocity, meter, mask, h).log_softmax(-1)
        return (logp.gather(-1, labels[..., None]).squeeze(-1) * mask).sum(1)
