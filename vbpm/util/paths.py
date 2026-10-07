"""Format latent trajectories for readouts and evaluation."""

import torch


def format_latent_path(phase, velocity):
    """Expose physical trajectories in the shared VBPM path format."""
    return dict(
        phi_path=phase,
        velocity_path=torch.cat((velocity, velocity[:, -1:]), 1),
        meter_path=phase.new_ones((*phase.shape, 1)),
    )
