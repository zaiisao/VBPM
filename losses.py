"""The training objective.

For Baseline B the objective is simply the negative exact log-likelihood -- optionally plus a
supervised term if you want to nudge the latent states toward annotated beats/downbeats.
"""
import torch


def objective(model, batch) -> torch.Tensor:
    """Return a scalar loss to MINIMIZE.

    Default: -model.log_likelihood(features, observations).mean().
    """
    raise NotImplementedError
