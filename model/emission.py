"""Emission factor: p(obs_t | z_t = j).

MUST be factorized: depends only on the current state z_t (and fixed params), never on other
frames' states or observations. This is condition 3.
"""
import torch
import torch.nn as nn


class EmissionModel(nn.Module):
    def __init__(self, cfg, state_space):
        super().__init__()
        raise NotImplementedError

    def log_emission(self, observations: torch.Tensor) -> torch.Tensor:
        """observations: [B, T, obs_dim].

        Returns log_B: [B, T, K] with log_B[b, t, j] = log p(obs_t | z_t = j).
        """
        raise NotImplementedError
