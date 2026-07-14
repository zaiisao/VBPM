"""Baseline B: the input-driven neural bar-pointer HMM.

Ties together StateSpace + TransitionModel + EmissionModel + exact inference. Because the transition
is Markov and the emission factorized, log_likelihood() and decode() are EXACT -- there is no VAE,
no amortized encoder, and no ELBO anywhere in this model.
"""
import torch
import torch.nn as nn

from model.state_space import StateSpace
from model.transition import TransitionModel
from model.emission import EmissionModel


class NeuralBarPointerHMM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    def context(self, features: torch.Tensor) -> torch.Tensor:
        """Encode audio features x: [B, T, feature_dim] into per-frame context h_t: [B, T, H].

        h is a function of x ONLY -- it must not see the latent states, or the model stops being
        Markov-in-z. (The Transformer, if any, attends over x here, never over z.)
        """
        raise NotImplementedError

    def log_likelihood(self, features: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """Exact marginal log p(obs | x): [B], via the forward algorithm. Maximize this to train."""
        raise NotImplementedError

    def decode(self, features: torch.Tensor, observations: torch.Tensor) -> dict:
        """Viterbi MAP path -> readout. Returns {'beats': ..., 'downbeats': ..., 'tempo': ...}."""
        raise NotImplementedError

    def posteriors(self, features: torch.Tensor, observations: torch.Tensor):
        """forward-backward marginals over states (EM E-step / diagnostics)."""
        raise NotImplementedError
