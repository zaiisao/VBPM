"""Transition factor: p(z_t = j | z_{t-1} = i, h_t).

MUST be Markov in z: it may depend on z_{t-1} and the audio context h_t = f(x), but NOT on
z_{1:t-2}. (Input-driven HMM, like madmom's DBN conditioned on its activation.) This is condition 2;
break it and the forward algorithm is no longer exact.
"""
import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    def __init__(self, cfg, state_space):
        super().__init__()
        raise NotImplementedError

    def log_transition_matrix(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, T, H] audio-derived h_t.

        Returns log_A: [B, T, K, K] with log_A[b, t, i, j] = log p(z_t = j | z_{t-1} = i, h_t),
        rows normalized over j. (A structured/sparse form is fine as long as it stays Markov.)
        """
        raise NotImplementedError
