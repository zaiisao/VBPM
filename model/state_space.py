"""The discrete state space: a flat index over (phase bin, tempo bin, meter).

Enumerating the per-frame state as K = n_phase * n_tempo * n_meter is what turns the continuous
bar-pointer model into a FINITE HMM (condition 1). The Markov transition and factorized emission
(conditions 2 and 3) live in transition.py / emission.py.
"""
import torch


class StateSpace:
    def __init__(self, cfg):
        """Precompute the (phase, tempo, meter) value for every flat state index."""
        raise NotImplementedError

    @property
    def n_states(self) -> int:
        """Total number of discrete states K."""
        raise NotImplementedError

    def phase(self, index: torch.Tensor) -> torch.Tensor:
        """Bar phase in [0, 2pi) for each state index."""
        raise NotImplementedError

    def tempo(self, index: torch.Tensor) -> torch.Tensor:
        """Tempo (phase advance per frame, or BPM) for each state index."""
        raise NotImplementedError

    def meter(self, index: torch.Tensor) -> torch.Tensor:
        """Beats-per-bar for each state index."""
        raise NotImplementedError
