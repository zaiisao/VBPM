"""The constructor's argument groups: one dataclass per cluster of knobs."""
from __future__ import annotations

import dataclasses

from .constants import PRIOR_PHASE_KAPPA_PER_SECOND


@dataclasses.dataclass
class EmissionSpec:
    """p(labels | z): the size of the emission Transformer."""

    layers: int = 2
    dim: int = 64
    positional: bool = False


@dataclasses.dataclass
class PriorSpec:
    """p(path | x): the phase concentration around the tempo advance."""

    phase_kappa_per_second: float = PRIOR_PHASE_KAPPA_PER_SECOND

    def __post_init__(self):
        self.phase_kappa_per_second = float(self.phase_kappa_per_second)


@dataclasses.dataclass
class PosteriorSpec:
    """q(path | x, labels): the size of the encoder trunk."""

    d_model: int = 128
