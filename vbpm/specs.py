"""The constructor's argument groups: one dataclass per cluster of knobs."""
from __future__ import annotations

import dataclasses

from .constants import PRIOR_PHASE_KAPPA


@dataclasses.dataclass
class EmissionSpec:
    """p(labels | z): the size of the emission Transformer."""

    layers: int = 2
    dim: int = 64
    positional: bool = False


@dataclasses.dataclass
class WalkSpec:
    """The prior's phase concentration around the tempo advance."""

    prior_phase_kappa: float = PRIOR_PHASE_KAPPA

    def __post_init__(self):
        self.prior_phase_kappa = float(self.prior_phase_kappa)
