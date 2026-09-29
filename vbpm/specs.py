"""The constructor's argument groups: one dataclass per cluster of knobs."""
from __future__ import annotations

import dataclasses
import math

from .constants import (PRIOR_PHASE_KAPPA, TEMPO_PRIOR_MU, TEMPO_PRIOR_SIGMA,
                        TEMPO_WALK_SIGMA)


@dataclasses.dataclass
class EmissionSpec:
    """p(y_t | phi_t): which shape reads the latent, and how big it is."""

    kind: str = "triangle"
    layers: int = 2
    dim: int = 64
    positional: bool = False
    bump_kappa: float = 20.0
    frozen: bool = False
    floor: float = 0.0
    nested: bool = False


@dataclasses.dataclass
class WalkSpec:
    """p(phi_t | phi_t-1): the tempo walk's law and the phase prior's tightness."""

    kind: str = "gauss"
    prior_phase_kappa: float = PRIOR_PHASE_KAPPA
    kappa_gate: bool = False
    tempo_mu: float = TEMPO_PRIOR_MU
    tempo_sigma: float = TEMPO_PRIOR_SIGMA
    walk_sigma: float = TEMPO_WALK_SIGMA

    def __post_init__(self):
        self.prior_phase_kappa = float(self.prior_phase_kappa)
        self.kappa_gate = bool(self.kappa_gate)
        self.tempo_mu = float(self.tempo_mu)
        self.tempo_sigma = float(self.tempo_sigma)
        self.walk_sigma = float(self.walk_sigma)


@dataclasses.dataclass
class PlacementSpec:
    """How the placement factor reads phase, and which paths reach it."""

    coord: str = "first"
    lift: float = 0.0
    attach: bool = False
    offset_marginal: int = 1

    def __post_init__(self):
        self.lift = float(self.lift)
        self.attach = bool(self.attach)


@dataclasses.dataclass
class UpdateSpec:
    """q(phi_t)'s free mean: the update half of the filter."""

    delta_on: bool = False
    gate_cond: bool = True

    def __post_init__(self):
        self.delta_on = bool(self.delta_on)
        self.gate_cond = bool(self.gate_cond)


@dataclasses.dataclass
class DecoderSpec:
    """The knot decoder that emits the rate correction and delta."""

    dim: int = 32
    knot_stride: int = 25

    def __post_init__(self):
        self.dim = int(self.dim)
        self.knot_stride = int(self.knot_stride)


@dataclasses.dataclass
class RateSpec:
    """q(rate | x): the candidate set and how far a candidate may be trimmed.

    The candidates are summed exactly, so this is enumeration over a small
    discrete set rather than a discretisation of the tempo axis -- the
    distinction the project's continuity rule turns on. `resid` lets each
    candidate move continuously off its nominal value, which is also why the
    prior must be priced at the shifted rate rather than at the nominal one.
    """

    grid: int = 24
    lo: float = 0.020
    hi: float = 0.200
    per_bar: bool = True
    meters: tuple = (2, 3, 4, 5, 6, 7, 8)
    meter_prior: str = "corpus"
    posterior: str = "categorical"
    resid: float = 0.0


@dataclasses.dataclass
class ChainSpec:
    """The autoregressive phase chain: its step, its kernel, and its base case."""

    stride: int = 1
    phase_kernel: str = "vonmises"
    delta_max: float = math.pi
    delta_rel: float = 0.0
    phi0: str = "amortized"          # amortized | anchor
    phi0_grid: int = 0               # retired; an atomic q against a continuous prior


@dataclasses.dataclass
class TempoWalkSpec:
    """Whether log-tempo moves within a window, and under which law.

    Off by default: the corpus says tempo is mean-reverting rather than
    diffusing (structure-function growth exponent 0.26 over six datasets), and
    every driftless walk law measured so far pays a large bribe for standing
    still. The scale comes from WalkSpec.walk_sigma.
    """

    enabled: bool = False
    kernel: str = "cauchy"
    revert: bool = False
