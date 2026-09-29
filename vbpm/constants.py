"""Numeric constants of the bar-pointer model: geometry, priors, and measured fits.

Every value here is either a mathematical constant, a numerical safety bound, or a
corpus-measured fit. Nothing here is a per-run choice -- those live in the yaml
configs and reach the model through vbpm/variants/*.py.
"""
from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi

PRIOR_PHASE_KAPPA = 383.0
KAPPA_INTER = 17.0
MAX_KAPPA = 1.0e7

DELTA_MAX = 0.35
CORRECTION_MAX = 0.5

FPS = 50.0
TOLERANCE_SECONDS = 0.070

METER_STAY = 0.999487
METER0_SHARE = {3: 0.15, 4: 0.85}
TEMPO0_BPM = 120.0
TEMPO0_BPM_SD = 40.0

METER_SONG_SHARE = {2: 0.0432, 3: 0.0911, 4: 0.8307, 5: 0.0069,
                    6: 0.0063, 7: 0.0100, 8: 0.0118}

EMISSION_FIT_A = -9.730409
EMISSION_FIT_B = 11.958140
EMISSION_FIT_TAU = 0.349944
EMISSION_FIT_TAU_BACK = 0.341275

TEMPO_BOUND_MARGIN = 0.35
TEMPO_LO, TEMPO_HI = math.log(0.01), math.log(0.2)
TEMPO_PRIOR_MU = -2.6827
TEMPO_PRIOR_SIGMA = 0.3903
TEMPO_PRIOR_MU_LEGACY = -2.5028
TEMPO_PRIOR_SIGMA_LEGACY = 0.5005
TEMPO_PRIOR_EPS = 0.02
TEMPO_WALK_SIGMA = 0.00212
TEMPO_SIGMA_CEIL = 0.25
TEMPO_SIGMA_INIT = 0.15

WALK_MIX_W = (0.687, 0.313)
WALK_MIX_SIGMA = (0.00029, 0.00377)
WALK_INTRA_SIGMA = 0.00029
WALK_INTER_W = (0.646, 0.354)
WALK_INTER_SIGMA = (0.0247, 0.198)

BAR_POOL_ITERS = 8
KAPPA_Q_MIN = 0.01
