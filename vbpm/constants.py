"""Numeric constants of the bar-pointer model: geometry, priors, and measured fits.

Every value here is either a mathematical constant, a numerical safety bound, or a
corpus-measured fit. Nothing here is a per-run choice -- those live in the yaml
configs and reach the model through vbpm/model.py's build_model.
"""
from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi
FPS = 50.0

PRIOR_PHASE_KAPPA = 383.0
KAPPA_Q_MIN = 0.01

METER0_SHARE = {3: 0.15, 4: 0.85}
TEMPO0_BPM = 120.0
TEMPO0_BPM_SD = 40.0
