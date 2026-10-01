"""Numeric constants of the bar-pointer model: geometry, priors, and measured fits.

Every value here is either a mathematical constant, a numerical safety bound, or a
corpus-measured fit. Nothing here is a per-run choice -- those live in the yaml
configs and reach the model through vbpm/model.py's build_model.
"""
from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi

PRIOR_PHASE_KAPPA = 383.0

METER0_SHARE = {3: 0.15, 4: 0.85}
LOG_TEMPO0 = {3: (-1.056, 0.584), 4: (-1.355, 0.346)}
METER_TRANSITION = {3: {3: 0.99792, 4: 0.00208}, 4: {3: 0.00087, 4: 0.99913}}
LOG_TEMPO_CHANGE_SCALE = 0.03
Q_LOG_TEMPO_CHANGE_UNIT = 0.004
LOG_TEMPO_CHANGE_SD_Q0 = 0.0005
Q_PHASE0_KAPPA = 100.0
Q_LOG_TEMPO0_SD = 0.002
Q_METER0_LOGIT = 5.0
Q_FEATURE_SCALE = 0.01
CLASS_FREQ = (0.9635, 0.0255, 0.0110)
