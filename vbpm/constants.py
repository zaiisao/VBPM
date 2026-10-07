"""Numeric constants of the bar-pointer model: geometry, priors, and measured fits.

Every value here is either a mathematical constant, a numerical safety bound, or a
corpus-measured fit. Nothing here is a per-run choice -- those live in the yaml
configs and reach the model through vbpm/model.py's build_model.
"""

from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi

PRIOR_PHASE_KAPPA = 383.0
KAPPA_MIN = 1.0
LOG_VELOCITY_STEP_PER_SQRT_SECOND = 0.074

METER0_SHARE = {3: 0.15, 4: 0.85}
METER_TRANSITION = {3: {3: 0.99792, 4: 0.00208}, 4: {3: 0.00087, 4: 0.99913}}
Q_METER0_LOGIT = 5.0
