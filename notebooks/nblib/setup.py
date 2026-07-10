"""Fixed infrastructure shared across the notebook: constants (sourced from the ACTIVE FRONTEND
YAML, never hardcoded), unit conversions, palette, seeding.

Only *non-tweaked* infrastructure lives here. The tweakable experiment dials (SMOKE_MODE, hidden
size, training steps, song counts) stay inline in the notebook so they remain visible next to the
experiments that use them.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

# The notebook imports the repo package (config/, model/, data/): make the repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import load_frontend_config  # noqa: E402

TWO_PI = 2.0 * math.pi
ACTIVE_FRONTEND = load_frontend_config("beat_this")   # swap the name to change frontend everywhere
FRAMES_PER_SECOND = ACTIVE_FRONTEND.cache_fps         # grid every cache/target uses
FPS = FRAMES_PER_SECOND
FEATURE_DIM = ACTIVE_FRONTEND.feature_dim
# Labelled fallback ONLY (illustration cells and pre-meter-readout records); the model's
# beats-per-bar always comes from the soft meter latent.
BEATS_PER_BAR = 4
READOUT_BEATS_PER_BAR = BEATS_PER_BAR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed categorical color order for every figure in the notebook (colorblind-validated set).
COLOR_BLUE, COLOR_AQUA, COLOR_YELLOW, COLOR_VIOLET, COLOR_RED = (
    "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948")


def rad_per_frame_to_bpm(omega, beats_per_bar=BEATS_PER_BAR, fps=FPS):
    return omega * beats_per_bar * 60.0 * fps / TWO_PI


def bpm_to_rad_per_frame(bpm, beats_per_bar=BEATS_PER_BAR, fps=FPS):
    return bpm * TWO_PI / (beats_per_bar * 60.0 * fps)


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


__all__ = [
    "TWO_PI", "FPS", "FRAMES_PER_SECOND", "BEATS_PER_BAR", "FEATURE_DIM", "READOUT_BEATS_PER_BAR",
    "ACTIVE_FRONTEND", "DEVICE", "COLOR_BLUE", "COLOR_AQUA", "COLOR_YELLOW", "COLOR_VIOLET",
    "COLOR_RED", "rad_per_frame_to_bpm", "bpm_to_rad_per_frame", "set_all_seeds",
]
