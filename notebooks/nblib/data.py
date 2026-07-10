"""Thin re-export: the notebook's data layer IS the package's (data/dataset.py) -- single source.

``sample_training_crops`` is bound to the notebook's DEVICE so existing call sites are unchanged.
"""
import functools

from .setup import DEVICE
from data.dataset import Song, load_cached_songs  # noqa: F401
from data.dataset import sample_training_crops as _sample_training_crops

sample_training_crops = functools.partial(_sample_training_crops, device=DEVICE)

__all__ = ["Song", "load_cached_songs", "sample_training_crops"]
