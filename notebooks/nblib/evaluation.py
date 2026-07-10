"""Thin re-export: deployment scoring lives in the package (model/readout.py) -- single source.

The package functions take ``fps``/``device`` explicitly; here they are bound to the notebook's
active-frontend constants so existing notebook call sites keep their old signatures.
"""
import functools

from .setup import DEVICE, FRAMES_PER_SECOND
from model.readout import (DEFAULT_EVAL_MAX_FRAMES, TWO_PI, beat_f_measure,  # noqa: F401
                           peak_pick_times, song_beats_per_bar)
from model.readout import (evaluate_filter_readout as _filter, evaluate_geometric_readout as _geometric,
                           evaluate_prior_readout as _prior, phase_wrap_times as _wrap)


def phase_wrap_times(phase_trajectory, min_separation_seconds, fps=FRAMES_PER_SECOND):
    return _wrap(phase_trajectory, min_separation_seconds, fps)


def evaluate_prior_readout(model, songs, max_frames=1600, **kwargs):
    return _prior(model, songs, FRAMES_PER_SECOND, max_frames, device=DEVICE, **kwargs)


def evaluate_geometric_readout(model, songs, audio_condition="real", max_frames=1600, **kwargs):
    return _geometric(model, songs, FRAMES_PER_SECOND, audio_condition, max_frames, device=DEVICE, **kwargs)


def evaluate_filter_readout(model, songs, max_frames=1600, num_particles=400, **kwargs):
    return _filter(model, songs, FRAMES_PER_SECOND, max_frames, device=DEVICE,
                   num_particles=num_particles, **kwargs)


__all__ = ["phase_wrap_times", "beat_f_measure", "peak_pick_times", "song_beats_per_bar",
           "evaluate_prior_readout", "evaluate_geometric_readout", "evaluate_filter_readout", "TWO_PI"]
