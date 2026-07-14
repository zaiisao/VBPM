"""Baseline A: madmom's handcrafted DBN (Krebs et al. 2015) -- the reference point for the ladder."""
import numpy as np


def madmom_dbn_decode(beat_activation: np.ndarray, downbeat_activation: np.ndarray, fps: float) -> dict:
    """Wrap madmom's DBNBeatTrackingProcessor / DBNDownBeatTrackingProcessor.

    Returns {'beats': ..., 'downbeats': ...} in seconds.
    """
    raise NotImplementedError
