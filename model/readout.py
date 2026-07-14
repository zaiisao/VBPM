"""Turn a decoded state path into musical events."""
import torch


def state_path_to_events(path: torch.Tensor, state_space, fps: float) -> dict:
    """path: [T] long state indices from Viterbi.

    Returns {'beats': ..., 'downbeats': ..., 'tempo': ...} -- beats where the beat phase wraps,
    downbeats where the bar phase wraps, times in seconds via `fps`.
    """
    raise NotImplementedError
