"""Sinusoidal positional encoding."""
import math

import torch


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding [length, dim]."""
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                      * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * scale)
    pe[:, 1::2] = torch.cos(pos * scale)
    return pe
