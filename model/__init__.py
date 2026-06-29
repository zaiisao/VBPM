"""Model layer: the bar-pointer DVAE, its latents, the geometric read-out, and divergence modules."""
from .bar_pointer_vae import BarPointerVAE, RolloutResult
from . import latents, readout, divergences

__all__ = ["BarPointerVAE", "RolloutResult", "latents", "readout", "divergences"]
