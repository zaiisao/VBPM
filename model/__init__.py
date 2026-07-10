"""Model layer: the bar-pointer conditional DMM, its latent families, read-outs, and the filter."""
from .bar_pointer_vae import VariationalBarPointerModel, RolloutResult
from . import latents, readout, particle_filter

__all__ = ["VariationalBarPointerModel", "RolloutResult", "latents", "readout", "particle_filter"]
