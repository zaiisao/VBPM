"""Adapt MusicFM inputs while inheriting the canonical tutorial computations."""
import torch
from torch import nn

from reference.vae_dbn import VAEDBN


class MusicFMCVAEDBN(VAEDBN):
    def __init__(self, projected_dim=64):
        # The original transition, sampler, phase/velocity/meter heads,
        # categorical emission and hybrid loss are inherited unchanged.
        super().__init__(x_dim=projected_dim, Delta=1.0)
        self.projected_dim = projected_dim
        self.feature_normalization = nn.LayerNorm(1024, elementwise_affine=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11000)
            self.input_projection = nn.Linear(12 * 1024, projected_dim)

    def project_features(self, x):
        if x.shape[-1] != 12288:
            raise ValueError(f'Expected 12 concatenated MusicFM layers, got {x.shape}')
        # Normalize each layer per frame. No validation statistics are fitted.
        normalized = self.feature_normalization(x.reshape(*x.shape[:-1], 12, 1024))
        return self.input_projection(normalized.flatten(-2))

    def backbone_feats(self, x):
        return super().backbone_feats(self.project_features(x))

    def context(self, b, x):
        return super().context(b, self.project_features(x))
