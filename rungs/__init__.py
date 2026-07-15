"""The A -> B -> C ladder. Each rung exposes the same deployment interface:
    decode(...) -> {'beats': seconds, 'downbeats': seconds}
so evaluate.py can score every rung identically.

    r0_madmom_dbn   Baseline A -- literal madmom DBN (black box)          [tractable]
    r1 ...          handcrafted bar-pointer HMM in our framework           [tractable]
    r2 ...          + learn parametric factors by exact forward            [tractable]
    r3 ...          + audio-conditioned scales / meter transitions         [tractable]
    r4 ...          full neural HMM (MLP transition + emission)            [tractable]
    r5 ...          Transformer VAE-DBN (non-Markov prior + emission)      [INTRACTABLE -> ELBO]
"""
from rungs.base import Rung
from rungs.r0_madmom_dbn import MadmomDBN

__all__ = ["Rung", "MadmomDBN"]
