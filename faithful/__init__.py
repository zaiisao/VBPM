"""Faithful, from-scratch implementation of the variational bar-pointer model
described in *ELBO for DBN* (Jaehoon Ahn).

This package is the production-scale counterpart of ``notebooks/ELBO_for_DBN.ipynb``:
the notebook is the line-by-line reference on a toy sequence; this package wires the
*same* strict-ELBO model to the real datasets and trains it END-TO-END FROM RANDOM
WEIGHTS (no frozen Beat This frontend, no pretrained anything).

Faithfulness contract (see ``faithful/README.md`` for the full mapping to the paper):
  * objective = strict ELBO:  L = sum_t BCE(b_t) + sum_t [KL_meter + KL_phase + KL_tempo]
  * beta = 1 from step 0, NO free-bits, NO KL annealing
  * decoder p_theta(b_t | z_t, h) reads the audio h (paper §5.4); the optional
    ``--latent_only`` flag drops h from the decoder as a DOCUMENTED DEVIATION
  * prior means are the deterministic bar-pointer dynamics (phi_{t-1}+phidot_{t-1},
    log-tempo random walk) with NO audio-driven correction
  * three latents only: meter m (Categorical), phase phi (von Mises), log-tempo (Log-Normal)
  * NO latent supervision, NO scheduled sampling, NO tempo clamps, NO pos_weight,
    NO extra latents (no bar-phase, no tau_bar), NO delta-VAE / DVBF

Anything that deviates from the above is, by definition, not faithful and must be
named and justified -- this is the reference the rest of the project is measured against.
"""
