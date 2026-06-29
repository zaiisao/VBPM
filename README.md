# VBPM — Variational Bar Pointer Model for beat & downbeat tracking

A minimal, readable implementation of the generative **bar-pointer DVAE** derived from the *ELBO for
DBN* paper, plus controlled **divergence flags** that ablate it to study *why* each design choice matters.

VBPM factorizes a song into three per-frame latents and reads beats/downbeats out of them
geometrically (no DBN, decoder discarded at inference):

| latent | distribution | meaning |
|--------|--------------|---------|
| meter `m` | Categorical | beats per bar |
| bar phase `phi` | von Mises | angular position in the bar (wraps once per bar = a downbeat) |
| log tempo `s` | Normal (LogNormal tempo) | bar-phase advance per frame |

Generative dynamics: `phi_t ~ vM(phi_{t-1} + exp(s_{t-1}), kappa)`, `s_t ~ N(s_{t-1}, sigma)`.
At inference we **throw away the decoder** and piece beats/downbeats together from `phi`
(`beats = (M·phi) wraps`, `downbeats = phi wraps`).

## Layout
```
train.py            # entry: training loop (runs the VBPM model by default)
evaluate.py         # entry: geometric read-out from z + leak test (real / shuffle / zero)
losses.py           # ELBO (recon + KLs) + optional divergence losses
config.py           # all flags; DEFAULTS = the VBPM model
model/
  bar_pointer_vae.py  # encoder, prior dynamics, rollout, decoder
  latents.py          # von Mises / Normal / Categorical sampling + KL
  readout.py          # phase -> beat/downbeat times; peak-pick; F-measure
  divergences.py      # autocorrelation tempo head, geometric emission, Kalman phase filter
data/
  dataset.py          # cached-feature loading + batch sampling
  targets.py          # ground-truth beats/tempo + sawtooth phase target
  feature_extractor.py# modular frontend (cached default; Beat This for end-to-end)
external/             # vendored upstream code (see external/README.md for provenance)
cache/                # cached frontend features (gitignored, regenerable)
```

## Running
```bash
python train.py                      # the VBPM model
python train.py --num_steps 1500 \   # the working synthesis (each flag = one divergence)
    --divergence_sawtooth_weight 0.5 \
    --divergence_tempo_source autocorr \
    --divergence_phase_update filter
```

## Ablation flags (all OFF by default → the VBPM model)
Each flag isolates one departure so we can attribute behaviour to it:

| flag | default | what turning it on does |
|------|------------------|--------------------------|
| `--divergence_phase_update` | `free` | `integrator` (phi=∫tempo) or `filter` (predict+correct) |
| `--divergence_tempo_source` | `latent` | `autocorr`: compute tempo from features instead of inferring it |
| `--divergence_decoder` | `mlp` | `geometric`: likelihood `beat~cos(M·phi)`, `downbeat~cos(phi)` |
| `--divergence_sawtooth_weight` | `0.0` | add sawtooth phase supervision |
| `--divergence_free_bits` | `0.0` | floor each KL (anti-collapse) |
| `--divergence_beat_pos_weight` / `_downbeat_pos_weight` | `1.0` | reweight the reconstruction BCE |
| `--divergence_beat_dropout` | `0.0` | hide beats from the encoder with this probability |
| `--divergence_meter` | `latent` | `fixed`: hard-set meter to `beats_per_bar` |
| `--divergence_end_to_end` | off | unfreeze the Beat This frontend (audio pipeline) |

The decoder is used only as the training likelihood; **inference always reads beats from `phi`**.
