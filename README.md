# VBPM — phase-only bar-pointer VAE (branch `phase-min`)

This branch is the phase-only bar-phase VAE: one continuous latent (bar phase, one turn
per bar), the bar period given per crop, downbeats read off the phase trajectory by the
deterministic rule g. Architecture follows the tutorial's §7 configuration (encoder-only
deployment, fixed physical prior, no psi); every knob's measured rationale is in
`docs/vbpm_decisions.md`.

Stage-0 (meter latent) and the earlier campaign surfaces are deleted here — recover them
from `master` or git history. v1 lives in `vbpm-campaign-2026-07-26`.

## Layout

| path | what |
|---|---|
| `vbpm/` | the model (`model.py`), training CLI (`run.py`), the config schema (`config.py` + `vbpm/config_schema.json`), hooks modules (`variants/`), data (`data/`), pre-flight controls and scoring (`scoring/`), tests (`tests/`) |
| `docs/vbpm_decisions.md` | measured rationale behind every flag and recorded deviations |
| `vbpm/data.py` | fold-honest frontend feature pass (the single authority) |
| `vbpm/data/songs.py` | Beat This annotation catalog and 8-fold splits |
| `frontends/` | Beat This / Beat Transformer wrappers over `external/` submodules |
| `logs/vbpm/` | full training logs of the 2026-08 campaign |

## Run

    PYTHONPATH=. python train.py --config vbpm/configs/anchor_k.yaml --gpu 1

The recipe is the config's business; the CLI carries only run mechanics (device, seed,
paths). Override one key for one run with `--set`, repeatable:

    PYTHONPATH=. python train.py --config vbpm/configs/baseline.yaml \
        --set epochs=2 --set emission=cosine --save-dir checkpoints/<name>

Every mainline key, its default, its type and why it has that value:
`vbpm/config_schema.json` (a variant's extra keys are in its own module's `DEFAULTS`).
Anything else in a config -- or a value of the wrong type, or outside the declared range --
refuses at parse time.

## MusicFM frontend

The MusicFM encoder and pretrained checkpoint loader remain available. MusicFM training
requires normalized 100 fps MusicFM mel inputs. The supplied Beat This spectrograms are
50 fps, so the conversion and data-loading path for MusicFM inputs still needs to be added.
Use `beat_this` for training with the currently wired Beat This spectrogram bundles.

## Beat This spectrogram data

The default Beat This frontend reads Beat This's supplied, memory-mapped spectrogram
bundles from `/disk4/shared/beat_this/data/audio/spectrograms/<dataset>.npz`. The matching
annotation repository is pinned as `external/beat_this_annotations`. Training uses these
spectrograms directly, so local audio is not required for Beat This datasets. Install the
project dependencies with `python -m pip install -e .` to include Beat This's dataset-loader
dependencies.

    PYTHONPATH=. python -m pytest tests -q     # 69 tests, CPU, ~2 s
