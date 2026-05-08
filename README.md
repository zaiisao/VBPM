# CHART

**Continuous Hierarchical Autoregressive Rhythm Tracker** — a variational bar-pointer model for joint beat/downbeat tracking.

## Method

CHART is a Sequential Variational Transformer (SVT) realisation of the bar-pointer VAE. A bidirectional Transformer prior over audio activations parameterises a structured transition model on (phase φ, log-tempo, meter): von Mises for phase, Log-Normal for tempo, Categorical for meter. A second Transformer posterior is amortised on `[audio, beat_distance, downbeat_distance]`. A small MLP decoder emits 2-channel (beat, downbeat) logits from `[cos φ, sin φ, log_tempo, meter, h_prior]`.

Training optimises the ELBO: BCE-with-logits on Gaussian-smoothed beat/downbeat targets, plus closed-form KLs (Categorical, von Mises with implicit reparameterisation, Gaussian-in-log-space) with optional per-latent free-bits. Inference samples sequentially from the prior alone; beats are read off phase wrap-arounds.

The audio frontend is [WaveBeat](extractors/wavebeat) (Steinmetz & Reiss, AES 2021), vendored under [extractors/wavebeat/](extractors/wavebeat) with local patches for PyTorch Lightning 2.x and dataset-path resolution, and trained jointly in end-to-end mode.

## Setup

```bash
git clone <repo-url> CHART
cd CHART

conda env create -f environment.yml
conda activate chart
```

The pre-trained WaveBeat checkpoint (`wavebeat_epoch=98-step=24749.ckpt`, ~33 MB) is expected at the repo root. If missing, fetch it from Zenodo:

```bash
wget "https://zenodo.org/record/5525120/files/wavebeat_epoch%3D98-step%3D24749.ckpt?download=1" \
    -O wavebeat_epoch=98-step=24749.ckpt
```

## Project layout

```
models/
  svt_core.py         SVTModel: prior + posterior Transformers, parallel forward
  distributions.py    von Mises sampler (TFP-style implicit reparam), KLs, Gumbel-Softmax
  loss.py             ELBO with Gaussian-smoothed BCE and per-latent free-bits
training/
  train.py            entrypoint for both modes; DDP-aware; top-3 ckpt by val F-measure
  dataset.py          ActivationDataset, AudioPhaseBridgeDataset, multi-source aggregator
  extractors/         pluggable frontends (currently: wavebeat)
  phase_generation/   per-dataset beat-annotation -> phase .npy converters
evaluation/
  inference.py        prior-only sampling CLI
  phase_converter.py  peak-pick + phase-wrap beat extraction
  score.py            mir_eval wrappers (F-measure, CMLc/t, AMLc/t)
tests/test_pipeline.py
extractors/wavebeat   vendored WaveBeat package (csteinmetz1/wavebeat + local patches)
```

## Data preparation

### Dataset folder layout

The training pipeline auto-discovers up to six datasets under a single root. Either layout works:

```
<dataset_root>/<key>/{data,label}/
<dataset_root>/labeled_data/<key>/{data,label}/
```

where `<key>` is one of `ballroom, beatles, beatles_old, gtzan, hains, rwc_popular`. `data/` holds `.wav` files; `label/` holds annotations (`.beats`, `.txt`, or `.BEAT.TXT` depending on the dataset).

### Generate phase targets (required before training)

Convert beat annotations to per-frame phase/tempo/meter targets at 100 fps:

```bash
python -m training.phase_generation.all_datasets \
    --root <dataset_root> --output_mode inside --fps 100
```

This writes `<dataset_root>/<key>/phases/<song>.npy` with shape `[T, 4]` and columns `[tempo_bpf, beat_phase∈[0,1), bar_phase∈[0,1), meter_class]`. Pass `--output_mode separate --out_root <dir>` to mirror to a sibling tree.

Per-dataset entrypoints exist too: `python -m training.phase_generation.{ballroom,beatles,gtzan,hains,rwc_popular} --root <root>`.

## Training

### End-to-end mode (recommended)

Audio → WaveBeat → SVT, jointly trained. With dataset auto-discovery:

```bash
python -m training.train --mode end2end --extractor wavebeat \
    --dataset_root <dataset_root> \
    --extractor_ckpt wavebeat_epoch=98-step=24749.ckpt \
    --batch_size 16 --num_epochs 100 --lr 1e-4 \
    --save_ckpt_path checkpoints/chart_run.pt
```

Restrict datasets with `--dataset_include ballroom,beatles` (default: `ballroom,beatles,gtzan,hains,rwc_popular`; `all` includes every supported key). Freeze the audio frontend with `--freeze_extractor` to train only the SVT.

### Activation mode

If you have pre-extracted activations of shape `[T, 2]` and matching phase `.npy`:

```bash
python -m training.train --mode activation \
    --activations_dir <dir> --phases_dir <dir> \
    --batch_size 16 --num_epochs 50 --lr 1e-4 \
    --save_ckpt_path checkpoints/chart_act.pt
```

Activations and phase files are matched by basename.

### Distributed training

Launch with `torchrun`; the trainer initialises NCCL, keeps the `DistributedSampler` epoch in sync, and runs validation/visualisation/checkpointing on rank 0 only:

```bash
torchrun --nproc_per_node=4 -m training.train --mode end2end ...
```

### Useful flags

| Flag | Purpose |
| --- | --- |
| `--free_bits_meter/_phase/_tempo` | Per-latent free-bits floor (nats) — primary lever against posterior collapse |
| `--smooth_sigma`, `--smooth_sigma_db` | Gaussian kernel σ (frames) for beat/downbeat target smoothing |
| `--bce_pos_weight` | BCE positive-class weight (default 20.0) |
| `--gumbel_temp_start/_end`, `--kl_anneal_epochs` | Annealing schedules |
| `--num_meter_classes` | Categorical K (default 8) |
| `--max_grad_norm` | Gradient clipping (default 1.0) |
| `--examples_per_epoch`, `--train_length`, `--audio_sample_rate`, `--target_factor` | WaveBeat-side dataloader settings |
| `--no_wandb` | Disable Weights & Biases logging |

The trainer keeps the **top-3 checkpoints by validation beat F-measure**, named `<stem>_ep<epoch>_f<F>.pt`, and saves a per-epoch diagnostic panel to `<ckpt_dir>/viz/beat_viz_ep<epoch>.png` (probability tracks, prior vs posterior phase, BPM, per-frame KL, prior κ/σ).

NaN/Inf losses and gradients are detected before the optimiser step; offending batches are skipped and a one-time diagnostic dump is printed.

The WaveBeat backend uses the **paper-correct values from the WaveBeat README** (`audio_sample_rate=22050`, `train_length=2097152`, `target_factor=256`), not the WaveBeat code defaults — see [training/extractors/wavebeat_backend.py](training/extractors/wavebeat_backend.py).

## Inference

Run prior-only sampling on pre-extracted WaveBeat activations:

```bash
python -m evaluation.inference \
    --checkpoint checkpoints/chart_run.pt \
    --input_npy /path/to/activations.npy \
    --output_npy /path/to/beats.npy
```

- **Input**: `.npy` of shape `[T, 2]` or `[B, T, 2]` (sigmoid of WaveBeat dsTCN logits at 86.13 fps by default).
- **Output**: 1-D `np.ndarray` of beat timestamps in seconds at `--output_npy`, plus a sibling `<output>.trajectories.npz` containing `beat_times, phase, tempo, meter, beat_probs`.
- `--temperature` controls the Gumbel-Softmax for meter (low ⇒ more discrete; default 0.1).
- `--fps` must match the activation frame rate (default 86.1328125 = 22050 / 256).

The CLI loads only the SVT weights from a saved end-to-end checkpoint; WaveBeat activations must be produced separately (e.g. by running the WaveBeat dsTCN saved under `extractor_model` in the same checkpoint).

## Evaluation

`mir_eval` beat/downbeat metrics (`F-measure, CMLc, CMLt, AMLc, AMLt`) are computed every validation epoch by [evaluation/score.py](evaluation/score.py). Both decoder-based peak-picked beats and phase-wrap-derived beats are scored, so you can compare the two read-out paths directly in W&B.

## Tests

```bash
pytest tests/test_pipeline.py
```

Covers KL identities, von Mises sampling at extreme κ, log-normal stability at the σ-clamp boundary, gradient flow through every module, and a 100-iteration NaN soak.
