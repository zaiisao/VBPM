# MusicFM CVAE-DBN experiment

This branch contains the 30-second real-audio experiment and its required
source code. It starts from VBPM's initial commit, `b5af0ed`, with that commit's
boilerplate removed. The experiment is independent of the production VBPM
package and the other experiments.

## Code

| Path | Purpose |
|---|---|
| `cvae_dbn/run.py` | Main runner: extraction/cache, existing debug checks, training and evaluation |
| `cvae_dbn/data.py` | Native audio crops, official song folds, categorical labels and diagnostic references |
| `cvae_dbn/features.py` | Frozen MusicFM extraction; concatenate all 12 Conformer layers |
| `cvae_dbn/model.py` | Per-layer normalization and learned feature projection |
| `cvae_dbn/plot.py` | Training, label scores, phase and velocity plots |
| `cvae_dbn/debug.py` | Per-term gradient routing helper used by the runner |
| `cvae_dbn/reference/vae_dbn.py` | Unmodified tutorial model, distributions, transitions, loss and inference |
| `cvae_dbn/reference/train_logger.py` | Unmodified tutorial gradient/health logger |
| `cvae_dbn/reference/plot_logs.py` | Unmodified tutorial log plotting tool |
| `cvae_dbn/audit_reference.py` | Audit reference computations against PDF appendix extractions or a supplied PDF |
| `vendor/musicfm/` | Required upstream MusicFM source, pinned and copied unchanged with its license |
| `provenance.json` | Original source hashes, initial commit and MusicFM revision |

The reference model still has the learned emission `p(b_t | z_t, h_t)`.
This branch migration introduces no new model mechanism or training objective.

## Experiment settings

- Eight distinct Ballroom songs: four training, four held out.
- Official Beat This fold 7 is held out; both splits have two songs each in
  3/4 and 4/4. This is a small fixed-batch exercise.
- **30 seconds per clip, 750 frames at 25 fps**. MusicFM and the CVAE-DBN process
  the full sequence. No shorter training crops or sequence truncation.
- Frozen float32 MusicFM MSD features: layers 1–12 concatenated into 12,288
  channels per frame; hidden state 0 excluded, no temporal pooling.
- Normalize each layer per frame, then learn a `12288 -> 64` projection.
- Train both projection and CVAE-DBN, with initialization seeds 0 and 1,
  batch size four, 200 Adam updates per seed, learning rate .003.
- Tutorial appendix defaults: alpha .7, beta 1, Gumbel temperature .5.
  Training uses CPU with one thread; feature extraction defaults to CUDA.
- Phase in radians, velocity in radians per 25 Hz frame, transition Delta=1
  frame (.04 seconds). Gaussian velocity is unrestricted, as in the reference.
- Observations are non-beat / ordinary beat / downbeat labels. Phase, velocity
  and meter references from annotations are diagnostic only.

At the recording edges, frames without two surrounding annotated downbeats
have no interpolated phase/velocity reference. `reference_valid` masks those
diagnostics. Training and label/event scoring still use all 750 frames.

## Run

Use the existing environment on this machine:

```sh
cd /home/sogang/jaehoon/VBPM_musicfm_cvae_dbn
/disk4/anaconda3/envs/vbpm/bin/python -m cvae_dbn
```

The ignored local `assets/musicfm/` directory can contain or link to
`pretrained_msd.pt` and `msd_stats.json`. Audio/annotations are external inputs;
their default root is `/disk1/jaehoon/dataset_store`. All resource paths can
be set explicitly:

```sh
python -m cvae_dbn \
  --data-store /path/to/dataset_store \
  --musicfm-weights /path/to/pretrained_msd.pt \
  --musicfm-stats /path/to/msd_stats.json \
  --output outputs/30s
```

For another environment, install `requirements.txt` with compatible Torch and
torchaudio builds. MusicFM's Conformer configuration must be available in the
Hugging Face cache or downloadable. Upstream model information and MSD weight
links: <https://github.com/minzwon/musicfm>.

Use `--extract-only` to prepare the cache, `--rebuild-cache` to regenerate it,
or `--extract-device cpu` for CPU extraction. Reused caches are checked for
their checksum and configured sequence lengths.

Reference audit uses archived code extractions from the original, already
audited PDF. The PDF hash and reference hashes are recorded in
`cvae_dbn/reference/provenance.json`. To audit against the original PDF again:

```sh
python -m cvae_dbn --tutorial-pdf /path/to/CVAE_DBN_Debug_tutorial.pdf
```

That option also requires `pdftotext`. The archived audit does not require the
private PDF or any file from the previous VBPM checkout.

## Outputs and interpretation

Outputs go to the ignored `outputs/30s/` directory:

- `features.pt`, `feature_manifest.json`: exact waveform/feature dimensions,
  all-layer concatenation, crop offsets, resource hashes and diagnostic coverage.
- `report.json`: settings, runtime checks and each completed seed's metrics.
- `seed_*/training.csv`, `.log`, `model.pt`, `step_*.pt`: training and checkpoints.
- `seed_*/training_diagnostics.pt`, `heldout_diagnostics.pt`: references and outputs.
- `training.png`, `label_scores.png`, `heldout_seed_*.png`: final graphs.

Audio-only evaluation averages 64 prior paths. Event F1 uses a fixed .5 peak
threshold, .12-second peak spacing and 70 ms one-to-one matching tolerance.
Report label scores, phase/velocity diagnostics and decoder shuffle probes
separately. A lower loss or accurate training labels alone does not establish
that the latent state represents physical bar phase and tempo.

The original active run continues in its existing checkout. The local cache
can be copied here for reuse; its provenance records where extraction occurred.
This branch contains no historical results or production VBPM code.
