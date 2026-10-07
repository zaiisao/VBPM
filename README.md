# VBPM

Work in this checkout on `fix/gsnn-generator`, based on `archain-anchor`
commit `6bf398a`. The original Git history and the MusicFM experiment branch
are retained. Prior, posterior, and decoder implementation lives in `vbpm/`.

Use `/disk4/anaconda3/envs/vbpm/bin/python` or activate the `vbpm` environment.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
PYTHONPATH=.:external/beat_this python train.py --config vbpm/configs/baseline.yaml --gpu 0 --seed 0 --save-dir runs/new_training
```

`baseline.yaml` uses 70% CVAE and 30% GSNN. `gsnn.yaml` uses GSNN only.
These recipes are experimental; prior-only generalization is unresolved.
The actual posterior still uses prior-relative corrections, whose direct KL
teaching gradients require investigation. No posterior repair was promoted
into production during the handover.

Reusable experiments are in `diagnostics/`; use `python -m` and `--help`:

| Module | Purpose |
|---|---|
| `synthetic_ladder` | `phase`, `tempo`, `sparse`, and `frames` synthetic stages |
| `cache_audio` | Frozen audio caches; explicit exclusions with `--exclude-cache` |
| `audio_generator` | GSNN phase/tempo training and controlled variants |
| `phase_hybrid` | Matched phase-only hybrid training |
| `readouts`, `acceptance` | Prior/decoder event evaluation and ±70 ms gates |
| `likelihood` | Prior versus hybrid conditional likelihood |
| `posterior_labels` | One experiment runner with `copied`, `residual`, `broad`, or `circular` recognition |
| `posterior_audit` | `origin`, `penalty`, and `saturation` diagnostics |
| `probe_bar_origin_candidates`, `probe_crop_origin_consistency` | Origin likelihood and crop consistency |

`emissions`, `synthetic_reference`, `audio_metrics`, and
`real_event_references` provide shared implementations.

Cached inputs, checkpoints, and numerical results from the separate worktree
are preserved in `runs/`. Original per-run source snapshots remain there as
experiment provenance, not active scripts. Old JSON reports may record their
original absolute paths; pass current paths explicitly when running tools.
Recognition controls use labels at evaluation and do not establish prior-only
performance: independent circular recognition improved the four failing
clips' downbeat F1 from zero to 1.00/0.731 across two seeds.

Generated narrative reports and obsolete scripts were removed from the active
checkout. An external `../VBPM_handover_backup_*` archive retains those files,
pre-cleanup diffs, and the earlier synthetic outputs. The retained external
frontend clones include their existing local modifications.
