# VBPM

Work in this checkout on `fix/gsnn-generator`, based on `archain-anchor`
commit `6bf398a`. The original Git history and the MusicFM experiment branch
are retained. Model implementation lives in `vbpm/`.

Use `/disk4/anaconda3/envs/vbpm/bin/python` or activate the `vbpm` environment.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
PYTHONPATH=.:external/beat_this python train.py --config vbpm/configs/baseline.yaml --gpu 0 --seed 0 --save-dir runs/new_training
```

`baseline.yaml` and `gsnn.yaml` now use GSNN only, with fixed meter four.
The active model is a fresh implementation of the transition-prior rollout in
`CVAE_DBN_Debug_tutorial.pdf`; it does not use pretrained beat-peak proposals.
`PriorModel` predicts Gaussian velocity and von Mises phase parameters from
frontend features and the previous sampled state. `LatentSampler` draws the
state, and `EmissionModel` predicts labels from phase and velocity only; audio enters through the prior.
Initial phase is uniform. Velocity is in radians per second; frame duration
comes from the frontend. The chain has an unobserved initial state followed by
one transition per label frame; the ELBO includes the initial-state KL.
`PosteriorModel` reads audio and labels with a
bidirectional encoder and conditions on the previous sampled state. A nonzero
`gsnn_alpha` enables posterior reconstruction and phase/velocity KL terms in
the existing hybrid training objective. Defaults remain GSNN only. Training uses full-length crops;
evaluation accepts padded clips.

`train.py --init-from` loads weights saved by this implementation. Earlier
proposal-based experiment checkpoints are incompatible. Historical experiments
and their results remain in `diagnostics/` and `runs/`.

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
