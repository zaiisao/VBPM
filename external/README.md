# external/ — vendored third-party code

Code here is borrowed from upstream projects. We prefer calling upstream code directly over
reimplementing it. Each entry records exactly where it came from and any local changes.

## beat_this/
- **Upstream:** "Beat This! Accurate beat tracking without DBN postprocessing" (Foscarin, Schlüter,
  Widmer, ISMIR 2024) — https://github.com/CPJKU/beat_this
- **Why vendored:** it is our feature extractor. We use the 512-dim penultimate representation
  `transformer_blocks(frontend(spectrogram))` (everything Beat This computes *before* collapsing to its
  two beat/downbeat logits) as the input features for the bar-pointer DVAE. See
  `data/feature_extractor.py` (`BeatThisFeatureExtractor`).
- **Local modifications:** none. We only *call* the upstream model; we do not edit its source. If a
  future change becomes necessary, follow the repo rule: keep the original line commented out directly
  above the change, with a second comment marking it as ours, e.g.
  ```python
  # original: head_dim = 32
  head_dim = 64  # VBPM: widened for ...   <-- ours
  ```

## Not vendored here (available in the external archive)
The following reference repos were used during exploration but are **not needed** by the VBPM codebase.
boilerplate. They live in the archived tree (see the Claude memory note `project_codebase_refactor`):
- `kalman-vae` — https://github.com/nkiyohara/kalman-vae (exact differentiable Kalman filter)
- `kvae` — https://github.com/simonkamronn/kvae (original Kalman-VAE)
- `Joint-beat-and-downbeat-estimation` — https://github.com/Tsung-Ping/Joint-beat-and-downbeat-estimation
  (Chen & Su 2022; source of the sawtooth/label-embedding phase target)
