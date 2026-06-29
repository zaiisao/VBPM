"""Train the bar-pointer DVAE on cached frontend features.

Runs the VBPM model by default; pass --divergence_* flags to enable controlled departures (see config.py).
Example (default):                python train.py
Example (the working synthesis):   python train.py --divergence_sawtooth_weight 0.5 \
                                       --divergence_tempo_source autocorr --divergence_phase_update filter
"""
from __future__ import annotations

import random

import numpy as np
import torch

from config import Config, parse_config
from data.dataset import load_songs, sample_training_batch
from losses import compute_loss
from model.bar_pointer_vae import BarPointerVAE
from evaluate import evaluate_with_leak_test


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def apply_beat_dropout(beat_targets: torch.Tensor, downbeat_targets: torch.Tensor, dropout_probability: float):
    """Hide the beats from the ENCODER (not the loss) on a random subset of sequences.

    Returns the (possibly zeroed) beat/downbeat channels that the encoder reads. The reconstruction loss
    still uses the full targets -- this only weakens the posterior's teacher-forcing.
    """
    if dropout_probability <= 0.0:
        return beat_targets, downbeat_targets
    keep = (torch.rand(beat_targets.shape[0], 1, device=beat_targets.device) >= dropout_probability).float()
    return beat_targets * keep, downbeat_targets * keep


def train(config: Config) -> BarPointerVAE:
    set_all_seeds(config.seed)
    train_songs = load_songs(config.train_feature_dir, config.num_train_songs, seed=1)
    val_songs = load_songs(config.val_feature_dir, config.num_val_songs, seed=2)
    print(f"[train] vbpm_default={config.is_default_vbpm} | train={len(train_songs)} val={len(val_songs)}", flush=True)

    model = BarPointerVAE(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for step in range(1, config.num_steps + 1):
        features, beat_targets, downbeat_targets = sample_training_batch(
            train_songs, config.crop_length_frames, config.batch_size, config.device)
        encoder_beats, encoder_downbeats = apply_beat_dropout(beat_targets, downbeat_targets, config.divergence_beat_dropout)

        rollout = model.rollout(features, encoder_beats, encoder_downbeats,
                                gumbel_temperature=config.gumbel_temperature(step), sample=True, compute_kl=True)
        loss, info = compute_loss(rollout, beat_targets, downbeat_targets, config)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        if step % 200 == 0 or step == config.num_steps:
            real = evaluate_with_leak_test(model, val_songs, config)["real"]
            terms = " ".join(f"{k} {v:.2f}" for k, v in info.items())
            print(f"  step {step:5d} | {terms} | GEOM beat {real['beat_f']:.3f} db {real['downbeat_f']:.3f}", flush=True)

    leak = evaluate_with_leak_test(model, val_songs, config)
    print("\n[final] geometric read-out (decoder discarded):", flush=True)
    for condition in ("real", "shuffle", "zero"):
        print(f"  {condition:8s}: beat {leak[condition]['beat_f']:.3f}  downbeat {leak[condition]['downbeat_f']:.3f}", flush=True)
    print("  (real high + shuffle/zero collapsed => the read-out genuinely tracks the audio)", flush=True)

    if config.save_path:
        torch.save({"model": model.state_dict(), "config": vars(config)}, config.save_path)
        print(f"[train] saved -> {config.save_path}", flush=True)
    return model


if __name__ == "__main__":
    train(parse_config())
