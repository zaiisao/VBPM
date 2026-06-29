"""The training objective: the VBPM ELBO, plus the optional divergence loss terms.

The VBPM loss = reconstruction (Bernoulli NLL of beats/downbeats under the decoder) + the three KLs
(meter, phase, tempo), i.e. the negative ELBO with beta = 1. The divergence flags add/modify terms:
  * free_bits    -> floor each KL (anti-collapse), departing from strict ELBO.
  * pos_weight   -> reweight the reconstruction BCE (departs from a plain Bernoulli likelihood).
  * sawtooth     -> add lambda * (1 - cos(phi - phi_ground_truth)) phase supervision.
  * tempo_source=autocorr -> add cross-entropy training the autocorrelation tempo head against GT period.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from model.bar_pointer_vae import RolloutResult
from model.divergences import AUTOCORR_LAG_FRAMES
from data.targets import build_sawtooth_phase_target_batch


def _autocorr_target_lag_indices(beat_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each example, the index (into AUTOCORR_LAG_FRAMES) of the lag nearest the GT beat period."""
    beat_numpy = beat_targets.detach().cpu().numpy()
    target_indices, valid = [], []
    for example in beat_numpy:
        beat_frames = np.where(example > 0.5)[0]
        if len(beat_frames) < 2:
            target_indices.append(0)
            valid.append(0.0)
        else:
            period = float(np.median(np.diff(beat_frames)))
            target_indices.append(int(np.argmin(np.abs(AUTOCORR_LAG_FRAMES - period))))
            valid.append(1.0)
    device = beat_targets.device
    return (torch.tensor(target_indices, device=device), torch.tensor(valid, device=device))


def compute_loss(rollout: RolloutResult, beat_targets: torch.Tensor, downbeat_targets: torch.Tensor,
                 config: Config) -> tuple[torch.Tensor, dict]:
    """Return (scalar loss, info dict of per-term means) for one batch."""
    num_frames = beat_targets.shape[1]

    # ---- reconstruction (Bernoulli NLL of the decoder) ----
    positive_weight = torch.tensor([config.divergence_beat_pos_weight, config.divergence_downbeat_pos_weight],
                                   device=beat_targets.device)
    reconstruction = F.binary_cross_entropy_with_logits(
        rollout.decoder_logits, torch.stack([beat_targets, downbeat_targets], dim=-1),
        pos_weight=positive_weight, reduction="none",
    ).sum(dim=(1, 2))

    # ---- KL terms (each per-sequence KL floored at free_bits * num_frames nats; free_bits=0 -> strict ELBO) ----
    free_bits_floor = config.divergence_free_bits * num_frames
    total = reconstruction.clone()
    info = {"recon": float(reconstruction.mean())}
    for name, kl_term in (("kl_meter", rollout.kl_meter), ("kl_phase", rollout.kl_phase), ("kl_tempo", rollout.kl_tempo)):
        if kl_term is not None:
            floored = kl_term.clamp(min=free_bits_floor)
            total = total + floored
            info[name] = float(floored.mean())

    # ---- divergence: sawtooth phase supervision ----
    if config.divergence_sawtooth_weight > 0.0:
        phase_target, valid_mask = build_sawtooth_phase_target_batch(beat_targets, downbeat_targets, config.beats_per_bar)
        per_frame = (1.0 - torch.cos(rollout.phase - phase_target)) * valid_mask
        sawtooth_loss = per_frame.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        # scaled by num_frames so its magnitude is comparable to the summed KL/recon terms
        total = total + config.divergence_sawtooth_weight * num_frames * sawtooth_loss
        info["sawtooth"] = float(sawtooth_loss.mean())

    loss = total.mean()

    # ---- divergence: train the autocorrelation tempo head (separate cross-entropy, not in the ELBO) ----
    if config.divergence_tempo_source == "autocorr" and rollout.tempo_lag_scores is not None:
        target_lag_index, valid = _autocorr_target_lag_indices(beat_targets)
        cross_entropy = F.cross_entropy(rollout.tempo_lag_scores, target_lag_index, reduction="none") * valid
        tempo_loss = cross_entropy.sum() / valid.sum().clamp(min=1.0)
        loss = loss + tempo_loss
        info["tempo_ce"] = float(tempo_loss)

    return loss, info
