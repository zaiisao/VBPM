"""The training objective: the VBPM negative ELBO with free bits, plus the auxiliary EMISSION terms
(meter CE, sawtooth phase, tempo slope). Every auxiliary is a likelihood term on an observed
quantity -- an emission, not a penalty -- so the objective stays a valid (tempered) ELBO.

Pinned recipe (2026-07-10 minimality ladder, docs/emission_sidechannel_report.md): parametric
emission + meter CE + prior-preserving free bits + ONE tempo-grounding emission (tempo slope by
default; sawtooth optional -- NOSAW verdict). With neither grounding term the phase/tempo latents
collapse (MIN_neither: 0.367/0.326).
"""
import math

import torch
import torch.nn.functional as F

from data.targets import (TWO_PI, build_sawtooth_phase_targets, build_tempo_slope_targets,
                          crop_beats_per_bar_classes)


def negative_elbo_terms(rollout, beat_targets, downbeat_targets, free_bits_nats_per_frame=0.0,
                        prior_preserving=False, meter_ce_weight=0.0):
    """Reconstruction + the three floored KLs (+ optional meter emission). Returns ([batch] negative
    ELBO, dict of scalar term means for logging -- the reported means stay raw, so any collapse
    remains visible in the logs even when the floor hides it from the optimizer)."""
    # Reconstruction: -log p(b | z) = BCE with the decoder logits, summed over frames and channels.
    reconstruction_nats = F.binary_cross_entropy_with_logits(
        rollout.event_logits, torch.stack([beat_targets, downbeat_targets], dim=-1),
        reduction="none").sum(dim=(1, 2))                                            # [batch]
    # Free bits (Kingma et al. 2016): floor each latent group's KL so the optimizer cannot profit
    # from collapsing it to zero. 0.0 = the strict ELBO.
    kl_floor = free_bits_nats_per_frame * beat_targets.shape[1]
    negative_elbo = (reconstruction_nats + rollout.kl_meter.clamp(min=kl_floor)
                     + rollout.kl_phase.clamp(min=kl_floor) + rollout.kl_tempo.clamp(min=kl_floor))
    if prior_preserving and rollout.kl_phase_pg is not None:
        # Prior-preserving free bits: the clamp above protects the POSTERIOR from collapse but has
        # zero gradient below the floor -- which starves every KL-only-trained prior network
        # (initial head, scale heads, meter transition, correction head). These value-0 terms
        # restore the full prior-side gradient without changing any loss value or the clamp's
        # effect on q. (Same spirit as DreamerV2's KL balancing: the prior always learns toward q.)
        for pg in (rollout.kl_meter_pg, rollout.kl_phase_pg, rollout.kl_tempo_pg):
            negative_elbo = negative_elbo + (pg - pg.detach())
    term_means = {"reconstruction": float(reconstruction_nats.mean()),
                  "kl_meter": float(rollout.kl_meter.mean()),
                  "kl_phase": float(rollout.kl_phase.mean()),
                  "kl_tempo": float(rollout.kl_tempo.mean())}
    if meter_ce_weight > 0.0 and rollout.meter_logits is not None:
        # Meter emission (semi-supervised, Kingma M2 style): the song's annotated beats-per-bar M
        # is an OBSERVED variable with categorical emission p(M | m_t); this is its frame-summed
        # NLL. Units lesson (SymbTr campaign): frame-SUMMED like every sibling term -- the weight
        # is nats/frame; 0.1 broke the dead all-4/4 equilibrium reliably.
        num_meters = rollout.meter_logits.shape[-1]
        target_classes, valid_mask = crop_beats_per_bar_classes(
            beat_targets, downbeat_targets, num_meters)
        num_frames = rollout.meter_logits.shape[1]
        per_frame_ce = F.cross_entropy(
            rollout.meter_logits.permute(0, 2, 1),
            target_classes.unsqueeze(1).expand(-1, num_frames), reduction="none")   # [batch, frames]
        meter_ce = (per_frame_ce.mean(dim=1) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        negative_elbo = negative_elbo + meter_ce_weight * num_frames * meter_ce
        term_means["meter_ce"] = float(meter_ce)
    return negative_elbo, term_means


def auxiliary_emission_terms(rollout, beat_targets, downbeat_targets, sawtooth_weight=0.0,
                             tempo_slope_weight=0.0, sawtooth_family="von_mises",
                             sawtooth_wc_rho=0.7, target_beats_per_bar=4):
    """Sawtooth phase emission and/or tempo-slope emission (each independent: the ramp targets are
    built whenever EITHER weight is positive -- the slope term does NOT require the sawtooth term).
    Returns ([batch]-broadcastable scalar loss contribution, dict of term means)."""
    loss = 0.0
    term_means = {}
    if sawtooth_weight <= 0.0 and tempo_slope_weight <= 0.0:
        return loss, term_means
    phase_target, valid_mask = build_sawtooth_phase_targets(
        beat_targets, downbeat_targets, beats_per_bar=target_beats_per_bar)
    num_frames = beat_targets.shape[1]
    if sawtooth_weight > 0.0:
        angular_error = rollout.bar_phase - phase_target
        if sawtooth_family == "wrapped_cauchy":
            # WC emission NLL (shifted so 0 at zero error): heavy tails forgive the frames where
            # the linear-interpolation ramp is a WRONG target (rubato/microtiming) while pulling
            # as strongly as vM near the ramp.
            rho = sawtooth_wc_rho
            circular_error = (torch.log(1.0 + rho ** 2 - 2.0 * rho * torch.cos(angular_error))
                              - 2.0 * math.log(1.0 - rho)) * valid_mask
        else:
            # von Mises emission NLL (up to constants): the 1 - cos form.
            circular_error = (1.0 - torch.cos(angular_error)) * valid_mask
        sawtooth_per_sequence = circular_error.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        # scaled by the frame count so the term is commensurate with the frame-summed ELBO terms
        loss = loss + sawtooth_weight * num_frames * sawtooth_per_sequence.mean()
        term_means["sawtooth"] = float(sawtooth_per_sequence.mean())
    if tempo_slope_weight > 0.0:
        # Tempo-slope emission: supervise the TEMPO latent with the ramp's own finite difference --
        # Laplace NLL (L1), matching the tempo family.
        target_log_advance, slope_valid = build_tempo_slope_targets(phase_target, valid_mask)
        tempo_l1 = (rollout.log_tempo[:, 1:] - target_log_advance).abs() * slope_valid
        tempo_slope_per_sequence = tempo_l1.sum(dim=1) / slope_valid.sum(dim=1).clamp(min=1.0)
        loss = loss + tempo_slope_weight * num_frames * tempo_slope_per_sequence.mean()
        term_means["tempo_slope"] = float(tempo_slope_per_sequence.mean())
    return loss, term_means
