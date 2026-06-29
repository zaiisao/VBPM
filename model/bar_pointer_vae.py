"""The VBPM bar-pointer Dynamical VAE (and the switch-points for its ablations).

Generative story (the prior), per frame t:
    meter      m_t        ~ Categorical(transition from m_{t-1})
    log tempo  s_t        ~ Normal(s_{t-1}, sigma)                 # random walk on log tempo
    bar phase  phi_t      ~ vonMises(phi_{t-1} + exp(s_{t-1}), kappa)  # phase advances by the tempo
    decoder    b_t,db_t   ~ Bernoulli( decode(z_t) )

Inference (the posterior) reads the audio features h to propose the latents each frame. The VBPM
posterior reads the phase directly from h ("free"); the KL to the dynamics prior is what (softly) ties
the phase to the tempo. The divergence switches replace pieces of this with computed/filtered variants.

At deployment we discard the decoder and read beats/downbeats geometrically from the phase (readout.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .latents import (
    sample_von_mises, kl_von_mises,
    sample_normal, kl_normal,
    sample_gumbel_softmax, kl_categorical,
)
from .divergences import AutocorrelationTempoHead, GeometricEmission, blend_phase

TWO_PI = 2.0 * math.pi


@dataclass
class RolloutResult:
    phase: torch.Tensor                       # [batch, num_frames] posterior bar phase (the deploy signal)
    log_tempo: torch.Tensor                   # [batch, num_frames] log bar-phase advance per frame
    decoder_logits: torch.Tensor             # [batch, num_frames, 2] (beat, downbeat) training logits
    kl_meter: torch.Tensor | None            # [batch] summed over frames, or None when not computed
    kl_phase: torch.Tensor | None
    kl_tempo: torch.Tensor | None            # None when tempo is computed (autocorr) rather than a latent
    tempo_lag_scores: torch.Tensor | None    # [batch, num_lags] autocorr scores, for the tempo CE loss


class BarPointerVAE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        feature_dim, hidden, num_meters = config.feature_dim, config.hidden_size, config.num_meters

        # The "latent feature" handed to the decoder / next-step posterior: (cos phi, sin phi, log_tempo, meter).
        self.latent_feature_dim = 3 + num_meters
        # Packed posterior/prior parameters: meter logits, phase (cos,sin), phase concentration,
        # log-tempo mean, log-tempo std (raw, pre-softplus).
        self.packed_param_dim = num_meters + 5

        # Posterior encoder reads features plus the (possibly dropped-out) beat/downbeat channels.
        self.posterior_gru = nn.GRU(feature_dim + 2, hidden, batch_first=True, bidirectional=True)
        self.posterior_context = nn.Linear(2 * hidden, hidden)
        # Audio-conditioned prior encoder (reads features only).
        self.prior_gru = nn.GRU(feature_dim, hidden, batch_first=True, bidirectional=True)
        self.prior_context = nn.Linear(2 * hidden, hidden)

        self.posterior_head = nn.Sequential(
            nn.Linear(hidden + self.latent_feature_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, self.packed_param_dim),
        )
        self.prior_initial = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, self.packed_param_dim),
        )
        self.prior_phase_concentration = nn.Linear(hidden, 1)   # data-conditioned kappa of the phase prior
        self.prior_tempo_std = nn.Linear(hidden, 1)             # data-conditioned sigma of the tempo prior
        self.meter_transition = nn.Sequential(
            nn.Linear(num_meters + 4 + hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, num_meters * num_meters),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, 2),
        )
        self.initial_latent_feature = nn.Parameter(torch.zeros(self.latent_feature_dim))

        # ---- divergence modules (only instantiated when the corresponding flag is set) ----
        self.tempo_head = (
            AutocorrelationTempoHead(feature_dim) if config.divergence_tempo_source == "autocorr" else None
        )
        self.geometric_emission = (
            GeometricEmission() if config.divergence_decoder == "geometric" else None
        )
        self.filter_gain = (
            nn.Parameter(torch.tensor(0.0)) if config.divergence_phase_update == "filter" else None
        )

    # ---- small helpers -------------------------------------------------------------------------

    def _unpack(self, packed: torch.Tensor):
        """Split a packed parameter vector into (meter_logits, phase_mean, phase_conc, tempo_mean, tempo_std)."""
        num_meters = self.config.num_meters
        meter_logits = packed[:, :num_meters]
        # Phase mean is parameterized as an unconstrained (cos, sin) pair; atan2 recovers the angle,
        # which sidesteps any 0/2*pi wrap discontinuity a direct angle output would have.
        phase_mean = torch.atan2(packed[:, num_meters + 1], packed[:, num_meters]) % TWO_PI
        phase_concentration = F.softplus(packed[:, num_meters + 2]) + 0.01
        log_tempo_mean = packed[:, num_meters + 3]
        log_tempo_std = F.softplus(packed[:, num_meters + 4]) + 1e-3
        return meter_logits, phase_mean, phase_concentration, log_tempo_mean, log_tempo_std

    def _latent_feature(self, meter: torch.Tensor, phase: torch.Tensor, log_tempo: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.cos(phase).unsqueeze(-1), torch.sin(phase).unsqueeze(-1), log_tempo.unsqueeze(-1), meter], dim=-1
        )

    def _meter_prior_logprob(self, previous_meter, phase, previous_phase, prior_context):
        features = torch.cat([
            previous_meter,
            torch.cos(phase).unsqueeze(-1), torch.sin(phase).unsqueeze(-1),
            torch.cos(previous_phase).unsqueeze(-1), torch.sin(previous_phase).unsqueeze(-1),
            prior_context,
        ], dim=-1)
        transition = F.softmax(self.meter_transition(features).reshape(-1, self.config.num_meters, self.config.num_meters), dim=2)
        return torch.log(torch.bmm(previous_meter.unsqueeze(1), transition).squeeze(1) + 1e-9)

    def _decode(self, latent_feature, phase):
        if self.geometric_emission is not None:
            return self.geometric_emission.logits(phase, self.config.beats_per_bar)
        return self.decoder(latent_feature)

    def _sample_meter(self, meter_logits, gumbel_temperature, sample):
        if self.config.divergence_meter == "fixed":
            # Hard one-hot at the configured beats-per-bar (meter latent disabled).
            fixed = torch.zeros_like(meter_logits)
            fixed[:, min(self.config.beats_per_bar - 1, self.config.num_meters - 1)] = 1.0
            return fixed
        if sample:
            return sample_gumbel_softmax(meter_logits, gumbel_temperature)
        return F.softmax(meter_logits, dim=-1)

    # ---- the rollout ---------------------------------------------------------------------------

    def rollout(self, features, beat_channel, downbeat_channel,
                gumbel_temperature: float = 0.5, sample: bool = True, compute_kl: bool = True) -> RolloutResult:
        """Run the posterior forward over time, applying the prior dynamics and any divergence switches."""
        batch_size, num_frames, _ = features.shape
        device = features.device

        posterior_sequence, _ = self.posterior_gru(torch.cat([features, beat_channel.unsqueeze(-1),
                                                              downbeat_channel.unsqueeze(-1)], dim=-1))
        posterior_context = self.posterior_context(posterior_sequence)            # [batch, frames, hidden]
        prior_context = self.prior_context(self.prior_gru(features)[0]) if compute_kl else None

        # Per-clip computed tempo (divergence): one bar-phase advance for the whole crop, from autocorrelation.
        autocorr_advance, tempo_lag_scores = (None, None)
        if self.tempo_head is not None:
            autocorr_advance, tempo_lag_scores = self.tempo_head.bar_phase_advance(features, self.config.beats_per_bar)

        kl_meter = kl_phase = kl_tempo = (torch.zeros(batch_size, device=device) if compute_kl else None)

        # ---- frame 0: initialise the latents from the data ----
        packed = self.posterior_head(torch.cat([posterior_context[:, 0],
                                                self.initial_latent_feature.expand(batch_size, -1)], dim=-1))
        meter_logits, phase_mean, phase_conc, tempo_mean, tempo_std = self._unpack(packed)
        meter = self._sample_meter(meter_logits, gumbel_temperature, sample)
        log_tempo = self._frame_log_tempo(tempo_mean, tempo_std, autocorr_advance, sample)
        phase = sample_von_mises(phase_mean, phase_conc) if sample else phase_mean   # data-read initial offset

        if compute_kl:
            prior_packed = self.prior_initial(prior_context.mean(dim=1))
            prior_meter_logits, prior_phase_mean, prior_phase_conc, prior_tempo_mean, prior_tempo_std = self._unpack(prior_packed)
            kl_meter = kl_meter + kl_categorical(torch.log_softmax(meter_logits, -1), torch.log_softmax(prior_meter_logits, -1))
            kl_phase = kl_phase + kl_von_mises(phase_mean, phase_conc, prior_phase_mean, prior_phase_conc)
            if kl_tempo is not None and self.tempo_head is None:
                kl_tempo = kl_tempo + kl_normal(tempo_mean, tempo_std, prior_tempo_mean, prior_tempo_std)

        phase_sequence = [phase]
        log_tempo_sequence = [log_tempo]
        latent_features = [self._latent_feature(meter, phase, log_tempo)]
        previous_meter, previous_phase, previous_log_tempo = meter, phase, log_tempo

        # ---- frames 1..T-1: apply the dynamics + chosen phase update ----
        for frame_index in range(1, num_frames):
            packed = self.posterior_head(torch.cat([
                posterior_context[:, frame_index], self._latent_feature(previous_meter, previous_phase, previous_log_tempo)
            ], dim=-1))
            meter_logits, phase_data_mean, phase_conc, tempo_mean, tempo_std = self._unpack(packed)
            meter = self._sample_meter(meter_logits, gumbel_temperature, sample)
            log_tempo = self._frame_log_tempo(tempo_mean, tempo_std, autocorr_advance, sample)

            # Prior dynamics mean: advance phase by the per-frame tempo. The clamp bounds exp() so a
            # runaway log_tempo cannot produce an inf/NaN phase advance (>~20 rad/frame is already absurd).
            predicted_phase = (previous_phase + torch.exp(previous_log_tempo.clamp(-10.0, 3.0))) % TWO_PI   # prior dynamics mean
            phase = self._update_phase(predicted_phase, phase_data_mean, phase_conc, sample)

            if compute_kl:
                prior_phase_conc = F.softplus(self.prior_phase_concentration(prior_context[:, frame_index]).squeeze(-1)) + 0.01
                prior_tempo_std = F.softplus(self.prior_tempo_std(prior_context[:, frame_index]).squeeze(-1)) + 1e-3
                kl_meter = kl_meter + kl_categorical(
                    torch.log_softmax(meter_logits, -1),
                    self._meter_prior_logprob(previous_meter, phase, previous_phase, prior_context[:, frame_index]),
                )
                # Phase KL pulls the posterior phase toward the dynamics prediction -> couples phase to tempo.
                posterior_phase_mean = phase_data_mean if self.config.divergence_phase_update == "free" else predicted_phase
                kl_phase = kl_phase + kl_von_mises(posterior_phase_mean, phase_conc, predicted_phase, prior_phase_conc)
                if kl_tempo is not None and self.tempo_head is None:
                    kl_tempo = kl_tempo + kl_normal(tempo_mean, tempo_std, previous_log_tempo, prior_tempo_std)

            phase_sequence.append(phase)
            log_tempo_sequence.append(log_tempo)
            latent_features.append(self._latent_feature(meter, phase, log_tempo))
            previous_meter, previous_phase, previous_log_tempo = meter, phase, log_tempo

        decoder_logits = torch.stack([self._decode(latent_features[t], phase_sequence[t]) for t in range(num_frames)], dim=1)
        return RolloutResult(
            phase=torch.stack(phase_sequence, dim=1),
            log_tempo=torch.stack(log_tempo_sequence, dim=1),
            decoder_logits=decoder_logits,
            kl_meter=kl_meter, kl_phase=kl_phase, kl_tempo=kl_tempo,
            tempo_lag_scores=tempo_lag_scores,
        )

    def _frame_log_tempo(self, tempo_mean, tempo_std, autocorr_advance, sample):
        """The frame's log tempo: from the latent (default) or from the autocorrelation head (divergence)."""
        if self.tempo_head is not None:
            return torch.log(autocorr_advance)   # computed, constant per clip
        return sample_normal(tempo_mean, tempo_std) if sample else tempo_mean

    def _update_phase(self, predicted_phase, phase_data_mean, phase_concentration, sample):
        """Apply the configured phase-update rule (free / integrator / filter)."""
        rule = self.config.divergence_phase_update
        if rule == "free":
            return sample_von_mises(phase_data_mean, phase_concentration) % TWO_PI if sample else phase_data_mean
        if rule == "integrator":
            return sample_von_mises(predicted_phase, phase_concentration) % TWO_PI if sample else predicted_phase
        if rule == "filter":
            return blend_phase(predicted_phase, phase_data_mean % TWO_PI, torch.sigmoid(self.filter_gain))
        raise ValueError(f"unknown phase update rule: {rule}")
