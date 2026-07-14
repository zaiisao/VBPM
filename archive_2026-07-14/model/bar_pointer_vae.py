"""The Variational Bar Pointer Model: a conditional Deep Markov Model (Krishnan, Shalit & Sontag
2017; Girin et al. 2021 taxonomy) over three per-frame latents -- meter (Categorical), bar phase
(wrapped Cauchy), log-tempo (Laplace) -- conditioned on frozen frontend features, per the ELBO
derived in docs/ELBO_for_DBN.md and notebooks/vbpm_from_first_principles.ipynb.

The generative ancestry is the bar-pointer model (Whiteley, Cemgil & Godsill 2006): the emission
depends on pointer POSITION only; tempo parameterizes the transition. The default emission is the
madmom-style parametric cosine bump (side-channel fix, docs/emission_sidechannel_report.md); the
MLP emission modes are kept only as the documented diagnosis-ladder axis.

Deployment inference lives in model/particle_filter.py (bootstrap filter with the fixed 2026-07-10
settings); ``VariationalBarPointerModel.filter_deploy`` delegates to it.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace

from model.latents import (TWO_PI, kl_between_categoricals, kl_between_laplaces,
                           kl_between_wrapped_cauchy, phase_concentration_from_score,
                           sample_wrapped_cauchy, tempo_std_from_score)


def predicted_phase_mean(prev_phase, prev_log_tempo):
    # Deterministic bar-pointer advance: phi_{t-1} + exp(log_tempo_{t-1}), wrapped to [0, 2pi).
    return (prev_phase + torch.exp(prev_log_tempo.clamp(-10.0, 3.0))) % TWO_PI


@dataclass
class RolloutResult:
    bar_phase: torch.Tensor               # [batch, frames] sampled posterior phase (deployment signal)
    log_tempo: torch.Tensor               # [batch, frames]
    meter_probabilities: torch.Tensor     # [batch, frames, num_meters] (soft one-hot samples)
    event_logits: torch.Tensor            # [batch, frames, 2] decoder logits for (beat, downbeat)
    kl_meter: torch.Tensor | None         # [batch] each KL summed over frames (None when compute_kl=False)
    kl_phase: torch.Tensor | None
    kl_tempo: torch.Tensor | None
    # Prior-gradient channel: the SAME KLs with the posterior side detached (gradients reach only
    # the prior-side networks). Used by the prior-preserving free-bits objective; None otherwise.
    kl_meter_pg: torch.Tensor | None = None
    kl_phase_pg: torch.Tensor | None = None
    kl_tempo_pg: torch.Tensor | None = None
    meter_logits: torch.Tensor | None = None   # [batch, frames, K] posterior logits (meter emission)
    # Prior transition scales along the rollout (for the tutorial's eq-27 physical anchoring):
    prior_tempo_scales: torch.Tensor | None = None       # [batch, frames-1]
    prior_phase_concentrations: torch.Tensor | None = None  # [batch, frames-1]


class TransformerContext(nn.Module):
    """Bidirectional Transformer context encoder: [B, T, input_dim] -> [B, T, output_dim].

    Sinusoidal positional encoding + full (unmasked) self-attention, so -- like a bidirectional RNN --
    it is a SMOOTHING encoder: every frame attends over the whole sequence (past and future). Output
    width is 2*hidden_size so the downstream context projection is identical to the earlier BiGRU's.
    """
    def __init__(self, input_dim, output_dim, num_heads=4, num_layers=2, feedforward_multiplier=2,
                 max_len=8192):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, output_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        frequency = torch.exp(torch.arange(0, output_dim, 2).float() * (-math.log(10000.0) / output_dim))
        positional_encoding = torch.zeros(max_len, output_dim)
        positional_encoding[:, 0::2] = torch.sin(position * frequency)
        positional_encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("positional_encoding", positional_encoding, persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(
            output_dim, num_heads, dim_feedforward=feedforward_multiplier * output_dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        hidden = self.input_projection(x) + self.positional_encoding[: x.shape[1]].unsqueeze(0)
        return self.transformer(hidden)   # no attention mask => bidirectional (smoothing)


class ParametricPointerEmission(nn.Module):
    """Madmom-style emission with a FIXED functional form: the beat logit is a learned-height cosine
    bump at the beats-per-bar harmonic of the bar phase, the downbeat logit a bump at the fundamental.
    Five scalars total -- structurally incapable of reading event timing out of tempo or meter wiggles
    (the diagnosed side channel). beats_per_bar comes from the SOFT meter latent (never a hardcoded 4).
    """
    def __init__(self, num_meters):
        super().__init__()
        self.register_buffer("beats_per_class", torch.arange(1, num_meters + 1).float())
        self.beat_bias = nn.Parameter(torch.tensor(-2.0))
        self.beat_gain = nn.Parameter(torch.tensor(2.0))
        self.downbeat_bias = nn.Parameter(torch.tensor(-3.0))
        self.downbeat_gain = nn.Parameter(torch.tensor(2.0))
        self.sharpness = nn.Parameter(torch.tensor(2.0))

    def forward(self, latent_feature):
        cos_phi, sin_phi = latent_feature[..., 0], latent_feature[..., 1]
        meter = latent_feature[..., 3:]
        phase = torch.atan2(sin_phi, cos_phi)
        beats_per_bar = (meter * self.beats_per_class).sum(-1).clamp(min=1.0)
        k = F.softplus(self.sharpness)
        beat_bump = torch.exp(k * (torch.cos(beats_per_bar * phase) - 1.0))
        downbeat_bump = torch.exp(k * (torch.cos(phase) - 1.0))
        return torch.stack([self.beat_bias + F.softplus(self.beat_gain) * beat_bump,
                            self.downbeat_bias + F.softplus(self.downbeat_gain) * downbeat_bump], dim=-1)


class VariationalBarPointerModel(nn.Module):
    def __init__(self, feature_dim=512, hidden_size=64, num_meters=4,
                 transition_correction_scale=0.0, decoder_sees_tempo=True,
                 decoder_input_mode="parametric", fixed_prior_scales=None):
        super().__init__()
        self.num_meters = num_meters
        self.latent_feature_dim = 3 + num_meters            # (cos phi, sin phi, log tempo, meter one-hot)
        # One packed parameter vector per frame: meter logits, phase mean as (cos, sin),
        # raw phase concentration, log-tempo mean, raw log-tempo std.
        self.packed_parameter_dim = num_meters + 5

        # Posterior context reads features AND the observed event channels. A bidirectional
        # Transformer encoder (full self-attention) -- a smoothing context: every frame sees the
        # whole sequence. Prior context reads features only (the generative side never sees answers).
        self.posterior_encoder = TransformerContext(feature_dim + 2, 2 * hidden_size)
        self.posterior_context_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.prior_encoder = TransformerContext(feature_dim, 2 * hidden_size)
        self.prior_context_projection = nn.Linear(2 * hidden_size, hidden_size)

        self.posterior_parameter_head = nn.Sequential(          # all q parameters at frame t
            nn.Linear(hidden_size + self.latent_feature_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, self.packed_parameter_dim))
        self.initial_prior_head = nn.Sequential(                # p(z_1 | h)
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, self.packed_parameter_dim))
        self.prior_phase_concentration_head = nn.Linear(hidden_size, 1)   # phase concentration c^p_t
        self.prior_tempo_std_head = nn.Linear(hidden_size, 1)             # tempo Laplace scale b_t
        self.meter_transition_network = nn.Sequential(                    # K x K meter transition
            nn.Linear(num_meters + 4 + hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, num_meters * num_meters))
        if decoder_input_mode == "parametric":
            self.event_decoder = ParametricPointerEmission(num_meters)
        else:
            self.event_decoder = nn.Sequential(                           # MLP emission (ladder arms)
                nn.Linear(self.latent_feature_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2))
        self.initial_latent_feature = nn.Parameter(torch.zeros(self.latent_feature_dim))
        # g-prior: bounded audio correction of the transition MEANS. Head constructed ONLY when
        # enabled, so the baseline's parameter list and init RNG stream are byte-identical at scale 0.
        self.transition_correction_scale = transition_correction_scale
        # Side-channel cut (2026-07-10 root cause): when False, the event decoder sees
        # (phase, meter) ONLY. Tempo then affects the likelihood solely through phase advance, so
        # the encoder cannot Morse-code event timing through unphysical tempo wiggles.
        self.decoder_sees_tempo = decoder_sees_tempo
        # Ladder over what the emission may see (side-channel cuts, weakest -> strongest):
        #   'full'       -- z = (phase, tempo, meter)          [L0: the diagnosed broken baseline]
        #   'no_tempo'   -- tempo dim zeroed                    [L1]
        #   'phase_only' -- tempo AND meter dims zeroed         [L2: no side channel left]
        #   'parametric' -- madmom-style cosine-bump emission   [L3: fixed functional form; DEFAULT]
        self.decoder_input_mode = decoder_input_mode or ("full" if decoder_sees_tempo else "no_tempo")
        # L4 (scale-leak cut): freeze the PRIOR transition scales at physical values
        # (tempo_sigma_nats_per_frame, phase_concentration). The learned heads otherwise inflate
        # them to zero the KL price of posterior side-channel wiggles. ELBO stays exact -- this is
        # a spec choice of p(z_t|z_{t-1},h), not a loss change.
        self.fixed_prior_scales = fixed_prior_scales   # None or (sigma, concentration)
        if transition_correction_scale > 0.0:
            self.transition_correction_head = nn.Sequential(
                nn.Linear(hidden_size + self.latent_feature_dim, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, 2))

    # ---- helpers ------------------------------------------------------------------------------
    def unpack_distribution_parameters(self, packed):
        meter_logits = packed[:, :self.num_meters]
        phase_mean = torch.atan2(packed[:, self.num_meters + 1], packed[:, self.num_meters]) % TWO_PI
        phase_concentration = F.softplus(packed[:, self.num_meters + 2]) + 0.01
        log_tempo_mean = packed[:, self.num_meters + 3]
        log_tempo_std = F.softplus(packed[:, self.num_meters + 4]) + 1e-3
        return meter_logits, phase_mean, phase_concentration, log_tempo_mean, log_tempo_std

    def latent_feature_vector(self, meter, bar_phase, log_tempo):
        return torch.cat([torch.cos(bar_phase).unsqueeze(-1), torch.sin(bar_phase).unsqueeze(-1),
                          log_tempo.unsqueeze(-1), meter], dim=-1)

    def decoder_input(self, latent_feature):
        # Emission input per ladder rung (dims: 0-1 cos/sin phase, 2 log tempo, 3: meter).
        if self.decoder_input_mode in ("full", "parametric"):
            return latent_feature
        masked = latent_feature.clone()
        masked[..., 2] = 0.0
        if self.decoder_input_mode == "phase_only":
            masked[..., 3:] = 0.0
        return masked

    def prior_tempo_sigma(self, prior_context_frames):
        if self.fixed_prior_scales is not None:
            sigma, _ = self.fixed_prior_scales
            return torch.full(prior_context_frames.shape[:-1], sigma,
                              device=prior_context_frames.device)
        return tempo_std_from_score(self.prior_tempo_std_head(prior_context_frames).squeeze(-1))

    def prior_phase_concentration(self, prior_context_frames):
        if self.fixed_prior_scales is not None:
            _, concentration = self.fixed_prior_scales
            return torch.full(prior_context_frames.shape[:-1], concentration,
                              device=prior_context_frames.device)
        return phase_concentration_from_score(
            self.prior_phase_concentration_head(prior_context_frames).squeeze(-1))

    def transition_mean_corrections(self, prior_context_frame, previous_latent_feature, previous_log_tempo):
        """g: bounded residual corrections to the transition MEANS from (z_{t-1}, h_t).
        Phase correction bounded to (scale x one frame's advance) -- tempo-proportional, so it can
        re-lock but never teleport the pointer; tempo correction bounded to 0.04*scale nats/frame
        (~2%/frame at scale 0.5, the magnitude real tempo increments actually have).
        Identically zero when disabled: the exact spec transition."""
        if self.transition_correction_scale == 0.0:
            zero = torch.zeros_like(previous_log_tempo)
            return zero, zero
        raw = self.transition_correction_head(
            torch.cat([prior_context_frame, previous_latent_feature], dim=-1))
        frame_advance = torch.exp(previous_log_tempo.clamp(-10.0, 3.0))
        delta_phase = self.transition_correction_scale * frame_advance * torch.tanh(raw[..., 0])
        delta_log_tempo = 0.04 * self.transition_correction_scale * torch.tanh(raw[..., 1])
        return delta_phase, delta_log_tempo

    def meter_transition_log_probabilities(self, previous_meter, bar_phase, previous_phase, prior_context):
        # A full K x K transition matrix conditioned on (m_{t-1}, phi_t, phi_{t-1}, h_t);
        # the (soft) previous meter selects its row via a batched vector-matrix product.
        network_input = torch.cat([
            previous_meter,
            torch.cos(bar_phase).unsqueeze(-1), torch.sin(bar_phase).unsqueeze(-1),
            torch.cos(previous_phase).unsqueeze(-1), torch.sin(previous_phase).unsqueeze(-1),
            prior_context], dim=-1)
        transition_matrix = F.softmax(
            self.meter_transition_network(network_input).reshape(-1, self.num_meters, self.num_meters), dim=2)
        return torch.log(torch.bmm(previous_meter.unsqueeze(1), transition_matrix).squeeze(1) + 1e-9)

    # ---- the SGVB rollout (single reparameterized sample path) ---------------------------------
    def rollout(self, features, observed_beats, observed_downbeats,
                gumbel_temperature=0.5, sample=True, compute_kl=True):
        batch_size, num_frames, _ = features.shape

        posterior_context = self.posterior_context_projection(self.posterior_encoder(
            torch.cat([features, observed_beats.unsqueeze(-1), observed_downbeats.unsqueeze(-1)], dim=-1)))
        prior_context = self.prior_context_projection(self.prior_encoder(features)) if compute_kl else None

        def sample_meter(meter_logits):
            return F.gumbel_softmax(meter_logits, tau=gumbel_temperature) if sample \
                else F.softmax(meter_logits, dim=-1)

        kl_meter = kl_phase = kl_tempo = (
            torch.zeros(batch_size, device=features.device) if compute_kl else None)
        kl_meter_pg = kl_phase_pg = kl_tempo_pg = (
            torch.zeros(batch_size, device=features.device) if compute_kl else None)

        # ---- frame 0: initial posterior q(z_1 | b, h) and initial prior p(z_1 | h) ----
        packed = self.posterior_parameter_head(torch.cat(
            [posterior_context[:, 0], self.initial_latent_feature.expand(batch_size, -1)], dim=-1))
        meter_logits, phase_mean, phase_concentration, log_tempo_mean, log_tempo_std = \
            self.unpack_distribution_parameters(packed)
        meter = sample_meter(meter_logits)
        log_tempo = Laplace(log_tempo_mean, log_tempo_std).rsample() if sample else log_tempo_mean
        bar_phase = sample_wrapped_cauchy(phase_mean, phase_concentration) if sample else phase_mean

        if compute_kl:  # the three t=1 KL terms
            prior_packed = self.initial_prior_head(prior_context.mean(dim=1))
            (prior_meter_logits, prior_phase_mean, prior_phase_concentration,
             prior_log_tempo_mean, prior_log_tempo_std) = self.unpack_distribution_parameters(prior_packed)
            kl_meter = kl_meter + kl_between_categoricals(
                F.log_softmax(meter_logits, -1), F.log_softmax(prior_meter_logits, -1))
            kl_phase = kl_phase + kl_between_wrapped_cauchy(
                phase_mean, phase_concentration, prior_phase_mean, prior_phase_concentration)
            kl_tempo = kl_tempo + kl_between_laplaces(
                log_tempo_mean, log_tempo_std, prior_log_tempo_mean, prior_log_tempo_std)
            # prior-gradient channel (posterior detached; prior heads keep their gradients)
            kl_meter_pg = kl_meter_pg + kl_between_categoricals(
                F.log_softmax(meter_logits, -1).detach(), F.log_softmax(prior_meter_logits, -1))
            kl_phase_pg = kl_phase_pg + kl_between_wrapped_cauchy(
                phase_mean.detach(), phase_concentration.detach(), prior_phase_mean, prior_phase_concentration)
            kl_tempo_pg = kl_tempo_pg + kl_between_laplaces(
                log_tempo_mean.detach(), log_tempo_std.detach(), prior_log_tempo_mean, prior_log_tempo_std)

        phase_frames, log_tempo_frames, meter_frames = [bar_phase], [log_tempo], [meter]
        meter_logits_frames = [meter_logits]
        prior_sigma_frames, prior_conc_frames = [], []
        latent_features = [self.latent_feature_vector(meter, bar_phase, log_tempo)]

        # ---- frames 1..T-1: transitions + per-frame KLs ----
        for frame_index in range(1, num_frames):
            previous_meter, previous_phase, previous_log_tempo = meter, bar_phase, log_tempo
            packed = self.posterior_parameter_head(torch.cat(
                [posterior_context[:, frame_index], latent_features[-1]], dim=-1))
            meter_logits, phase_mean, phase_concentration, log_tempo_mean, log_tempo_std = \
                self.unpack_distribution_parameters(packed)

            meter = sample_meter(meter_logits)
            log_tempo = Laplace(log_tempo_mean, log_tempo_std).rsample() if sample else log_tempo_mean
            # Prior mean: the bar-pointer advance, evaluated at the SAMPLED previous state.
            predicted_phase = predicted_phase_mean(previous_phase, previous_log_tempo)
            bar_phase = sample_wrapped_cauchy(phase_mean, phase_concentration) if sample else phase_mean

            if compute_kl:
                prior_phase_concentration = self.prior_phase_concentration(prior_context[:, frame_index])
                prior_log_tempo_std = self.prior_tempo_sigma(prior_context[:, frame_index])
                prior_sigma_frames.append(prior_log_tempo_std)
                prior_conc_frames.append(prior_phase_concentration)
                # g-prior: bounded audio corrections of the transition MEANS (zero when disabled --
                # then the two KLs below are the exact faithful transitions).
                delta_phase, delta_log_tempo = self.transition_mean_corrections(
                    prior_context[:, frame_index], latent_features[-1], previous_log_tempo)
                predicted_phase = (predicted_phase + delta_phase) % TWO_PI
                # Tempo piece: Laplace(mu_q, b_q) vs Laplace(s_{t-1} + delta, b_t).
                kl_tempo = kl_tempo + kl_between_laplaces(
                    log_tempo_mean, log_tempo_std, previous_log_tempo + delta_log_tempo, prior_log_tempo_std)
                # Phase piece: WC(mu_q, rho_q) vs WC(predicted advance, rho^p) -- THE term that
                # couples the phase to the tempo dynamics.
                kl_phase = kl_phase + kl_between_wrapped_cauchy(
                    phase_mean, phase_concentration, predicted_phase, prior_phase_concentration)
                # Meter piece: evaluated AFTER sampling phi_t (its prior conditions on it).
                kl_meter = kl_meter + kl_between_categoricals(
                    F.log_softmax(meter_logits, -1),
                    self.meter_transition_log_probabilities(
                        previous_meter, bar_phase, previous_phase, prior_context[:, frame_index]))
                # prior-gradient channel: q detached AND the sampled state inputs to the prior
                # detached, so gradients flow ONLY into the prior-side networks (scale heads,
                # meter transition, correction head, prior encoder).
                delta_phase_pg, delta_log_tempo_pg = self.transition_mean_corrections(
                    prior_context[:, frame_index], latent_features[-1].detach(), previous_log_tempo.detach())
                predicted_phase_pg = (predicted_phase_mean(
                    previous_phase.detach(), previous_log_tempo.detach()) + delta_phase_pg) % TWO_PI
                kl_phase_pg = kl_phase_pg + kl_between_wrapped_cauchy(
                    phase_mean.detach(), phase_concentration.detach(), predicted_phase_pg, prior_phase_concentration)
                kl_tempo_pg = kl_tempo_pg + kl_between_laplaces(
                    log_tempo_mean.detach(), log_tempo_std.detach(),
                    previous_log_tempo.detach() + delta_log_tempo_pg, prior_log_tempo_std)
                kl_meter_pg = kl_meter_pg + kl_between_categoricals(
                    F.log_softmax(meter_logits, -1).detach(),
                    self.meter_transition_log_probabilities(
                        previous_meter.detach(), bar_phase.detach(), previous_phase.detach(),
                        prior_context[:, frame_index]))

            phase_frames.append(bar_phase)
            log_tempo_frames.append(log_tempo)
            meter_frames.append(meter)
            meter_logits_frames.append(meter_logits)
            latent_features.append(self.latent_feature_vector(meter, bar_phase, log_tempo))

        event_logits = torch.stack(
            [self.event_decoder(self.decoder_input(latent_features[t])) for t in range(num_frames)], dim=1)
        return RolloutResult(
            bar_phase=torch.stack(phase_frames, dim=1),
            log_tempo=torch.stack(log_tempo_frames, dim=1),
            meter_probabilities=torch.stack(meter_frames, dim=1),
            event_logits=event_logits,
            kl_meter=kl_meter, kl_phase=kl_phase, kl_tempo=kl_tempo,
            kl_meter_pg=kl_meter_pg, kl_phase_pg=kl_phase_pg, kl_tempo_pg=kl_tempo_pg,
            meter_logits=torch.stack(meter_logits_frames, dim=1),
            prior_tempo_scales=(torch.stack(prior_sigma_frames, dim=1) if prior_sigma_frames else None),
            prior_phase_concentrations=(torch.stack(prior_conc_frames, dim=1) if prior_conc_frames else None))

    # ---- the PREDICTION pipeline (Sohn et al. 2015): z from the prior network only, never y ----
    def rollout_prior(self, features, gumbel_temperature=0.5, sample=True):
        """Generate through p_theta(z|x) alone -- the pipeline used at TEST time (Sohn et al. 2015,
        sec. 4.1: y* from z* = E[z|x] when sample=False). The event channels y are never an input
        anywhere on this path."""
        batch_size, num_frames, _ = features.shape
        prior_context = self.prior_context_projection(self.prior_encoder(features))

        def sample_meter(meter_logits):
            return F.gumbel_softmax(meter_logits, tau=gumbel_temperature) if sample \
                else F.softmax(meter_logits, dim=-1)

        # frame 0: the audio-conditioned initial prior p(z_1 | h)
        prior_packed = self.initial_prior_head(prior_context.mean(dim=1))
        meter_logits, phase_mean, phase_concentration, log_tempo_mean, log_tempo_std = \
            self.unpack_distribution_parameters(prior_packed)
        meter = sample_meter(meter_logits)
        log_tempo = Laplace(log_tempo_mean, log_tempo_std).rsample() if sample else log_tempo_mean
        bar_phase = sample_wrapped_cauchy(phase_mean, phase_concentration) if sample else phase_mean

        phase_frames, log_tempo_frames, meter_frames = [bar_phase], [log_tempo], [meter]
        latent_features = [self.latent_feature_vector(meter, bar_phase, log_tempo)]

        # frames 1..T-1: the transition priors -- means from the dynamics, scales (and the meter
        # transition) audio-conditioned through the prior context.
        for frame_index in range(1, num_frames):
            previous_meter, previous_phase, previous_log_tempo = meter, bar_phase, log_tempo
            prior_log_tempo_std = self.prior_tempo_sigma(prior_context[:, frame_index])
            delta_phase, delta_log_tempo = self.transition_mean_corrections(
                prior_context[:, frame_index], latent_features[-1], previous_log_tempo)
            prior_log_tempo_mean = previous_log_tempo + delta_log_tempo
            log_tempo = (Laplace(prior_log_tempo_mean, prior_log_tempo_std).rsample()
                         if sample else prior_log_tempo_mean)
            predicted_phase = (predicted_phase_mean(previous_phase, previous_log_tempo) + delta_phase) % TWO_PI
            prior_phase_concentration = self.prior_phase_concentration(prior_context[:, frame_index])
            bar_phase = (sample_wrapped_cauchy(predicted_phase, prior_phase_concentration)
                         if sample else predicted_phase)
            meter_logits = self.meter_transition_log_probabilities(
                previous_meter, bar_phase, previous_phase, prior_context[:, frame_index])
            meter = sample_meter(meter_logits)

            phase_frames.append(bar_phase)
            log_tempo_frames.append(log_tempo)
            meter_frames.append(meter)
            latent_features.append(self.latent_feature_vector(meter, bar_phase, log_tempo))

        event_logits = torch.stack(
            [self.event_decoder(self.decoder_input(latent_features[t])) for t in range(num_frames)], dim=1)
        return RolloutResult(
            bar_phase=torch.stack(phase_frames, dim=1),
            log_tempo=torch.stack(log_tempo_frames, dim=1),
            meter_probabilities=torch.stack(meter_frames, dim=1),
            event_logits=event_logits,
            kl_meter=None, kl_phase=None, kl_tempo=None)

    # ---- deployment by FILTERING: the lineage's own inference ----------------------------------
    @torch.no_grad()
    def filter_deploy(self, features, observations, **filter_kwargs):
        """Bootstrap particle filter on the learned model; see model/particle_filter.py for the
        implementation and the fixed 2026-07-10 deployment defaults."""
        from model.particle_filter import run_particle_filter
        return run_particle_filter(self, features, observations, **filter_kwargs)
