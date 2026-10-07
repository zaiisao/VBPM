"""Historical generator variants retained for reproducible diagnostic commands."""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from diagnostics.emissions import (
    AngularFrameBinDecoder,
    BernoulliClockDecoder,
    ClockMassDecoder,
    FrameBinDecoder,
)
from diagnostics.synthetic_ladder import SparseDecoder
from vbpm.util.vonmises import _VonMisesInvCDF


class PhaseContextNormalization(nn.Module):
    """Normalize each pooled audio phase-context block."""

    def forward(self, context):
        """Evaluate the network on its input tensors."""
        return F.layer_norm(context.reshape(len(context), 3, 512), (512,)).reshape(
            len(context), 1536
        )


class Generator(nn.Module):
    """Audio-only phase/tempo prior with a trainable observation decoder."""

    def __init__(
        self,
        learn_tempo=False,
        aligned_context=False,
        proposal_max_bpm=None,
        tempo_log_scale=0.1,
        tempo_basis="framewise",
        smooth_concentration=False,
        frame_bin_emission=False,
        angular_frame_bin_emission=False,
        log_tempo_noise=False,
        clock_mass_emission=False,
        proposal_min_probability=None,
        tempo_feedback=False,
        bernoulli_clock_emission=False,
        prediction_checkpoint=None,
    ):
        super().__init__()
        self.factor = 8
        self.fps = 50.0
        self.proposal_meter = 4
        self.learn_tempo = learn_tempo
        self.aligned_context = aligned_context
        self.proposal_max_bpm = proposal_max_bpm
        self.proposal_min_probability = proposal_min_probability
        self.tempo_log_scale = tempo_log_scale
        self.tempo_basis = tempo_basis
        self.tempo_feedback = tempo_feedback
        self.smooth_concentration = smooth_concentration
        self.frame_bin_emission = frame_bin_emission
        self.log_tempo_noise = log_tempo_noise
        self.proposal_shift = None
        self.cached_prediction = None
        checkpoint = (
            prediction_checkpoint
            or Path.home() / ".cache/torch/hub/checkpoints/beat_this-final0.ckpt"
        )
        state = torch.load(checkpoint, weights_only=False, map_location="cpu")["state_dict"]
        self.register_buffer(
            "prediction_weight", state["model.task_heads.beat_downbeat_lin.weight"].clone()
        )
        self.register_buffer(
            "prediction_bias", state["model.task_heads.beat_downbeat_lin.bias"].clone()
        )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(512, elementwise_affine=False),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
        )
        if aligned_context:
            self.phase_head[0] = PhaseContextNormalization()
            self.phase_head[1] = nn.Linear(1536, 128)
        with torch.no_grad():
            self.phase_head[-1].weight.zero_()
            self.phase_head[-1].bias.copy_(torch.tensor([0.0, math.log(39.0)]))
        self.log_initial_sigma = nn.Parameter(torch.tensor(-9.0))
        self.log_increment_sigma = nn.Parameter(torch.tensor(-12.0))
        self.decoder = (
            BernoulliClockDecoder()
            if bernoulli_clock_emission
            else (
                ClockMassDecoder()
                if clock_mass_emission
                else (
                    AngularFrameBinDecoder()
                    if angular_frame_bin_emission
                    else (FrameBinDecoder() if frame_bin_emission else SparseDecoder())
                )
            )
        )
        if learn_tempo:
            self.tempo_head = nn.Sequential(
                nn.LayerNorm(512, elementwise_affine=False),
                nn.Linear(512, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.tempo_head[-1].weight)
            nn.init.zeros_(self.tempo_head[-1].bias)
            if tempo_feedback:
                original = self.tempo_head[1]
                expanded = nn.Linear(517, 64)
                with torch.no_grad():
                    expanded.weight.zero_()
                    expanded.weight[:, :512].copy_(original.weight)
                    expanded.bias.copy_(original.bias)
                self.tempo_head[1] = expanded

    def enable_temporal_phase_context(self):
        # Zero residual projection preserves the existing prior at initialization.
        """Add a zero-initialized temporal residual to phase conditioning."""
        self.temporal_phase_context = nn.Sequential(
            nn.Conv1d(512, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(64, 64, 5, dilation=8, padding=16),
            nn.Tanh(),
            nn.Conv1d(64, 64, 5, dilation=32, padding=64),
            nn.Tanh(),
            nn.Conv1d(64, 512, 1),
        )
        nn.init.zeros_(self.temporal_phase_context[-1].weight)
        nn.init.zeros_(self.temporal_phase_context[-1].bias)

    def parameters_for(self, h):
        """Return conditional phase, velocity, and concentration parameters."""
        guess = self.prediction_for(h)
        if self.proposal_shift is not None:
            guess["phase"] = guess["phase"] + self.proposal_shift[:, None]
        temporal_h = h
        if hasattr(self, "temporal_phase_context"):
            temporal_h = h + self.temporal_phase_context(h.transpose(1, 2)).transpose(1, 2)
        context = temporal_h.mean(1)
        if getattr(self, "periodic_context_only", False):
            # Remove constant audio content from the phase-weighted pools.
            # Finite windows otherwise leak their DC component into these pools.
            temporal_h = temporal_h - context[:, None]
            context = torch.zeros_like(context)
        if self.aligned_context:
            context = torch.cat(
                (
                    context,
                    (temporal_h * guess["phase"].cos().unsqueeze(-1)).mean(1),
                    (temporal_h * guess["phase"].sin().unsqueeze(-1)).mean(1),
                ),
                -1,
            )
        raw = self.phase_head(context)
        velocity = self.velocity_for(h, guess)
        relative = torch.cat((torch.zeros_like(velocity[:, :1]), velocity.cumsum(-1)), -1)
        anchor = guess["phase"].mean(1) + math.pi * raw[:, 0]
        phase0 = anchor - relative.mean(1)
        kappa = self.transform_concentration(raw[:, 1])
        return phase0, velocity, kappa

    def prediction_for(self, h):
        """Return cached or freshly computed audio-only proposals."""
        from diagnostics.audio_proposals import extract_audio_phase

        if self.cached_prediction is not None:
            return dict(self.cached_prediction)
        return extract_audio_phase(
            self,
            h,
            h.new_ones(h.shape[:2]),
            max_bpm=math.inf if self.proposal_max_bpm is None else self.proposal_max_bpm,
            min_probability=0.3
            if self.proposal_min_probability is None
            else self.proposal_min_probability,
            recover_missing_beats=getattr(self, "recover_missing_beats", False),
        )

    def transform_concentration(self, raw):
        """Map raw network output to positive phase concentration."""
        if self.smooth_concentration:
            limit = math.log(1e8)
            return 1 + (limit - F.softplus(limit - raw)).exp()
        return 1 + raw.clamp(-9, math.log(1e8)).exp()

    def velocity_for(self, h, guess):
        """Return the audio-conditioned mean velocity trajectory."""
        velocity = guess["phase"][:, 1:] - guess["phase"][:, :-1]
        if self.learn_tempo:
            if getattr(self, "tempo_feedback", False):
                # Sohn-style initial-prediction feedback: the residual head
                # must observe the proposal whose error it is correcting.
                per_frame_velocity = torch.cat((velocity, velocity[:, -1:]), -1)
                reference = 2 * math.pi * 2 / (self.proposal_meter * self.fps)
                cues = torch.stack(
                    (
                        guess["beat_logits"].sigmoid(),
                        guess["downbeat_logits"].sigmoid(),
                        (4 * guess["phase"]).cos(),
                        (4 * guess["phase"]).sin(),
                        (per_frame_velocity / reference).clamp_min(1e-6).log(),
                    ),
                    -1,
                )
                context = torch.cat((self.tempo_head[0](h), cues), -1)
                correction = self.tempo_head[1:](context).squeeze(-1)
            else:
                correction = self.tempo_head(h).squeeze(-1)
            if self.tempo_basis == "constant-correction":
                # Keep the audio proposal's within-window tempo variation,
                # but prevent the learned mean from chasing individual frames.
                correction = correction.mean(-1, keepdim=True).expand_as(correction)
            else:
                correction = F.avg_pool1d(
                    F.pad(correction.unsqueeze(1), (8, 8), mode="replicate"), 17, stride=1
                ).squeeze(1)
            if self.tempo_basis == "bounded-correction":
                # The existing physical gate uses 20..300 BPM. Parameterize
                # that mean domain smoothly rather than clipping predictions.
                minimum = 20 * (2 * math.pi) / (4 * 60 * self.fps)
                span = 280 * (2 * math.pi) / (4 * 60 * self.fps)
                fraction = ((velocity - minimum) / span).clamp(1e-6, 1 - 1e-6)
                logit = fraction.log() - torch.log1p(-fraction)
                fraction = (-F.softplus(-logit - self.tempo_log_scale * correction[:, :-1])).exp()
                velocity = minimum + span * fraction
            else:
                velocity = velocity * (self.tempo_log_scale * correction[:, :-1].tanh()).exp()
        if self.tempo_basis == "linear":
            # The mean log-tempo has a level and slope, as in the original
            # prior's linear trajectory option. Innovation noise still varies
            # by frame; this restricts only the conditional mean trajectory.
            log_velocity = velocity.clamp_min(1e-6).log()
            centered_time = torch.linspace(
                -1.0, 1.0, velocity.shape[1], device=h.device, dtype=h.dtype
            )
            level = log_velocity.mean(-1, keepdim=True)
            slope = (log_velocity * centered_time).mean(
                -1, keepdim=True
            ) / centered_time.square().mean()
            velocity = (level + slope * centered_time).exp()
        return velocity

    def load_previous(self, state):
        """Load compatible parameters from a previous staged checkpoint."""
        state = dict(state)
        if isinstance(self.decoder, ClockMassDecoder) and (
            "decoder.raw_width" not in state
            or state["decoder.raw_width"].shape != self.decoder.raw_width.shape
        ):
            state = {n: p for n, p in state.items() if not n.startswith("decoder.")}
        if self.aligned_context and state["phase_head.1.weight"].shape[1] == 512:
            expanded = torch.zeros_like(self.phase_head[1].weight)
            expanded[:, :512] = state["phase_head.1.weight"]
            state["phase_head.1.weight"] = expanded
        return self.load_state_dict(state, strict=False)

    def concentration(self, h):
        """Return input-conditioned phase concentration."""
        return self.parameters_for(h)[2]

    def trajectory(self, h, noise=None, parameters=None, noise_scales=None):
        """Integrate conditional phase and velocity trajectories."""
        phase0, velocity, _ = self.parameters_for(h) if parameters is None else parameters
        if noise is not None:
            scales = (
                (self.log_initial_sigma, self.log_increment_sigma)
                if noise_scales is None
                else noise_scales
            )
            delta0 = scales[0].clamp(-18, -1).exp() * noise["initial"]
            increments = scales[1].clamp(-18, -1).exp() * noise["increments"]
            offsets = torch.cat((delta0[..., None], delta0[..., None] + increments.cumsum(-1)), -1)
            if self.log_tempo_noise:
                mean_velocity = velocity.unsqueeze(0)
                velocity = mean_velocity * offsets.exp()
                drift = (velocity - mean_velocity).cumsum(-1)
                centered_drift = torch.cat((torch.zeros_like(drift[..., :1]), drift), -1).mean(-1)
                phase0 = phase0.unsqueeze(0) + noise["phase"] - centered_drift
            else:
                velocity = velocity.unsqueeze(0) + offsets
                phase0 = phase0.unsqueeze(0) + noise["phase"] - delta0 * ((h.shape[1] - 1) / 2)
        phase = torch.cat((phase0[..., None], phase0[..., None] + velocity.cumsum(-1)), -1)
        return phase, velocity

    def draw(self, h, noise):
        """Sample conditional phase and velocity trajectories."""
        parameters = self.parameters_for(h)
        kappa = parameters[2].unsqueeze(0).expand_as(noise["uniform"])
        phase_noise = _VonMisesInvCDF.apply(kappa, noise["uniform"])
        return self.trajectory(
            h,
            dict(phase=phase_noise, initial=noise["initial"], increments=noise["increments"]),
            parameters=parameters,
        )

    def loss(self, h, labels, noise):
        """Return sampled negative observation log likelihood."""
        phase, velocity = self.draw(h, noise)
        logits = self.decoder(phase, velocity)
        loss = F.cross_entropy(
            logits.reshape(-1, 3),
            labels.unsqueeze(0).expand_as(phase).reshape(-1),
            reduction="none",
        ).reshape_as(phase)
        return loss.sum(-1).mean()


class AttentionGenerator(Generator):
    """Learn phase origin relative to the current predicted tempo trajectory."""

    def __init__(
        self,
        learn_tempo=False,
        aligned_context=False,
        proposal_max_bpm=None,
        tempo_log_scale=0.1,
        tempo_basis="framewise",
        smooth_concentration=False,
        frame_bin_emission=False,
        phase_residual=False,
        angular_frame_bin_emission=False,
        log_tempo_noise=False,
        clock_mass_emission=False,
        proposal_min_probability=None,
        tempo_feedback=False,
        bernoulli_clock_emission=False,
    ):
        if aligned_context:
            raise ValueError("Attention phase origin already uses the current trajectory directly")
        super().__init__(
            learn_tempo,
            False,
            proposal_max_bpm,
            tempo_log_scale,
            tempo_basis,
            smooth_concentration,
            frame_bin_emission,
            angular_frame_bin_emission,
            log_tempo_noise,
            clock_mass_emission,
            proposal_min_probability,
            tempo_feedback,
            bernoulli_clock_emission,
        )
        self.phase_head[-1] = nn.Linear(128, 1)
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.constant_(self.phase_head[-1].bias, math.log(39.0))
        self.phase_attention = nn.Linear(512, 1, bias=False)
        with torch.no_grad():
            self.phase_attention.weight.copy_(self.prediction_weight[1:2])
        self.use_phase_residual = phase_residual
        if phase_residual:
            self.phase_residual = nn.Sequential(
                PhaseContextNormalization(), nn.Linear(1536, 128), nn.Tanh(), nn.Linear(128, 2)
            )
            nn.init.zeros_(self.phase_residual[-1].weight)
            nn.init.zeros_(self.phase_residual[-1].bias)

    def parameters_for(self, h):
        """Return conditional phase, velocity, and concentration parameters."""
        guess = self.prediction_for(h)
        velocity = self.velocity_for(h, guess)
        relative = torch.cat((torch.zeros_like(velocity[:, :1]), velocity.cumsum(-1)), -1)
        attention = self.phase_attention(h).squeeze(-1).softmax(-1)
        x = (attention * relative.cos()).sum(-1)
        y = (attention * relative.sin()).sum(-1)
        if self.use_phase_residual:
            context = torch.cat(
                (
                    h.mean(1),
                    (h * relative.cos().unsqueeze(-1)).mean(1),
                    (h * relative.sin().unsqueeze(-1)).mean(1),
                ),
                -1,
            )
            correction = self.phase_residual(context)
            x = x + correction[:, 0]
            y = y + correction[:, 1]
        phase0 = -torch.atan2(y, x + 1e-8)
        raw = self.phase_head(h.mean(1)).squeeze(-1)
        kappa = self.transform_concentration(raw)
        return phase0, velocity, kappa
