"""The encoder trunk, the emission, the prior chain, and the posterior."""

from __future__ import annotations

import math

import torch
from torch import nn

from .constants import KAPPA_MIN, METER0_SHARE, METER_TRANSITION, Q_METER0_LOGIT, TWO_PI
from .specs import EmissionSpec, PriorSpec, WalkSpec
from .util.positional_encoding import sinusoidal_encoding
from .util.vonmises import sample_vonmises_icdf


class Encoder(nn.Module):
    """The shared trunk (tutorial 9.2), reading AUDIO ONLY.

    It produces context, not posterior parameters: the variants build their own
    heads on top of its output.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 2,
        max_len: int = 4096,
    ):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.blocks = nn.TransformerEncoder(layer, layers)

        self.register_buffer("pe", sinusoidal_encoding(max_len, d_model), persistent=False)

    def forward(self, h, mask=None):
        """[B, T, D] -> [B, T, d_model]: the trunk shared by every head."""
        pad = None if mask is None else (mask <= 0)
        if pad is not None:
            # An empty item still needs one attention key; its pooled output
            # and likelihood are masked out by the caller.
            pad = pad.clone()
            pad[pad.all(1), 0] = False

        h = self.proj(h) + self.pe[: h.shape[1]]

        return self.blocks(h, src_key_padding_mask=pad)


class PosteriorModel(nn.Module):
    """Autoregressive q(path | h, labels) used by the optional mixed objective.

    Its encoder reads audio and labels; local heads condition on the previous
    sampled state. The deployed prior is an argument, not a registered child.
    """

    def __init__(
        self, input_dim: int, d_model: int, prior: "PriorModel", label_positions: bool = False
    ):
        super().__init__()
        self.label_positions = label_positions
        self.phase_kappa_min = None
        self.prior_residuals = True

        n_meters = len(prior.meters)
        state_dim = 3 + n_meters

        self.meters = prior.meters
        self.encoder = Encoder(input_dim + 3 + (4 if label_positions else 0), d_model)

        self.phase0_head = nn.Linear(d_model, 3)
        self.velocity0_head = nn.Linear(d_model, 2)
        self.phase_head = nn.Linear(d_model + state_dim, 2)
        self.velocity_head = nn.Linear(d_model + state_dim, 2)
        self.meter_head = nn.Linear(d_model + state_dim, n_meters)
        # Dimensionless corrections to p. At initialization q's continuous
        # factors equal p's; observed labels may subsequently refine them.
        for head in (
            self.phase0_head,
            self.velocity0_head,
            self.phase_head,
            self.velocity_head,
            self.meter_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.phase0_head.bias[0] = 1.0

    def forward(
        self,
        h,
        labels,
        mask,
        tau=1.0,
        hard=False,
        *,
        prior=None,
        velocity_ref=None,
        velocity_step=None,
    ):
        """q's initial-state parameters and draws, and the context the per-frame step reads."""
        inputs = [h, nn.functional.one_hot(labels, 3).to(h.dtype)]
        if self.label_positions:
            inputs.append(self.label_position_features(labels).to(h.dtype))
        feats = self.encoder(torch.cat(inputs, dim=-1), mask)

        meter0_logits = self.meter0_from_labels(labels)
        # JA: q_meter is returned in log form because it is solely used for the KL, and
        # log_softmax is safer than log(softmax) for numerical stability
        log_meter0 = torch.log_softmax(meter0_logits, dim=-1)

        phase0_cos, phase0_sin, phase0_raw = self.phase0_head(feats[:, 0]).unbind(-1)
        phase0_offset = torch.atan2(phase0_sin, phase0_cos)
        velocity0_offset, velocity0_raw = self.velocity0_head(feats[:, 0]).unbind(-1)
        if prior is None:
            # Standalone posterior diagnostics retain the legacy absolute units.
            phase0_mu = phase0_offset
            phase0_kappa = nn.functional.softplus(phase0_raw) + KAPPA_MIN
            velocity0_mu = velocity0_offset
            velocity0_sigma = nn.functional.softplus(velocity0_raw)
        else:
            p_mu, p_sigma = prior["velocity0"]
            velocity0_mu = p_mu + p_sigma * velocity0_offset
            velocity0_sigma = p_sigma * velocity0_raw.exp()

        B, T = feats.shape[:2]
        velocity_draw = feats.new_zeros(B, T)
        velocity_draw[:, 0] = velocity0_mu + torch.randn_like(velocity0_mu) * velocity0_sigma
        if prior is not None:
            p_mu, p_kappa = PriorModel.phase0_given_velocity(
                prior, velocity_draw[:, 0], mask, velocity_ref, velocity_step
            )
            phase0_mu = p_mu + phase0_offset
            phase0_kappa = p_kappa * phase0_raw.exp()
        phase0_draw = phase0_mu + sample_vonmises_icdf(phase0_kappa)
        meter_draw = feats.new_zeros(B, T, len(log_meter0[0]))
        meter_draw[:, 0] = nn.functional.gumbel_softmax(meter0_logits, tau=tau, hard=hard)

        draw = {"phase0": phase0_draw, "velocity": velocity_draw, "meter": meter_draw}
        q_phi = {
            "phase0": (phase0_mu, phase0_kappa),
            "velocity0": (velocity0_mu, velocity0_sigma),
            "feats": feats,
            "log_meter0": log_meter0,
        }
        return draw, q_phi

    @staticmethod
    def interpolated_position(events, T):
        """Interpolate event counts; return NaN outside the labelled interval."""
        frames = torch.arange(T, device=events.device, dtype=torch.float32)
        position = torch.full((T,), float("nan"), device=events.device)
        if len(events) < 2:
            return position
        count = torch.arange(len(events), device=events.device, dtype=torch.float32)
        index = torch.searchsorted(events, frames).clamp(1, len(events) - 1)
        left, right = events[index - 1], events[index]
        inside = (frames >= events[0]) & (frames <= events[-1])
        position[inside] = (count[index - 1] + (frames - left) / (right - left))[inside]
        return position

    def label_position_features(self, labels):
        """Return sine/cosine beat and bar positions, zero where undefined."""
        B, T = labels.shape
        out = torch.zeros(B, T, 4, device=labels.device)
        for b in range(B):
            for i, events in enumerate(
                (
                    torch.nonzero(labels[b] > 0).flatten().float(),
                    torch.nonzero(labels[b] == 2).flatten().float(),
                )
            ):
                position = self.interpolated_position(events, T)
                known = ~torch.isnan(position)
                angle = TWO_PI * position[known]
                out[b, known, 2 * i] = angle.cos()
                out[b, known, 2 * i + 1] = angle.sin()
        return out

    def meter0_from_labels(self, labels):
        """Infer initial meter from annotations, falling back to prior shares."""
        B = labels.shape[0]
        meter0_logits = (
            torch.tensor([METER0_SHARE[m] for m in self.meters], device=labels.device)
            .log()
            .expand(B, -1)
            .clone()
        )
        for b in range(B):
            downbeats = torch.nonzero(labels[b] == 2).flatten()
            beats = torch.nonzero(labels[b] > 0).flatten()
            if len(downbeats) < 2 or len(beats) < 2:
                continue
            first_bar = (beats >= downbeats[0]) & (beats < downbeats[1])
            beats_per_bar = int(first_bar.sum())
            if beats_per_bar not in self.meters:
                continue
            meter0_logits[b] = (
                Q_METER0_LOGIT
                * nn.functional.one_hot(
                    torch.tensor(self.meters.index(beats_per_bar)), len(self.meters)
                ).float()
            )
        return meter0_logits

    def step(self, feats_k, pred, velocity, meter, *, prior=None):
        """q's frame-k factors from c_k and the sampled previous state."""
        state = torch.cat([pred.cos()[:, None], pred.sin()[:, None], velocity[:, None], meter], -1)
        inputs = torch.cat([feats_k, state], -1)
        phase_offset, phase_log_kappa = self.phase_head(inputs).unbind(-1)
        velocity_offset, velocity_raw = self.velocity_head(inputs).unbind(-1)
        meter_logits = self.meter_head(inputs)
        if prior is None:
            phase_mu = pred + phase_offset
            phase_kappa = (
                phase_log_kappa.exp()
                if self.phase_kappa_min is None
                else nn.functional.softplus(phase_log_kappa) + self.phase_kappa_min
            )
            velocity_mu = velocity_offset
            velocity_sigma = nn.functional.softplus(velocity_raw)
        else:
            p_phase_mu, p_phase_kappa = prior["phase"]
            phase_mu = p_phase_mu + phase_offset / p_phase_kappa.sqrt()
            phase_kappa = p_phase_kappa * phase_log_kappa.exp()
            if self.phase_kappa_min is not None:
                phase_kappa = phase_kappa.clamp(min=self.phase_kappa_min)
            p_mu, p_sigma = prior["velocity"]
            velocity_mu = p_mu + p_sigma * velocity_offset
            velocity_sigma = p_sigma * velocity_raw.exp()
            meter_logits = prior["meter_logits"] + meter_logits
        return {
            "phase": (phase_mu, phase_kappa),
            "velocity": (velocity_mu, velocity_sigma),
            "meter_logits": meter_logits,
        }


class LandmarkEmission(nn.Module):
    """Normalized metrical N/B/D likelihood mixed over soft meter."""

    def __init__(self, meters):
        super().__init__()
        self.meters = tuple(int(m) for m in meters)
        for i, m in enumerate(self.meters):
            self.register_buffer(
                f"centers_{i}",
                torch.arange(1, m, dtype=torch.float32) * (TWO_PI / m),
                persistent=False,
            )
        self.event_bias = nn.Parameter(torch.full((2,), -1.5))
        self.log_kappa = nn.Parameter(torch.full((2,), math.log(64.0)))

    def forward(self, phi, velocity, meter, mask, h):
        """[B, T, 3] logits: beats at the meter's landmarks, the downbeat at phase zero."""
        kb, kd = self.log_kappa.exp().unbind()
        beat_by_meter = torch.stack(
            [
                torch.logsumexp(
                    -2 * kb * ((phi[..., None] - getattr(self, f"centers_{i}")) / 2).sin().square(),
                    dim=-1,
                )
                for i in range(len(self.meters))
            ],
            dim=-1,
        )
        beat = self.event_bias[0] + torch.logsumexp(
            beat_by_meter + meter.clamp(min=1e-8).log(), dim=-1
        )
        down = self.event_bias[1] - 2 * kd * (phi / 2).sin().square()
        return torch.stack([torch.zeros_like(phi), beat, down], dim=-1)

    def loglik(self, phi, velocity, meter, labels, mask, h):
        """[B]: log p(labels | path) summed over valid frames."""
        logp = self(phi, velocity, meter, mask, h).log_softmax(-1)
        return (logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum(1)


class EmissionModel(nn.Module):
    """p(labels | z): a Transformer over the latent sequence (tutorial 9.6)."""

    def __init__(self, spec: EmissionSpec, meters, input_dim: int, max_len: int = 4096):
        super().__init__()
        self.spec = spec

        self.proj = nn.Linear(
            2 + int(spec.reads_velocity) + len(meters) + (input_dim if spec.reads_audio else 0),
            spec.dim,
        )
        layer = nn.TransformerEncoderLayer(
            spec.dim,
            4,
            dim_feedforward=4 * spec.dim,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.blocks = nn.TransformerEncoder(layer, spec.layers)
        self.out = nn.Linear(spec.dim, 3)

        if spec.positional:
            self.register_buffer("pe", sinusoidal_encoding(max_len, spec.dim))

    def forward(self, phi, velocity, meter, mask, h):
        """Return N/B/D logits from the latent path and optional audio context."""
        parts = [phi.cos()[..., None], phi.sin()[..., None]]
        if self.spec.reads_velocity:
            parts.append(velocity[..., None])
        z = torch.cat(parts + [meter], -1)
        if self.spec.reads_audio:
            z = torch.cat([z, h], -1)
        x = self.proj(z)
        if self.spec.positional:
            x = x + self.pe[: x.shape[1]]
        x = self.blocks(x, src_key_padding_mask=mask <= 0)
        return self.out(x)

    def loglik(self, phi, velocity, meter, labels, mask, h):
        """[B]: log p(labels | path, h) summed over valid frames."""
        logp = torch.log_softmax(self(phi, velocity, meter, mask, h), -1)
        ll = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (ll * mask).sum(1)


class PriorModel(nn.Module):
    """p(path | h): a centered phase anchor and a coherent audio-conditioned tempo curve.

    The encoder sees audio only. All Gaussian means are velocity levels, in the
    units selected by VBPM. A random-walk transition retains the change between
    successive levels instead of discarding the learned mean.
    """

    def __init__(self, input_dim: int, meters, walk: WalkSpec, spec: PriorSpec | None = None):
        super().__init__()
        spec = spec or PriorSpec()
        self.meters = tuple(int(m) for m in meters)
        self.encoder = Encoder(input_dim, spec.dim, layers=spec.layers)
        self.phase0_head = nn.Linear(spec.dim, 3)
        self.velocity0_head = nn.Linear(spec.dim, 2)
        self.velocity_head = nn.Linear(spec.dim, 2)
        self.meter0_head = nn.Linear(spec.dim, len(self.meters))
        self.meter_head = nn.Linear(spec.dim + 3 + len(self.meters), len(self.meters))
        self.phase_log_kappa = nn.Parameter(torch.tensor(math.log(walk.prior_phase_kappa)))
        self.meter_gated = False
        self.register_buffer(
            "log_meter0_table",
            torch.tensor([METER0_SHARE[m] for m in self.meters]).log(),
            persistent=False,
        )
        self.register_buffer(
            "log_meter_transition",
            torch.tensor(
                [[METER_TRANSITION[a][b] for b in self.meters] for a in self.meters]
            ).log(),
            persistent=False,
        )
        # Start with a nonzero circular direction and modest uncertainty. No
        # annotation, input-angle estimator or pretrained decoder initializes p.
        nn.init.zeros_(self.phase0_head.weight)
        nn.init.zeros_(self.velocity0_head.weight)
        nn.init.zeros_(self.velocity_head.weight)
        with torch.no_grad():
            self.phase0_head.bias.copy_(
                torch.tensor(
                    [
                        1.0,
                        0.0,
                        (spec.phase0_kappa - KAPPA_MIN)
                        + math.log(-math.expm1(-(spec.phase0_kappa - KAPPA_MIN))),
                    ]
                )
            )
            raw_sigma = spec.velocity_sigma + math.log(-math.expm1(-spec.velocity_sigma))
            self.velocity0_head.bias.copy_(torch.tensor([0.0, raw_sigma]))
            self.velocity_head.bias.copy_(torch.tensor([0.0, raw_sigma]))

    def forward(self, h, mask=None, velocity_ref=None):
        """Encode once; express initial phase relative to the window's mean phase."""
        if mask is None:
            mask = h.new_ones(h.shape[:2])
        feats = self.encoder(h, mask)
        weights = mask / mask.sum(1, keepdim=True).clamp(min=1)
        pooled = (feats * weights[..., None]).sum(1)
        phase_cos, phase_sin, phase_raw = self.phase0_head(pooled).unbind(-1)
        anchor = torch.atan2(phase_sin, phase_cos)
        initial_mu, initial_raw = self.velocity0_head(pooled).unbind(-1)
        local_mu, local_raw = self.velocity_head(feats).unbind(-1)
        velocity_mu = initial_mu[:, None] + local_mu - local_mu[:, :1]
        physical = velocity_mu if velocity_ref is None else velocity_ref * velocity_mu.exp()
        advances = physical[:, :-1] * mask[:, 1:]
        relative = torch.cat([torch.zeros_like(physical[:, :1]), advances.cumsum(1)], 1)
        phase0_mu = anchor - (relative * weights).sum(1)
        return {
            "phase": self.phase_log_kappa.exp(),
            "phase0": (phase0_mu, nn.functional.softplus(phase_raw) + KAPPA_MIN),
            "phase_anchor": anchor,
            "velocity0": (initial_mu, nn.functional.softplus(initial_raw)),
            "velocity_mean": velocity_mu,
            "velocity_sigma": nn.functional.softplus(local_raw),
            "feats": feats,
            "log_meter0": torch.log_softmax(self.meter0_head(pooled), -1),
        }

    def step(self, p, frame, pred, velocity, meter, velocity_step=None):
        """Frame factors from cached audio context and the previous sampled state."""
        mu = p["velocity_mean"][:, frame]
        sigma = p["velocity_sigma"][:, frame]
        if velocity_step is not None:
            mu = velocity + mu - p["velocity_mean"][:, frame - 1]
            sigma = velocity_step * sigma
        state = torch.cat([pred.cos()[:, None], pred.sin()[:, None], velocity[:, None], meter], -1)
        inputs = torch.cat([p["feats"][:, frame], state], -1)
        meter_logits = self.meter_head(inputs)
        if self.meter_gated:
            meter_logits = meter_logits + meter @ self.log_meter_transition
        return {"velocity": (mu, sigma), "meter_logits": meter_logits}

    @staticmethod
    def phase0_given_velocity(p, velocity0, mask, velocity_ref=None, velocity_step=None):
        """Center the phase distribution conditional on the sampled initial tempo.

        The start factors are p(v0 | h) p(phi0 | v0, h). This keeps initial
        tempo noise from translating the whole window's phase anchor.
        """
        offset = velocity0 - p["velocity0"][0]
        if velocity_step is None:
            offset = torch.cat([offset[:, None], torch.zeros_like(p["velocity_mean"][:, 1:])], 1)
        else:
            offset = offset[:, None]
        levels = p["velocity_mean"] + offset
        physical = levels if velocity_ref is None else velocity_ref * levels.exp()
        relative = torch.cat(
            [torch.zeros_like(physical[:, :1]), (physical[:, :-1] * mask[:, 1:]).cumsum(1)], 1
        )
        weights = mask / mask.sum(1, keepdim=True).clamp(min=1)
        return p["phase_anchor"] - (relative * weights).sum(1), p["phase0"][1]

    def sample(
        self,
        h,
        mask,
        tau=1.0,
        hard=False,
        *,
        parameters=None,
        velocity_ref=None,
        velocity_step=None,
    ):
        """Reparameterized audio-conditioned start for draws_to_paths."""
        p = parameters if parameters is not None else self(h, mask, velocity_ref)
        B, T = mask.shape
        initial_mu, initial_sigma = p["velocity0"]
        velocity = h.new_zeros(B, T)
        velocity[:, 0] = initial_mu + torch.randn_like(initial_mu) * initial_sigma
        mu, kappa = self.phase0_given_velocity(p, velocity[:, 0], mask, velocity_ref, velocity_step)
        meter = h.new_zeros(B, T, len(self.meters))
        meter[:, 0] = nn.functional.gumbel_softmax(p["log_meter0"], tau=tau, hard=hard)
        return {"phase0": mu + sample_vonmises_icdf(kappa), "velocity": velocity, "meter": meter}

    @torch.no_grad()
    def mode(self, h, mask, *, parameters=None, velocity_ref=None):
        """Deterministic audio-conditioned start; consumes no random numbers."""
        p = parameters if parameters is not None else self(h, mask, velocity_ref)
        B, T = mask.shape
        velocity = h.new_zeros(B, T)
        velocity[:, 0] = p["velocity0"][0]
        meter = h.new_zeros(B, T, len(self.meters))
        meter[:, 0] = nn.functional.one_hot(p["log_meter0"].argmax(-1), len(self.meters)).to(
            h.dtype
        )
        return {"phase0": p["phase0"][0], "velocity": velocity, "meter": meter}
