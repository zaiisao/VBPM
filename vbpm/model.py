"""VBPM: the bar-pointer CVAE with phase, velocity and meter latents."""

from __future__ import annotations

import torch
from torch import nn

from .constants import TWO_PI
from .nets import EmissionModel, LandmarkEmission, PosteriorModel, PriorModel
from .util.kl import gaussian_kl
from .util.vonmises import kl_vonmises, sample_vonmises_icdf
from .specs import EmissionSpec, PriorSpec, WalkSpec

DEFAULTS = {}


class VBPM(nn.Module):
    """Posterior, prior and emission, joined by draws_to_paths."""

    posterior_parameterization = "prior_residual_v1"

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        meters=(3, 4),
        emission: EmissionSpec | None = None,
        walk: WalkSpec | None = None,
        velocity_ref: float | None = None,
        velocity_step: float | None = None,
        phase0_fixed: bool = False,
        label_positions: bool = False,
        prior_spec: PriorSpec | None = None,
        meter_gated: bool = False,
    ):
        super().__init__()
        self.phase0_fixed = phase0_fixed
        self.velocity_ref = velocity_ref
        self.velocity_step = velocity_step

        self.register_buffer(
            "meter_values", torch.tensor(meters, dtype=torch.float32), persistent=False
        )
        self.tau = 0.5
        self.meter_hard = False

        emission = emission or EmissionSpec()
        if emission.kind == "landmark":
            self.emission_model = LandmarkEmission(meters)
        else:
            self.emission_model = EmissionModel(emission, meters, input_dim)
        self.prior_model = PriorModel(input_dim, meters, walk or WalkSpec(), prior_spec)
        self.prior_model.meter_gated = meter_gated
        self.posterior_model = PosteriorModel(
            input_dim, d_model, self.prior_model, label_positions=label_positions
        )

    @property
    def deployed_net(self):
        """The network that runs at test time; it reads audio only."""
        return self.prior_model

    def draws_to_paths(
        self, draw, mask, posterior=None, feats=None, prior=None, h=None, tau=1.0, mode=False
    ):
        """Replay a latent draw, returning the path and per-step KL bookkeeping."""
        if h is not None and (prior is None or "feats" not in prior):
            prior = self.prior_model(h, mask, self.velocity_ref)
        state = _PathState.from_draw(draw, self.meter_values)
        path = _PathHistory.from_initial(draw, mask, state)

        for frame in range(1, mask.shape[1]):
            prediction = state.phi + self.phase_step(state.velocity) * mask[:, frame]
            factors = self._step_factors(frame, state, prediction, posterior, feats, prior, h)
            update = _advance_path(
                frame,
                state,
                prediction,
                draw,
                mask,
                factors,
                tau,
                self.meter_values,
                mode,
                self.meter_hard,
                self.prior_model.meter_gated,
            )
            path.record(frame, state, update)
            state = update.state

        return path.as_dict()

    def _start(self, draw):
        """The draw with phi0 pinned to zero when the crop is taken to start on a downbeat."""
        if self.phase0_fixed:
            draw = {**draw, "phase0": torch.zeros_like(draw["phase0"])}
        return draw

    def phase_step(self, velocity):
        """Return phase advance in physical or log-scaled velocity units."""
        if self.velocity_ref is None:
            return velocity
        return self.velocity_ref * velocity.exp()

    def _step_factors(self, frame, state, prediction, posterior, feats, prior, h):
        """Get the local distributions that steer this frame's transition."""
        prior_step = None
        if h is not None:
            prior_step = self.prior_model.step(
                prior, frame, prediction, state.velocity, state.meter, self.velocity_step
            )
            prior_step["phase"] = (prediction, prior["phase"].expand_as(state.phi))
        if posterior is not None:
            kwargs = {"prior": prior_step} if getattr(posterior, "prior_residuals", False) else {}
            factors = posterior.step(
                feats[:, frame], prediction, state.velocity, state.meter, **kwargs
            )
            if prior_step is not None:
                factors["prior_velocity"] = prior_step["velocity"]
                factors["prior_meter_logits"] = prior_step["meter_logits"]
            return factors
        if prior is not None:
            factors = {"phase": (prediction, prior["phase"].expand_as(state.phi))}
            if prior_step is not None:
                factors.update(prior_step)
            return factors
        return {}

    def kl(self, h, mask, q_phi, path):
        """[B] KL(q || p) along q's sampled path."""
        return sum(self.kl_terms(h, mask, q_phi, path).values())

    def kl_terms(self, h, mask, q_phi, path, prior=None):
        """Each [B] term of KL(q || p) along q's sampled path, by factor."""
        p = prior if prior is not None else self.prior_model(h, mask, self.velocity_ref)
        live = mask.clone()
        live[:, 0] = 0.0

        p_phase0 = self.prior_model.phase0_given_velocity(
            p, path["velocity_path"][:, 0], mask, self.velocity_ref, self.velocity_step
        )
        kl_phase0 = kl_vonmises(*q_phi["phase0"], *p_phase0) * (0.0 if self.phase0_fixed else 1.0)

        kl_phase = kl_vonmises(
            path["phase_mu"],
            path["phase_kappa"],
            path["phase_pred"],
            p["phase"].expand_as(path["phase_mu"]),
        )
        kl_phase = (kl_phase * live).sum(1)

        kl_velocity0 = gaussian_kl(*q_phi["velocity0"], *p["velocity0"])

        kl_velocity = gaussian_kl(
            path["velocity_mu"],
            path["velocity_sigma"],
            path["prior_velocity_mu"],
            path["prior_velocity_sigma"],
        )
        kl_velocity = (kl_velocity * live).sum(1)

        q_meter0 = q_phi["log_meter0"].exp()
        kl_meter0 = (q_meter0 * (q_phi["log_meter0"] - p["log_meter0"])).sum(-1)

        q_meter = path["log_meter"].exp()
        kl_meter = (q_meter * (path["log_meter"] - path["prior_log_meter"])).sum(-1)
        meter_frames = path["is_downbeat"] * live if self.prior_model.meter_gated else live
        kl_meter = (kl_meter * meter_frames).sum(1)

        return {
            "phase0": kl_phase0,
            "phase": kl_phase,
            "velocity0": kl_velocity0,
            "velocity": kl_velocity,
            "meter0": kl_meter0,
            "meter": kl_meter,
        }

    def forward(self, h, mask, cls=None, *, gsnn_only=False):
        """The ELBO for one batch: recon on q's sampled path minus KL(q || p)."""
        if gsnn_only:
            return self.forward_gsnn(h, mask, cls)
        prior = self.prior_model(h, mask, self.velocity_ref)
        draw, q_phi = self.posterior_model(
            h,
            cls,
            mask,
            tau=self.tau,
            hard=self.meter_hard,
            prior=prior,
            velocity_ref=self.velocity_ref,
            velocity_step=self.velocity_step,
        )
        draw = self._start(draw)
        feats = q_phi["feats"]
        path = self.draws_to_paths(
            draw, mask, posterior=self.posterior_model, feats=feats, prior=prior, h=h, tau=self.tau
        )

        recon = self.emission_model.loglik(
            path["phi_path"], path["velocity_path"], path["meter_path"], cls, mask, h
        )
        kl_terms = self.kl_terms(h, mask, q_phi, path, prior=prior)
        kl = sum(kl_terms.values())
        elbo = recon - kl

        prior_out = self.forward_gsnn(h, mask, cls, prior=prior)
        prior_path, recon_prior = prior_out["prior_path"], prior_out["recon_prior"]

        return {
            "elbo": elbo,
            "recon": recon,
            "kl": kl,
            "kl_terms": kl_terms,
            "recon_prior": recon_prior,
            "prior_path": prior_path,
            "q_phase0": q_phi["phase0"],
            "path": path,
            "phi": path["phi_path"],
            "kappa": path["phase_kappa"],
        }

    def forward_gsnn(self, h, mask, cls, *, prior=None):
        """Sample p(z | h), train p(labels | z); q and KL are never evaluated.

        Zero KL fields keep the training logger compatible; they mean "unused",
        not a measured divergence between the two networks.
        """
        prior = prior if prior is not None else self.prior_model(h, mask, self.velocity_ref)
        draw = self.prior_model.sample(
            h,
            mask,
            tau=self.tau,
            hard=self.meter_hard,
            parameters=prior,
            velocity_ref=self.velocity_ref,
            velocity_step=self.velocity_step,
        )
        path = self.draws_to_paths(self._start(draw), mask, prior=prior, h=h, tau=self.tau)
        recon = self.emission_model.loglik(
            path["phi_path"], path["velocity_path"], path["meter_path"], cls, mask, h
        )
        return {
            "recon_prior": recon,
            "prior_path": path,
            "elbo": recon,
            "recon": recon,
            "kl": torch.zeros_like(recon),
            "kl_terms": {},
            "path": path,
            "phi": path["phi_path"],
            "kappa": path["phase_kappa"],
        }

    @torch.no_grad()
    def infer_path(self, h, mask=None):
        """Label-free, deterministic deployment through the bar pointer."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        prior = self.prior_model(h, mask, self.velocity_ref)
        draw = self.prior_model.mode(h, mask, parameters=prior)
        return self.draws_to_paths(self._start(draw), mask, prior=prior, h=h, mode=True)

    @torch.no_grad()
    def label_probs(self, h, mask=None, rollouts=64):
        """[B, T, 3] probabilities averaged over sampled prior rollouts."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        B, T = mask.shape
        # Encode each song once, then replicate its factors for the MC draws.
        prior = self.prior_model(h, mask, self.velocity_ref)
        prior = {
            k: (
                tuple(vv.repeat_interleave(rollouts, 0) for vv in v)
                if isinstance(v, tuple)
                else v
                if v.ndim == 0
                else v.repeat_interleave(rollouts, 0)
            )
            for k, v in prior.items()
        }
        h_n = h.repeat_interleave(rollouts, 0)
        mask_n = mask.repeat_interleave(rollouts, 0)
        draw = self.prior_model.sample(
            h_n,
            mask_n,
            tau=self.tau,
            hard=self.meter_hard,
            parameters=prior,
            velocity_ref=self.velocity_ref,
            velocity_step=self.velocity_step,
        )
        path = self.draws_to_paths(self._start(draw), mask_n, prior=prior, h=h_n, tau=self.tau)
        logits = self.emission_model(
            path["phi_path"], path["velocity_path"], path["meter_path"], mask_n, h_n
        )
        return torch.softmax(logits, -1).view(B, rollouts, T, 3).mean(1)


class _PathState:
    """The bar pointer's state at one frame."""

    def __init__(self, phi, velocity, meter, bar_start, beat_index, spacing):
        self.phi = phi
        self.velocity = velocity
        self.meter = meter
        self.bar_start = bar_start
        self.beat_index = beat_index
        self.spacing = spacing

    @classmethod
    def from_draw(cls, draw, meter_values):
        phi = draw["phase0"]
        meter = draw["meter"][:, 0]
        spacing = TWO_PI / (meter @ meter_values)
        bar_start = torch.floor(phi / TWO_PI) * TWO_PI
        beat_index = torch.floor((phi - bar_start) / spacing) + 1
        return cls(phi, draw["velocity"][:, 0], meter, bar_start, beat_index, spacing)


class _PathHistory:
    """Framewise outputs and KL terms accumulated while replaying a draw."""

    def __init__(self, mask, draw, state):
        batch, frames = mask.shape
        self.phi_path = draw["velocity"].new_zeros(batch, frames)
        self.velocity_path = torch.zeros_like(self.phi_path)
        self.meter_path = draw["meter"].new_zeros(batch, frames, draw["meter"].shape[-1])
        self.phase_mu = torch.zeros_like(self.phi_path)
        self.phase_kappa = torch.zeros_like(self.phi_path)
        self.phase_pred = torch.zeros_like(self.phi_path)
        self.velocity_mu = torch.zeros_like(self.phi_path)
        self.velocity_sigma = torch.ones_like(self.phi_path)
        self.prior_velocity_mu = torch.zeros_like(self.phi_path)
        self.prior_velocity_sigma = torch.ones_like(self.phi_path)
        self.log_meter = draw["meter"].new_zeros(batch, frames, draw["meter"].shape[-1])
        self.prior_log_meter = torch.zeros_like(self.log_meter)
        self.phi_path[:, 0] = state.phi
        self.velocity_path[:, 0] = state.velocity
        self.meter_path[:, 0] = state.meter
        self.phase_mu[:, 0] = state.phi
        self.phase_pred[:, 0] = state.phi

        self.is_beat = torch.zeros(batch, frames, dtype=torch.bool, device=mask.device)
        self.is_beat[:, 0] = (mask[:, 0] > 0) & (state.phi == state.bar_start)
        self.is_downbeat = self.is_beat.clone()
        self.crossing = torch.zeros(batch, frames, device=mask.device)

    @classmethod
    def from_initial(cls, draw, mask, state):
        return cls(mask, draw, state)

    def record(self, frame, state, update):
        self.phi_path[:, frame] = update.new_phi
        self.velocity_path[:, frame] = update.state.velocity
        self.meter_path[:, frame] = update.state.meter
        self.phase_mu[:, frame] = update.phase_mu
        self.phase_kappa[:, frame] = update.phase_kappa
        self.phase_pred[:, frame] = update.prediction
        self.velocity_mu[:, frame] = update.velocity_mu
        self.velocity_sigma[:, frame] = update.velocity_sigma
        self.prior_velocity_mu[:, frame] = update.prior_velocity_mu
        self.prior_velocity_sigma[:, frame] = update.prior_velocity_sigma
        self.log_meter[:, frame] = update.log_meter
        self.prior_log_meter[:, frame] = update.prior_log_meter
        self.is_beat[:, frame] = update.crossed
        self.is_downbeat[:, frame] = update.downbeat
        self.crossing[:, frame] = update.crossing

    def as_dict(self):
        return {
            "phi_path": self.phi_path,
            "velocity_path": self.velocity_path,
            "meter_path": self.meter_path,
            "is_beat": self.is_beat,
            "is_downbeat": self.is_downbeat,
            "crossing": self.crossing,
            "phase_mu": self.phase_mu,
            "phase_kappa": self.phase_kappa,
            "phase_pred": self.phase_pred,
            "velocity_mu": self.velocity_mu,
            "velocity_sigma": self.velocity_sigma,
            "prior_velocity_mu": self.prior_velocity_mu,
            "prior_velocity_sigma": self.prior_velocity_sigma,
            "log_meter": self.log_meter,
            "prior_log_meter": self.prior_log_meter,
        }


class _PathUpdate:
    def __init__(
        self,
        state,
        prediction,
        new_phi,
        phase_mu,
        phase_kappa,
        velocity_mu,
        velocity_sigma,
        prior_velocity_mu,
        prior_velocity_sigma,
        log_meter,
        prior_log_meter,
        crossed,
        downbeat,
        crossing,
    ):
        self.state = state
        self.prediction = prediction
        self.new_phi = new_phi
        self.phase_mu = phase_mu
        self.phase_kappa = phase_kappa
        self.velocity_mu = velocity_mu
        self.velocity_sigma = velocity_sigma
        self.prior_velocity_mu = prior_velocity_mu
        self.prior_velocity_sigma = prior_velocity_sigma
        self.log_meter = log_meter
        self.prior_log_meter = prior_log_meter
        self.crossed = crossed
        self.downbeat = downbeat
        self.crossing = crossing


def _advance_path(
    frame,
    state,
    prediction,
    draw,
    mask,
    factors,
    tau,
    meter_values,
    mode=False,
    meter_hard=False,
    meter_gated=False,
):
    """Take one frame step and return its outputs with the next pointer state."""
    zeros = torch.zeros_like(state.phi)
    live = mask[:, frame] > 0

    if "phase" in factors:
        phase_mu, phase_kappa = factors["phase"]
        offset = torch.atan2(torch.sin(phase_mu - prediction), torch.cos(phase_mu - prediction))
        noise = zeros if mode else sample_vonmises_icdf(phase_kappa)
        new_phi = prediction + offset + noise
    else:
        phase_mu, phase_kappa, new_phi = prediction, zeros, prediction
    new_phi = torch.where(live, new_phi, state.phi)

    if "velocity" in factors:
        velocity_mu, velocity_sigma = factors["velocity"]
        noise = zeros if mode else torch.randn_like(velocity_mu)
        new_velocity = velocity_mu + noise * velocity_sigma
    else:
        velocity_mu, velocity_sigma = zeros, zeros + 1.0
        new_velocity = draw["velocity"][:, frame]
    new_velocity = torch.where(live, new_velocity, state.velocity)
    prior_velocity_mu, prior_velocity_sigma = factors.get("prior_velocity", (zeros, zeros + 1.0))

    if "meter_logits" in factors:
        meter_logits = factors["meter_logits"]
        if mode:
            next_meter = nn.functional.one_hot(meter_logits.argmax(-1), meter_logits.shape[-1])
            next_meter = next_meter.to(meter_logits.dtype)
        else:
            next_meter = nn.functional.gumbel_softmax(meter_logits, tau=tau, hard=meter_hard)
        log_meter = torch.log_softmax(meter_logits, -1)
    else:
        next_meter = draw["meter"][:, frame]
        log_meter = torch.zeros_like(state.meter)
    prior_log_meter = (
        torch.log_softmax(factors["prior_meter_logits"], -1)
        if "prior_meter_logits" in factors
        else torch.zeros_like(state.meter)
    )

    crossed = live & (new_phi >= state.bar_start + state.beat_index * state.spacing)
    landmark = state.bar_start + state.beat_index * state.spacing
    downbeat = crossed & (state.beat_index * state.spacing >= TWO_PI - 1e-4)
    fraction = ((landmark - state.phi) / (new_phi - state.phi).clamp(min=1e-8)).clamp(0.0, 1.0)
    crossing = torch.where(crossed, frame - 1 + fraction, zeros)

    meter_moves = downbeat if meter_gated else live
    meter = torch.where(meter_moves[:, None], next_meter, state.meter)
    spacing = TWO_PI / (meter @ meter_values)
    bar_start = torch.where(downbeat, landmark, state.bar_start)
    beat_index = torch.where(
        downbeat,
        torch.ones_like(state.beat_index),
        torch.where(crossed, state.beat_index + 1, state.beat_index),
    )
    next_state = _PathState(new_phi, new_velocity, meter, bar_start, beat_index, spacing)
    return _PathUpdate(
        next_state,
        prediction,
        new_phi,
        phase_mu,
        phase_kappa,
        velocity_mu,
        velocity_sigma,
        prior_velocity_mu,
        prior_velocity_sigma,
        log_meter,
        prior_log_meter,
        crossed,
        downbeat,
        crossing,
    )


def build_model(cfg, input_dim: int) -> VBPM:
    """One VBPM from a config."""
    emission = EmissionSpec(layers=cfg.emission_layers, positional=cfg.emission_positional)
    walk = WalkSpec(prior_phase_kappa=cfg.prior_phase_kappa)
    prior_spec = PriorSpec(
        dim=cfg.prior_dim,
        layers=cfg.prior_layers,
        phase0_kappa=cfg.prior_phase0_kappa,
        velocity_sigma=cfg.prior_velocity_sigma,
    )
    model = VBPM(
        input_dim,
        meters=tuple(cfg.meters),
        emission=emission,
        walk=walk,
        prior_spec=prior_spec,
        meter_gated=cfg.meter_gated,
        velocity_ref=cfg.velocity_ref if cfg.log_velocity else None,
        velocity_step=cfg.velocity_step if cfg.log_velocity else None,
    )

    return model
