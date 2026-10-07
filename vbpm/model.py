"""CVAE/GSNN training and inference for an audio-conditioned latent chain."""

import torch
from torch import nn
from torch.nn import functional as F

from .frontends import FrontendBase
from .nets import EmissionModel, LatentSampler, LatentState, PosteriorModel, PriorModel
from .util.kl import gaussian_kl
from .util.vonmises import kl_vonmises


class VBPM(nn.Module):
    """Roll out prior or posterior paths and evaluate the shared emission."""

    def __init__(self, frontend: FrontendBase, d_model=128, samples=4):
        super().__init__()
        self.samples = samples
        self.frame_period = 1 / frontend.output_fps
        self.prior_model = PriorModel(frontend.num_channels, self.frame_period, d_model)
        self.posterior_model = PosteriorModel(frontend.num_channels, d_model)
        self.latent_sampler = LatentSampler()
        self.emission_model = EmissionModel(d_model)
        self.register_buffer("meter_values", torch.tensor([4.0]), persistent=False)

    def rollout(self, h, samples=1, *, sample=True):
        """Return logits and states shaped [samples, batch, time, ...]."""
        logits, states, _ = self._rollout(h, samples, sample=sample)
        return logits, states

    def posterior_rollout(self, h, labels, samples=1, *, sample=True):
        """Return T label-bearing states and T+1 KLs, including the initial state."""
        context = self.posterior_model.encode(h, labels)
        return self._rollout(h, samples, sample=sample, context=context)

    def _rollout(self, h, samples, *, sample, context=None):
        batch_size, num_frames, input_dim = h.shape
        zero = h.new_zeros(samples * batch_size)
        previous = LatentState(zero, zero)
        logits, phases, velocities = [], [], []
        kl_terms = {"phase": [], "velocity": []}

        # z_0 has no label. The following T transitions generate the T observations.
        for step in range(num_frames + 1):
            initial = step == 0
            frame = max(step - 1, 0)
            audio = h[:, frame].expand(samples, -1, -1).reshape(-1, input_dim)
            parameters = self.prior_model(audio, previous, initial=initial)
            if context is not None:
                frame_context = context[:, frame].expand(samples, -1, -1).reshape(
                    samples * batch_size, -1
                )
                posterior = self.posterior_model(frame_context, previous)
                kl_terms["phase"].append(kl_vonmises(
                    posterior.phase_mean, posterior.phase_concentration,
                    parameters.phase_mean, parameters.phase_concentration,
                ).reshape(samples, batch_size))
                kl_terms["velocity"].append(gaussian_kl(
                    posterior.velocity_mean, posterior.velocity_log_std.exp(),
                    parameters.velocity_mean, parameters.velocity_log_std.exp(),
                ).reshape(samples, batch_size))
                parameters = posterior
            state = self.latent_sampler(
                parameters, initial=initial and context is None, sample=sample
            )
            if not initial:
                logits.append(self.emission_model(state).reshape(samples, batch_size, 3))
                phases.append(state.phase.reshape(samples, batch_size))
                velocities.append(state.velocity.reshape(samples, batch_size))
            previous = state

        states = LatentState(
            torch.stack(phases, dim=2), torch.stack(velocities, dim=2)
        )
        terms = {name: torch.stack(values, dim=2) for name, values in kl_terms.items() if values}
        return torch.stack(logits, dim=2), states, terms

    def forward(self, h, mask, cls, *, gsnn_only=True):
        """Evaluate prior reconstruction and optionally the posterior ELBO."""
        logits, states = self.rollout(h, self.samples)
        targets = cls.unsqueeze(0).expand(self.samples, -1, -1)
        nll = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), reduction="none")
        prior_recon = -(nll.reshape_as(targets) * mask).sum(-1).mean(0)
        path = self._path(LatentState(states.phase[0], states.velocity[0]))
        recon = prior_recon
        kl = torch.zeros_like(prior_recon)
        kl_terms = {}
        if not gsnn_only:
            logits, states, terms = self.posterior_rollout(h, cls, self.samples)
            nll = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), reduction="none")
            recon = -(nll.reshape_as(targets) * mask).sum(-1).mean(0)
            has_observations = mask.any(-1)
            kl_terms = {
                name: ((values[..., 1:] * mask).sum(-1)
                       + values[..., 0] * has_observations).mean(0)
                for name, values in terms.items()
            }
            kl = sum(kl_terms.values())
            posterior_path = self._path(LatentState(states.phase[0], states.velocity[0]))
        else:
            posterior_path = path
        return dict(
            prior_recon=prior_recon,
            recon=recon,
            elbo=recon - kl,
            kl=kl,
            kl_terms=kl_terms,
            prior_path=path,
            path=posterior_path,
            phi=posterior_path["phi_path"],
        )

    def _path(self, state):
        # Evaluation readouts expect phase advance per frame, not radians/second.
        return dict(
            phi_path=state.phase,
            velocity_path=state.velocity * self.frame_period,
            meter_path=state.phase.new_ones((*state.phase.shape, 1)),
        )

    @torch.no_grad()
    def infer_path(self, h, mask=None):
        """Deterministic transition rollout; uniform initial phase uses zero."""
        _, states = self.rollout(h, sample=False)
        state = LatentState(states.phase[0], states.velocity[0])
        path = self._path(state)
        if mask is not None:
            path = {
                name: value * (mask[..., None] if value.ndim == 3 else mask)
                for name, value in path.items()
            }
        return path

    @torch.no_grad()
    def predict_label_probs(self, h, mask=None, rollouts=64):
        """Average emission probabilities across audio-only prior rollouts."""
        logits, _ = self.rollout(h, rollouts)
        probabilities = logits.softmax(-1).mean(0)
        if mask is not None:
            nonbeat = torch.zeros_like(probabilities)
            nonbeat[..., 0] = 1
            probabilities = torch.where(mask[..., None] > 0, probabilities, nonbeat)
        return probabilities


def build_model(cfg, frontend: FrontendBase):
    """Build the fixed-meter model from frontend metadata."""
    return VBPM(frontend, samples=cfg.generator_samples)
