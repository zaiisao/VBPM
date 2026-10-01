"""VBPM: the bar-pointer CVAE with phase, tempo and meter latents."""
from __future__ import annotations

import torch
from torch import nn

from .constants import TWO_PI
from .nets import EmissionModel, PosteriorModel, PriorModel
from .util.kl import gaussian_kl, gaussian_laplace_kl
from .util.vonmises import kl_vonmises, sample_vonmises_icdf
from .specs import EmissionSpec, WalkSpec

DEFAULTS = {}


class VBPM(nn.Module):
    """Posterior, prior and emission, joined by draws_to_paths."""

    def __init__(self, input_dim: int, d_model: int = 128, meters=(3, 4),
                 emission: EmissionSpec | None = None, walk: WalkSpec | None = None,
                 encoder_pe: bool = False):
        super().__init__()

        self.register_buffer("meter_values", torch.tensor(meters, dtype=torch.float32),
                             persistent=False)
        self.tau = 1.0

        self.emission_model = EmissionModel(emission or EmissionSpec(), meters)
        self.prior_model = PriorModel(input_dim, meters, walk or WalkSpec())
        self.posterior_model = PosteriorModel(input_dim, d_model, self.prior_model,
                                              encoder_pe=encoder_pe)

    @property
    def deployed_net(self):
        """The network that runs at test time; it reads audio only."""
        return self.prior_model

    def draws_to_paths(self, draw, mask, posterior=None, feats=None, label_beats=None,
                       prior=None, tau=1.0):
        """Paths from one draw, frame by frame, with factors from the posterior, prior or draw."""
        phi0, log_tempo_draw, meter_draw = draw["phase0"], draw["log_tempo"], draw["meter"]

        B, T = log_tempo_draw.shape

        phi = phi0
        log_tempo = log_tempo_draw[:, 0]
        meter = meter_draw[:, 0]
        spacing = TWO_PI / (meter @ self.meter_values)
        bar_start = torch.floor(phi / TWO_PI) * TWO_PI
        beat_index = torch.floor((phi - bar_start) / spacing) + 1
        landmark = bar_start + beat_index * spacing

        zeros = torch.zeros_like(phi)
        phis, log_tempos, meters = [phi], [log_tempo], [meter]
        mus, kappas, preds = [phi], [zeros], [phi]
        tempo_mus, tempo_sigmas = [zeros], [zeros + 1.0]
        log_meters = [torch.zeros_like(meter)]
        is_beat = torch.zeros(B, T, dtype=torch.bool, device=mask.device)
        is_downbeat = torch.zeros(B, T, dtype=torch.bool, device=mask.device)
        crossing = torch.zeros(B, T, device=mask.device)

        for k in range(1, T):
            live = mask[:, k] > 0
            beats_per_bar = meter @ self.meter_values
            pred = phi + log_tempo.exp() / beats_per_bar * mask[:, k]
            drift = log_tempo - log_tempo_draw[:, 0]
            if posterior is not None:
                factors = posterior.step(feats[:, k], label_beats[:, k], pred, drift, meter)
            elif prior is not None:
                factors = {"phase": (pred, prior["phase"].expand_as(pred)),
                           "meter_logits": meter @ prior["log_meter_transition"]}
            else:
                factors = {}

            if "phase" in factors:
                mu, kappa = factors["phase"]
                offset = torch.atan2(torch.sin(mu - pred), torch.cos(mu - pred))
                new_phi = pred + offset + sample_vonmises_icdf(kappa)
            else:
                mu, kappa, new_phi = pred, zeros, pred
            new_phi = torch.where(live, new_phi, phi)

            if "tempo" in factors:
                tempo_mu, tempo_sigma = factors["tempo"]
                change = tempo_mu + torch.randn_like(tempo_mu) * tempo_sigma
            else:
                tempo_mu, tempo_sigma = zeros, zeros + 1.0
                change = log_tempo_draw[:, k]

            if "meter_logits" in factors:
                logits = factors["meter_logits"]
                meter_k = nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
                log_meter = torch.log_softmax(logits, -1)
            else:
                meter_k = meter_draw[:, k]
                log_meter = torch.zeros_like(meter)

            crossed = live & (new_phi >= landmark)
            downbeat = crossed & (beat_index * spacing >= TWO_PI - 1e-4)
            fraction = ((landmark - phi) / (new_phi - phi).clamp(min=1e-8)).clamp(0.0, 1.0)
            crossing[:, k] = torch.where(crossed, k - 1 + fraction, crossing[:, k])
            is_beat[:, k] = crossed
            is_downbeat[:, k] = downbeat

            phis.append(new_phi)
            log_tempos.append(log_tempo)
            meters.append(meter)
            mus.append(mu)
            kappas.append(kappa)
            preds.append(pred)
            tempo_mus.append(tempo_mu)
            tempo_sigmas.append(tempo_sigma)
            log_meters.append(log_meter)

            phi = new_phi
            log_tempo = torch.where(crossed, log_tempo + change, log_tempo)
            meter = torch.where(downbeat[:, None], meter_k, meter)
            spacing = TWO_PI / (meter @ self.meter_values)
            bar_start = torch.where(downbeat, landmark, bar_start)
            beat_index = torch.where(downbeat, torch.ones_like(beat_index),
                                     torch.where(crossed, beat_index + 1, beat_index))
            landmark = torch.where(crossed, bar_start + beat_index * spacing, landmark)

        return {"phi_path": torch.stack(phis, 1), "log_tempo_path": torch.stack(log_tempos, 1),
                "meter_path": torch.stack(meters, 1), "is_beat": is_beat,
                "is_downbeat": is_downbeat, "crossing": crossing,
                "phase_mu": torch.stack(mus, 1), "phase_kappa": torch.stack(kappas, 1),
                "phase_pred": torch.stack(preds, 1),
                "tempo_mu": torch.stack(tempo_mus, 1), "tempo_sigma": torch.stack(tempo_sigmas, 1),
                "log_meter": torch.stack(log_meters, 1)}

    def kl(self, h, mask, q_phi, path):
        """[B] KL(q || p) along q's sampled path."""
        p = self.prior_model(h)
        live = mask.clone()
        live[:, 0] = 0.0

        kl_phase0 = kl_vonmises(*q_phi["phase0"], *p["phase0"])

        kl_phase = kl_vonmises(path["phase_mu"], path["phase_kappa"], path["phase_pred"],
                               p["phase"].expand_as(path["phase_mu"]))
        kl_phase = (kl_phase * live).sum(1)

        q_log_tempo0_mu, q_log_tempo0_sigma = q_phi["log_tempo0"]
        p_log_tempo0_mu, p_log_tempo0_sigma = p["log_tempo0"]
        kl_tempo0_by_meter = gaussian_kl(q_log_tempo0_mu[:, None], q_log_tempo0_sigma[:, None],
                                         p_log_tempo0_mu, p_log_tempo0_sigma)
        kl_tempo0 = (q_phi["log_meter0"].exp() * kl_tempo0_by_meter).sum(-1)

        p_log_tempo_change_mu, p_log_tempo_change_scale = p["log_tempo"]
        kl_tempo = gaussian_laplace_kl(path["tempo_mu"], path["tempo_sigma"],
                                       p_log_tempo_change_mu, p_log_tempo_change_scale)
        kl_tempo = (kl_tempo * path["is_beat"]).sum(1)

        q_meter0 = q_phi["log_meter0"].exp()
        kl_meter0 = (q_meter0 * (q_phi["log_meter0"] - p["log_meter0"])).sum(-1)

        q_meter = path["log_meter"].exp()
        p_log_meter = path["meter_path"] @ p["log_meter_transition"]
        kl_meter = (q_meter * (path["log_meter"] - p_log_meter)).sum(-1)
        kl_meter = (kl_meter * path["is_downbeat"]).sum(1)

        return kl_phase0 + kl_phase + kl_tempo0 + kl_tempo + kl_meter0 + kl_meter

    def forward(self, h, mask, cls=None):
        """The ELBO for one batch: recon on q's sampled path minus KL(q || p)."""
        draw, q_phi = self.posterior_model(h, cls, mask, tau=self.tau)
        feats = q_phi["feats"]
        path = self.draws_to_paths(draw, mask, posterior=self.posterior_model, feats=feats,
                                   label_beats=q_phi["label_beats"], tau=self.tau)

        recon = self.emission_model.loglik(path["phi_path"], path["meter_path"], cls, mask)
        kl = self.kl(h, mask, q_phi, path)
        elbo = recon - kl

        return {"elbo": elbo, "recon": recon, "kl": kl, "path": path,
                "phi": path["phi_path"], "kappa": path["phase_kappa"]}

    @torch.no_grad()
    def infer_path(self, h, mask=None):
        """Label-free deployment: a draw from the prior through the bar pointer."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        draw = self.prior_model.sample(h, mask)
        return self.draws_to_paths(draw, mask, prior=self.prior_model(h))

    @torch.no_grad()
    def emission_probs(self, h, mask=None, path=None):
        """Per-frame downbeat probability along the deployed path."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        if path is None:
            path = self.infer_path(h, mask)
        logits = self.emission_model(path["phi_path"], path["meter_path"], mask)
        return torch.softmax(logits, -1)[..., 2]


def build_model(cfg, input_dim: int) -> VBPM:
    """One VBPM from a config."""
    emission = EmissionSpec(layers=cfg.emission_layers, positional=cfg.emission_positional)
    walk = WalkSpec(prior_phase_kappa=cfg.prior_phase_kappa)
    model = VBPM(input_dim, meters=tuple(cfg.meters), emission=emission, walk=walk)

    return model
