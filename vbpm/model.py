"""VBPM: the bar-pointer CVAE with phase, tempo and meter latents."""
from __future__ import annotations

import torch
from torch import nn

from .constants import TWO_PI
from .nets import EmissionModel, PosteriorModel, PriorModel, gaussian_kl
from .vonmises import kl_vonmises, log_i0, mean_resultant
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

    def draws_to_paths(self, draw, mask):
        """Paths from one draw: tempo held between beats, meter held between downbeats."""
        phi0, phase_draw = draw["phase0"], draw["phase"]
        log_tempo_draw, meter_draw = draw["log_tempo"], draw["meter"]

        B, T = phase_draw.shape
        batch_idx = torch.arange(B, device=mask.device)
        frames = torch.arange(T, device=mask.device)

        phase_step = phase_draw * mask
        phase_step = torch.cat([torch.zeros_like(phase_step[:, :1]), phase_step[:, 1:]], 1)

        phi_path = phi0[:, None].expand(B, T).clone()
        log_tempo_path = log_tempo_draw[:, :1].expand(B, T).clone()
        meter_path = meter_draw[:, :1].expand(B, T, -1).clone()
        is_beat = torch.zeros(B, T, dtype=torch.bool, device=mask.device)
        is_downbeat = torch.zeros(B, T, dtype=torch.bool, device=mask.device)
        crossing = torch.zeros(B, T, device=mask.device)

        seg_start = torch.zeros(B, dtype=torch.long, device=mask.device)
        seg_phi_start = phi0
        seg_log_tempo = log_tempo_draw[:, 0]
        seg_meter = meter_draw[:, 0]
        active = torch.ones(B, dtype=torch.bool, device=mask.device)

        for _ in range(T):
            if not active.any():
                break

            ahead = frames[None, :] > seg_start[:, None]
            steps = (seg_log_tempo.exp()[:, None] * mask + phase_step) * ahead
            seg_phi = seg_phi_start[:, None] + torch.cumsum(steps, 1)

            beat_spacing = TWO_PI / (seg_meter @ self.meter_values)
            bar_start = torch.floor(seg_phi_start / TWO_PI) * TWO_PI
            beat_index = torch.floor((seg_phi_start - bar_start) / beat_spacing) + 1
            landmark = bar_start + beat_index * beat_spacing

            hit = ahead & (seg_phi >= landmark[:, None]) & (mask > 0)
            crossed = hit.any(1) & active
            seg_end = torch.where(crossed, hit.int().argmax(1),
                                  torch.full_like(seg_start, T - 1))

            in_seg = ahead & (frames[None, :] <= seg_end[:, None]) & active[:, None]
            phi_path = torch.where(in_seg, seg_phi, phi_path)
            log_tempo_path = torch.where(in_seg, seg_log_tempo[:, None], log_tempo_path)
            meter_path = torch.where(in_seg[..., None], seg_meter[:, None, :], meter_path)

            downbeat = crossed & (beat_index * beat_spacing >= TWO_PI - 1e-4)
            before = seg_phi[batch_idx, (seg_end - 1).clamp(min=0)]
            after = seg_phi[batch_idx, seg_end]
            fraction = ((landmark - before) / (after - before).clamp(min=1e-8)).clamp(0.0, 1.0)
            crossing[batch_idx[crossed], seg_end[crossed]] = (seg_end - 1 + fraction)[crossed]
            is_beat[batch_idx[crossed], seg_end[crossed]] = True
            is_downbeat[batch_idx[downbeat], seg_end[downbeat]] = True

            seg_phi_start = seg_phi[batch_idx, seg_end]
            new_log_tempo = seg_log_tempo + log_tempo_draw[batch_idx, seg_end]
            seg_log_tempo = torch.where(crossed, new_log_tempo, seg_log_tempo)
            new_meter = meter_draw[batch_idx, seg_end]
            seg_meter = torch.where(downbeat[:, None], new_meter, seg_meter)
            seg_start = seg_end
            active = crossed

        return {"phi_path": phi_path, "log_tempo_path": log_tempo_path, "meter_path": meter_path,
                "is_beat": is_beat, "is_downbeat": is_downbeat, "crossing": crossing}

    def kl(self, h, mask, q_phi, path):
        """[B] KL(q || p) along q's sampled path."""
        p = self.prior_model(h)
        live = mask.clone()
        live[:, 0] = 0.0

        phase0_mu, phase0_kappa = q_phi["phase0"]
        kl_phase0 = phase0_kappa * mean_resultant(phase0_kappa) - log_i0(phase0_kappa)

        phase_mu, phase_kappa = q_phi["phase"]
        kl_phase = kl_vonmises(phase_mu, phase_kappa, torch.zeros_like(phase_mu),
                               p["phase"].expand_as(phase_mu))
        kl_phase = (kl_phase * live).sum(1)

        q_log_tempo0_mu, q_log_tempo0_sigma = q_phi["log_tempo0"]
        p_tempo0_mu, p_tempo0_sigma = p["tempo0"]
        kl_tempo0_by_meter = gaussian_kl(q_log_tempo0_mu[:, None], q_log_tempo0_sigma[:, None],
                                         p_tempo0_mu, p_tempo0_sigma)
        kl_tempo0 = (q_phi["log_meter0"].exp() * kl_tempo0_by_meter).sum(-1)

        q_log_tempo_change_mu, q_log_tempo_change_sigma = q_phi["log_tempo"]
        p_change_mu, p_change_sigma = p["tempo"]
        kl_tempo = gaussian_kl(q_log_tempo_change_mu, q_log_tempo_change_sigma,
                               p_change_mu, p_change_sigma)
        kl_tempo = (kl_tempo * path["is_beat"]).sum(1)

        q_meter0 = q_phi["log_meter0"].exp()
        kl_meter0 = (q_meter0 * (q_phi["log_meter0"] - p["log_meter0"])).sum(-1)

        q_meter = q_phi["log_meter"].exp()
        kl_meter = (q_meter * (q_phi["log_meter"] - p["log_meter"])).sum(-1)
        kl_meter = (kl_meter * path["is_downbeat"]).sum(1)

        return kl_phase0 + kl_phase + kl_tempo0 + kl_tempo + kl_meter0 + kl_meter

    def forward(self, h, mask, y, pos_weight: float = 1.0, cls=None, has_downbeats=None):
        """The ELBO for one batch: recon on q's sampled path minus KL(q || p)."""
        draw, q_phi = self.posterior_model(h, cls, mask, tau=self.tau)
        path = self.draws_to_paths(draw, mask)

        recon = self.emission_model.loglik(path["phi_path"], path["log_tempo_path"],
                                           path["meter_path"], cls, mask, has_downbeats)
        kl = self.kl(h, mask, q_phi, path)
        elbo = recon - kl

        return {"elbo": elbo, "recon": recon, "kl": kl,
                "phi": path["phi_path"], "kappa": q_phi["phase"][1]}

    @torch.no_grad()
    def infer_path(self, h, mask=None):
        """Label-free deployment: a draw from the prior through the bar pointer."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        return self.draws_to_paths(self.prior_model.sample(h, mask), mask)

    @torch.no_grad()
    def emission_probs(self, h, mask=None, path=None):
        """Per-frame downbeat probability along the deployed path."""
        if mask is None:
            mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        if path is None:
            path = self.infer_path(h, mask)
        logits = self.emission_model(path["phi_path"], path["log_tempo_path"],
                                     path["meter_path"], mask)
        return torch.softmax(logits, -1)[..., 2]


def build_model(cfg, input_dim: int) -> VBPM:
    """One VBPM from a config."""
    emission = EmissionSpec(layers=cfg.emission_layers, positional=cfg.emission_positional)
    walk = WalkSpec(prior_phase_kappa=cfg.prior_phase_kappa)
    return VBPM(input_dim, meters=tuple(cfg.meters), emission=emission, walk=walk)


def optimizer(model, cfg):
    """(optimizer, params-to-clip). One Adam group; everything clipped."""
    params = list(model.parameters())
    return torch.optim.Adam(params, lr=cfg.lr), params


def objective(out, beta: float, cfg):
    """Per-crop training objective [B]: the beta-annealed ELBO."""
    return out["recon"] - beta * out["kl"]


def on_epoch(model, cfg, epoch: int) -> None:
    """Nothing is scheduled per epoch."""
