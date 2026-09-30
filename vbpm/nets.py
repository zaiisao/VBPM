"""The encoder trunk, the emission, the prior chain, and the posterior."""
from __future__ import annotations

import math

import torch
from torch import nn

from .constants import (CLASS_FREQ, KAPPA_Q_MIN, LOG_TEMPO0, LOG_TEMPO_CHANGE_SD,
                        LOG_TEMPO_CHANGE_SD_Q0, METER0_SHARE, METER_TRANSITION)
from .specs import EmissionSpec, WalkSpec
from .vonmises import sample_vonmises_icdf


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding [length, dim]."""
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                      * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * scale)
    pe[:, 1::2] = torch.cos(pos * scale)
    return pe


class Encoder(nn.Module):
    """The shared trunk (tutorial 9.2), reading AUDIO ONLY.

    It produces context, not posterior parameters: the variants build their own
    heads on top of its output.
    """

    def __init__(self, input_dim: int, d_model: int = 128, heads: int = 4, layers: int = 2,
                 max_len: int = 4096, use_pe: bool = False):
        super().__init__()
        self.d_model = d_model
        self.use_pe = use_pe

        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, activation="relu",
                                           batch_first=True, norm_first=False)
        self.blocks = nn.TransformerEncoder(layer, layers)

        if use_pe:
            self.register_buffer("pe", sinusoidal_encoding(max_len, d_model))

    def forward(self, h, mask=None):
        """[B, T, D] -> [B, T, d_model]: the trunk shared by every head."""
        pad = None if mask is None else (mask <= 0)

        h = self.proj(h) * math.sqrt(self.d_model)
        if self.use_pe:
            h = h + self.pe[:h.shape[1]]

        return self.blocks(h, src_key_padding_mask=pad)


class PosteriorModel(nn.Module):
    """q(path | x) = p_prior(path) exp(sum_t evidence_t(phi_t) + head(c_0)) / Z.

    Marginal carriage: the chain prior tilted by per-frame recognition potentials
    read locally from the trunk, combined by exact forward-backward over the
    prior's quadrature. Nothing is sampled, so C_commit = 0 and

        KL(q || p) = E_q[sum_t evidence_t + head] - log Z        (exact, >= 0).

    The prior is an ARGUMENT of smooth(), not a submodule: it appears in q's
    definition, but it belongs to the generative model and must not be
    registered twice.
    """

    def __init__(self, input_dim: int, d_model: int, prior: "PriorModel",
                 encoder_pe: bool = False):
        super().__init__()

        n_meters = len(prior.meters)
        state_dim = 3 + n_meters

        self.encoder = Encoder(input_dim + 3, d_model, use_pe=encoder_pe)

        self.phase0_head = nn.Linear(d_model, 3)
        self.log_tempo0_head = nn.Linear(d_model, 2)
        self.meter0_head = nn.Linear(d_model, n_meters)

        self.phase_head = nn.Linear(d_model + state_dim, 3)
        self.log_tempo_head = nn.Linear(d_model + state_dim, 2)
        self.meter_head = nn.Linear(d_model + state_dim, n_meters)

        self.init_heads(prior, d_model)

    def init_heads(self, prior, d_model):
        """Set every head so that q starts at the prior's values."""
        feature_weight_sd = 0.01
        phase_feature_weight_sd = 1e-4
        for head in (self.phase0_head, self.log_tempo0_head, self.meter0_head,
                     self.phase_head, self.log_tempo_head, self.meter_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            sd = phase_feature_weight_sd if head is self.phase_head else feature_weight_sd
            nn.init.normal_(head.weight[:, :d_model], std=sd)

        phase0_mean = torch.tensor(0.0)
        phase0_direction = torch.stack([torch.cos(phase0_mean), torch.sin(phase0_mean)])
        phase0_kappa_raw = inverse_softplus(KAPPA_Q_MIN)

        tempo0_mean, tempo0_sd = log_tempo0_mixture(prior.meters)
        tempo0_sd_raw = inverse_softplus(tempo0_sd)
        meter0_log_share = torch.tensor([METER0_SHARE[m] for m in prior.meters]).log()

        phase_log_kappa = float(prior.phase_log_kappa)
        tempo_change_sd_raw = inverse_softplus(LOG_TEMPO_CHANGE_SD_Q0)
        meter_stay_logits = prior.log_meter_transition.T

        with torch.no_grad():
            self.phase0_head.bias[:2] = phase0_direction
            self.phase0_head.bias[2] = phase0_kappa_raw
            self.log_tempo0_head.bias[0] = tempo0_mean
            self.log_tempo0_head.bias[1] = tempo0_sd_raw
            self.meter0_head.bias.copy_(meter0_log_share)
            self.phase_head.weight[0, d_model] = 1.0
            self.phase_head.weight[1, d_model + 1] = 1.0
            self.phase_head.bias[2] = phase_log_kappa
            self.log_tempo_head.bias[1] = tempo_change_sd_raw
            self.meter_head.weight[:, d_model + 3:] = meter_stay_logits

    def forward(self, h, labels, mask, tau=1.0):
        """q's initial-state parameters and draws, and the context the per-frame step reads."""
        labels_onehot = nn.functional.one_hot(labels, 3).to(h.dtype)
        feats = self.encoder(torch.cat([h, labels_onehot], dim=-1), mask)

        phase0_cos, phase0_sin, phase0_kappa_raw = self.phase0_head(feats[:, 0]).unbind(-1)
        phase0_mu = torch.atan2(phase0_sin, phase0_cos)
        phase0_kappa = nn.functional.softplus(phase0_kappa_raw)

        log_tempo0_mu, log_tempo0_sigma_raw = self.log_tempo0_head(feats[:, 0]).unbind(-1)
        log_tempo0_sigma = nn.functional.softplus(log_tempo0_sigma_raw)

        meter0_logits = self.meter0_head(feats[:, 0])
        # JA: q_meter is returned in log form because it is solely used for the KL, and
        # log_softmax is safer than log(softmax) for numerical stability
        log_meter0 = torch.log_softmax(meter0_logits, dim=-1)

        B, T = feats.shape[:2]
        phase0_draw = phase0_mu + sample_vonmises_icdf(phase0_kappa)
        log_tempo_draw = feats.new_zeros(B, T)
        log_tempo_draw[:, 0] = log_tempo0_mu + torch.randn_like(log_tempo0_mu) * log_tempo0_sigma
        meter_draw = feats.new_zeros(B, T, len(log_meter0[0]))
        meter_draw[:, 0] = nn.functional.gumbel_softmax(meter0_logits, tau=tau, hard=True)

        draw = {"phase0": phase0_draw, "log_tempo": log_tempo_draw, "meter": meter_draw}
        q_phi = {"phase0": (phase0_mu, phase0_kappa), "feats": feats,
                 "log_tempo0": (log_tempo0_mu, log_tempo0_sigma), "log_meter0": log_meter0}
        return draw, q_phi

    def step(self, feats_k, pred, tempo_drift, meter):
        """q's frame-k factors from c_k and the sampled state, with tempo as drift from its start."""
        state = torch.cat([pred.cos()[:, None], pred.sin()[:, None], tempo_drift[:, None], meter],
                          -1)
        inputs = torch.cat([feats_k, state], -1)
        phase_cos, phase_sin, phase_log_kappa = self.phase_head(inputs).unbind(-1)
        raw_mu, sigma_raw = self.log_tempo_head(inputs).unbind(-1)
        return {"phase": (torch.atan2(phase_sin, phase_cos), phase_log_kappa.exp()),
                "tempo": (LOG_TEMPO_CHANGE_SD * raw_mu, nn.functional.softplus(sigma_raw)),
                "meter_logits": self.meter_head(inputs)}


class EmissionModel(nn.Module):
    """p(labels | z): a Transformer over the latent sequence (tutorial 9.6)."""

    def __init__(self, spec: EmissionSpec, meters, max_len: int = 4096):
        super().__init__()
        self.spec = spec

        self.proj = nn.Linear(2 + len(meters), spec.dim)
        layer = nn.TransformerEncoderLayer(spec.dim, 4, dim_feedforward=4 * spec.dim,
                                           dropout=0.0, activation="relu",
                                           batch_first=True, norm_first=False)
        self.blocks = nn.TransformerEncoder(layer, spec.layers)
        self.out = nn.Linear(spec.dim, 3)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor(CLASS_FREQ).log())

        if spec.positional:
            self.register_buffer("pe", sinusoidal_encoding(max_len, spec.dim))

    def forward(self, phi, meter, mask):
        """[B, T, 3] logits over {non-beat, beat, downbeat} from the whole path."""
        z = torch.cat([phi.cos()[..., None], phi.sin()[..., None], meter], -1)
        x = self.proj(z)
        if self.spec.positional:
            x = x + self.pe[:x.shape[1]]
        x = self.blocks(x, src_key_padding_mask=mask <= 0)
        return self.out(x)

    def loglik(self, phi, meter, labels, mask, has_downbeats=None):
        """[B]: log p(labels | path) summed over valid frames."""
        logp = torch.log_softmax(self(phi, meter, mask), -1)

        if has_downbeats is not None:
            union = torch.logaddexp(logp[..., 1], logp[..., 2])
            merged = torch.stack([logp[..., 0], union, union], -1)
            logp = torch.where(has_downbeats.reshape(-1, 1, 1).bool(), logp, merged)

        ll = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (ll * mask).sum(1)


def gaussian_kl(mu_q, sigma_q, mu_p, sigma_p):
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)), closed form, elementwise."""
    return (torch.log(sigma_p / sigma_q)
            + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2) - 0.5)


def inverse_softplus(x: float) -> float:
    """The raw value whose softplus is x."""
    return x + math.log(-math.expm1(-x))


def log_tempo0_mixture(meters) -> tuple[float, float]:
    """(mean, sd) of the starting log tempo, mixed over meters by their starting shares."""
    share0 = torch.tensor([METER0_SHARE[m] for m in meters])
    mu0 = torch.tensor([LOG_TEMPO0[m][0] for m in meters])
    sd0 = torch.tensor([LOG_TEMPO0[m][1] for m in meters])
    mix_mu = float((share0 * mu0).sum())
    mix_sd = float(((share0 * (sd0 ** 2 + mu0 ** 2)).sum() - mix_mu ** 2).sqrt())
    return mix_mu, mix_sd


class PriorModel(nn.Module):
    """p(path | x): uniform phase0, audio-read tempo and meter, bar-pointer transitions."""

    def __init__(self, input_dim: int, meters, walk: WalkSpec):
        super().__init__()
        self.meters = tuple(int(m) for m in meters)

        unknown = [m for m in self.meters if m not in METER0_SHARE]
        assert not unknown, f"no starting share for meters {unknown}"
        self.register_buffer("log_tempo0_mu",
                             torch.tensor([LOG_TEMPO0[m][0] for m in self.meters]),
                             persistent=False)
        self.register_buffer("log_tempo0_sigma",
                             torch.tensor([LOG_TEMPO0[m][1] for m in self.meters]),
                             persistent=False)
        self.register_buffer("log_meter0",
                             torch.tensor([METER0_SHARE[m] for m in self.meters]).log(),
                             persistent=False)

        self.register_buffer("log_meter_transition",
                             torch.tensor([[METER_TRANSITION[a][b] for b in self.meters]
                                           for a in self.meters]).log(),
                             persistent=False)

        self.phase_log_kappa = nn.Parameter(torch.tensor(math.log(walk.prior_phase_kappa)))

    def forward(self, h):
        """The prior's parameters read from the audio features."""
        phase_kappa = self.phase_log_kappa.exp()

        B = h.shape[0]
        log_tempo0_mu = self.log_tempo0_mu.expand(B, -1)
        log_tempo0_sigma = self.log_tempo0_sigma.expand(B, -1)

        log_tempo_change_mu = h.new_zeros(h.shape[:2])
        log_tempo_change_sigma = torch.full_like(log_tempo_change_mu, LOG_TEMPO_CHANGE_SD)

        log_meter0 = self.log_meter0.expand(B, -1)

        p_psi = {"phase": phase_kappa,
                 "log_tempo0": (log_tempo0_mu, log_tempo0_sigma),
                 "log_tempo": (log_tempo_change_mu, log_tempo_change_sigma),
                 "log_meter0": log_meter0,
                 "log_meter_transition": self.log_meter_transition}

        return p_psi

    @torch.no_grad()
    def sample(self, h, mask):
        """One draw from the prior, in the relative form draws_to_paths expects."""
        p = self(h)
        B, T = mask.shape

        phase0 = (torch.rand(B, device=h.device) * 2 - 1) * math.pi

        R = len(self.meters)
        meter0_idx = torch.distributions.Categorical(logits=p["log_meter0"]).sample()

        log_tempo0_mu, log_tempo0_sigma = p["log_tempo0"]
        batch_idx = torch.arange(B, device=h.device)
        log_tempo0_mu = log_tempo0_mu[batch_idx, meter0_idx]
        log_tempo0_sigma = log_tempo0_sigma[batch_idx, meter0_idx]
        log_tempo_change_mu, log_tempo_change_sigma = p["log_tempo"]
        log_tempo = (log_tempo_change_mu
                     + torch.randn_like(log_tempo_change_mu) * log_tempo_change_sigma)
        log_tempo[:, 0] = log_tempo0_mu + torch.randn_like(log_tempo0_mu) * log_tempo0_sigma
        meter = h.new_zeros(B, T, R)
        meter[:, 0] = nn.functional.one_hot(meter0_idx, R).to(h.dtype)
        return {"phase0": phase0, "log_tempo": log_tempo, "meter": meter}
