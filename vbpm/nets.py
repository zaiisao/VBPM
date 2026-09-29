"""The encoder trunk, the emission, the prior chain, and the posterior."""
from __future__ import annotations

import math

import torch
from torch import nn

from .constants import FPS, METER0_SHARE, TEMPO0_BPM, TEMPO0_BPM_SD, TWO_PI
from .specs import EmissionSpec, WalkSpec
from .vonmises import sample_vonmises, sample_vonmises_icdf


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

        self.encoder = Encoder(input_dim + 3, d_model, use_pe=encoder_pe)
        self.phase0_head = nn.Linear(d_model, 3)
        self.phase_head = nn.Linear(d_model, 3)
        self.tempo0_head = nn.Linear(d_model, 2)
        self.tempo_head = nn.Linear(d_model, 2)
        self.meter0_head = nn.Linear(d_model, len(prior.meters))
        self.meter_head = nn.Linear(d_model, len(prior.meters))

        for head in (self.phase0_head, self.phase_head, self.tempo0_head, self.tempo_head,
                     self.meter0_head, self.meter_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.phase0_head.bias[0] = 1.0
            self.phase_head.bias[0] = 1.0


    def forward(self, h, labels, mask, tau=1.0):
        """q's parameters and one draw of every latent, all read from (h, labels)."""
        labels_onehot = nn.functional.one_hot(labels, 3).to(h.dtype)
        feats = self.encoder(torch.cat([h, labels_onehot], dim=-1), mask)

        phase0_cos, phase0_sin, phase0_kappa_raw = self.phase0_head(feats[:, 0]).unbind(-1)
        phase0_mu = torch.atan2(phase0_sin, phase0_cos)
        phase0_kappa = nn.functional.softplus(phase0_kappa_raw)

        phase_cos, phase_sin, phase_kappa_raw = self.phase_head(feats).unbind(-1)
        phase_mu = torch.atan2(phase_sin, phase_cos)
        phase_kappa = nn.functional.softplus(phase_kappa_raw)

        tempo0_mu, tempo0_sigma_raw = self.tempo0_head(feats[:, 0]).unbind(-1)
        tempo0_sigma = nn.functional.softplus(tempo0_sigma_raw)

        tempo_change_mu, tempo_change_sigma_raw = self.tempo_head(feats).unbind(-1)
        tempo_change_sigma = nn.functional.softplus(tempo_change_sigma_raw)

        meter0_logits = self.meter0_head(feats[:, 0])
        meter_logits = self.meter_head(feats)
        # JA: q_meter is returned in log form because it is solely used for the KL, and
        # log_softmax is safer than log(softmax) for numerical stability
        log_meter0 = torch.log_softmax(meter0_logits, dim=-1)
        log_meter = torch.log_softmax(meter_logits, dim=-1)

        phase0_draw = phase0_mu + sample_vonmises_icdf(phase0_kappa)
        phase_draw = phase_mu + sample_vonmises_icdf(phase_kappa)

        tempo_draw = tempo_change_mu + torch.randn_like(tempo_change_mu) * tempo_change_sigma
        tempo0_draw = tempo0_mu + torch.randn_like(tempo0_mu) * tempo0_sigma
        tempo_draw = torch.cat([tempo0_draw[:, None], tempo_draw[:, 1:]], 1)

        meter_draw = nn.functional.gumbel_softmax(meter_logits, tau=tau, hard=True)
        meter0_draw = nn.functional.gumbel_softmax(meter0_logits, tau=tau, hard=True)
        meter_draw = torch.cat([meter0_draw[:, None], meter_draw[:, 1:]], 1)

        draw = {"phase0": phase0_draw, "phase": phase_draw, "tempo": tempo_draw,
                "meter": meter_draw}

        q_phi = {"phase0": (phase0_mu, phase0_kappa), "phase": (phase_mu, phase_kappa),
                 "tempo0": (tempo0_mu, tempo0_sigma),
                 "tempo": (tempo_change_mu, tempo_change_sigma),
                 "log_meter0": log_meter0, "log_meter": log_meter}

        return draw, q_phi


class EmissionModel(nn.Module):
    """p(labels | z): a Transformer over the latent sequence (tutorial 9.6)."""

    def __init__(self, spec: EmissionSpec, meters, max_len: int = 4096):
        super().__init__()
        self.spec = spec

        self.proj = nn.Linear(3 + len(meters), spec.dim)
        layer = nn.TransformerEncoderLayer(spec.dim, 4, dim_feedforward=4 * spec.dim,
                                           dropout=0.0, activation="relu",
                                           batch_first=True, norm_first=False)
        self.blocks = nn.TransformerEncoder(layer, spec.layers)
        self.out = nn.Linear(spec.dim, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        if spec.positional:
            self.register_buffer("pe", sinusoidal_encoding(max_len, spec.dim))

    def forward(self, phi, tempo, meter, mask):
        """[B, T, 3] logits over {non-beat, beat, downbeat} from the whole path."""
        z = torch.cat([phi.cos()[..., None], phi.sin()[..., None], tempo[..., None], meter], -1)
        x = self.proj(z)
        if self.spec.positional:
            x = x + self.pe[:x.shape[1]]
        x = self.blocks(x, src_key_padding_mask=mask <= 0)
        return self.out(x)

    def loglik(self, phi, tempo, meter, labels, mask, has_downbeats=None):
        """[B]: log p(labels | path) summed over valid frames."""
        logp = torch.log_softmax(self(phi, tempo, meter, mask), -1)

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


class PriorModel(nn.Module):
    """p(path | x): uniform phase0, audio-read tempo and meter, bar-pointer transitions."""

    def __init__(self, input_dim: int, meters, walk: WalkSpec):
        super().__init__()
        self.meters = tuple(int(m) for m in meters)

        unknown = [m for m in self.meters if m not in METER0_SHARE]
        assert not unknown, f"no starting share for meters {unknown}"
        beats_per_bar = torch.tensor(self.meters, dtype=torch.float32)
        rad_per_bpm = TWO_PI / (60.0 * beats_per_bar * FPS)
        self.register_buffer("tempo0_mu", TEMPO0_BPM * rad_per_bpm, persistent=False)
        self.register_buffer("tempo0_sigma", TEMPO0_BPM_SD * rad_per_bpm, persistent=False)
        self.register_buffer("log_meter0",
                             torch.tensor([METER0_SHARE[m] for m in self.meters]).log(),
                             persistent=False)

        self.tempo_head = nn.Linear(input_dim, 2)
        self.meter_head = nn.Linear(input_dim, len(self.meters))

        for head in (self.tempo_head, self.meter_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.phase_kappa_raw = nn.Parameter(torch.tensor(inverse_softplus(walk.prior_phase_kappa)))

    def forward(self, h):
        """The prior's parameters read from the audio features."""
        phase_kappa = nn.functional.softplus(self.phase_kappa_raw)

        B = h.shape[0]
        tempo0_mu = self.tempo0_mu.expand(B, -1)
        tempo0_sigma = self.tempo0_sigma.expand(B, -1)

        tempo_change_mu, tempo_change_sigma_raw = self.tempo_head(h).unbind(-1)
        tempo_change_sigma = nn.functional.softplus(tempo_change_sigma_raw)

        log_meter0 = self.log_meter0.expand(B, -1)
        log_meter = torch.log_softmax(self.meter_head(h), -1)

        p_psi = {"phase": phase_kappa,
                 "tempo0": (tempo0_mu, tempo0_sigma),
                 "tempo": (tempo_change_mu, tempo_change_sigma),
                 "log_meter0": log_meter0,
                 "log_meter": log_meter}

        return p_psi

    @torch.no_grad()
    def sample(self, h, mask):
        """One draw from the prior, in the relative form draws_to_paths expects."""
        p = self(h)
        B, T = mask.shape

        phase0 = (torch.rand(B, device=h.device) * 2 - 1) * math.pi
        phase = sample_vonmises(p["phase"].expand(B, T).contiguous())

        R = len(self.meters)
        meter_idx = torch.distributions.Categorical(logits=p["log_meter"]).sample()
        meter_idx[:, 0] = torch.distributions.Categorical(logits=p["log_meter0"]).sample()

        tempo0_mu, tempo0_sigma = p["tempo0"]
        batch_idx = torch.arange(B, device=h.device)
        tempo0_mu = tempo0_mu[batch_idx, meter_idx[:, 0]]
        tempo0_sigma = tempo0_sigma[batch_idx, meter_idx[:, 0]]
        tempo_change_mu, tempo_change_sigma = p["tempo"]
        tempo = tempo_change_mu + torch.randn_like(tempo_change_mu) * tempo_change_sigma
        tempo[:, 0] = tempo0_mu + torch.randn_like(tempo0_mu) * tempo0_sigma
        meter = nn.functional.one_hot(meter_idx, R).to(h.dtype)
        return {"phase0": phase0, "phase": phase, "tempo": tempo, "meter": meter}
