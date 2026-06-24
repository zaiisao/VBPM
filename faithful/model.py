"""The faithful bar-pointer VAE (priors p_psi, posteriors q_phi, decoder p_theta).

Batched port of ``BarPointerVAE`` from ``notebooks/build_elbo_notebook.py`` (§4),
which transcribes *ELBO for DBN* §3 and §5. The ONLY substantive differences from
the notebook are mechanical (operate over a batch dimension B) plus:

  * the observation ``h`` is a ``[B, T, n_mels]`` log-mel spectrogram (the model is
    trained END-TO-END FROM RANDOM WEIGHTS; there is no pretrained frontend), and
  * a ``latent_only`` flag that, when set, removes ``h`` from the decoder. This is a
    DOCUMENTED DEVIATION (the paper's §5.4 decoder is p_theta(b_t | z_t, h), i.e. it
    DOES read h); it exists only to contrast the shortcut-driven collapse.

Faithful prior structure (no audio-driven correction of the prior MEAN):
  * phase prior mean   mu^p_phi   = phi_{t-1} + phidot_{t-1}        (bar-pointer advance)
  * tempo prior mean   mu^p_tempo = log phidot_{t-1}                (log-space random walk)
  * phase prior kappa, tempo prior sigma, and the meter transition matrix DO read h
    (concentrations/transition, not the recursion mean) -- this is faithful to §5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import TWO_PI


class BarPointerVAE(nn.Module):
    def __init__(self, h_dim: int, hidden: int = 64, num_meters: int = 4,
                 latent_only: bool = False):
        super().__init__()
        self.K = num_meters
        self.hidden = hidden
        self.latent_only = latent_only
        z_feat_dim = 3 + num_meters            # cos phi, sin phi, log tempo, onehot(meter)
        self.z_feat_dim = z_feat_dim
        param_dim = num_meters + 2 + 1 + 1 + 1  # meter logits | phase(u,v) | log-kappa | tempo mu | log-sigma
        self.param_dim = param_dim

        # f_phi : inference read of (b, h)
        self.post_gru = nn.GRU(h_dim + 1, hidden, batch_first=True, bidirectional=True)
        self.post_ctx = nn.Linear(2 * hidden, hidden)
        # f_psi : generative read of h
        self.prior_gru = nn.GRU(h_dim, hidden, batch_first=True, bidirectional=True)
        self.prior_ctx = nn.Linear(2 * hidden, hidden)

        # posterior head: [context_t, z_{t-1} features] -> distribution params
        self.post_head = nn.Sequential(
            nn.Linear(hidden + z_feat_dim, hidden), nn.Tanh(), nn.Linear(hidden, param_dim))
        self.z0 = nn.Parameter(torch.zeros(z_feat_dim))   # learned initial token

        # prior heads
        self.prior_init_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, param_dim))
        self.prior_phase_kappa = nn.Linear(hidden, 1)     # f^phi_psi(h)  -> concentration only
        self.prior_tempo_sigma = nn.Linear(hidden, 1)     # f^phidot_psi(h) -> sigma only
        self.meter_prior = nn.Sequential(                 # f^m_psi(m_{t-1}, phi_t, phi_{t-1}, h)
            nn.Linear(num_meters + 4 + hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, num_meters * num_meters))

        # decoder NN_theta(z_t, h)  -- latent-only drops the h context (documented deviation)
        dec_in = z_feat_dim if latent_only else z_feat_dim + hidden
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    # ---- shared sequence encoders ----
    def encode_posterior(self, h: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, b.unsqueeze(-1)], dim=-1)        # [B, T, h_dim+1]
        out, _ = self.post_gru(x)
        return torch.tanh(self.post_ctx(out))              # [B, T, hidden]

    def encode_prior(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.prior_gru(h)
        return torch.tanh(self.prior_ctx(out))             # [B, T, hidden]

    # ---- unpack a batched raw parameter vector into named distribution params ----
    def unpack(self, vec: torch.Tensor):
        K = self.K
        meter_logits = vec[:, :K]                          # [B, K]
        u, v = vec[:, K], vec[:, K + 1]                    # [B]
        phase_mu = torch.atan2(v, u) % TWO_PI              # [B]
        phase_kappa = F.softplus(vec[:, K + 2]) + 0.01     # [B]
        tempo_mu = vec[:, K + 3]                           # [B]
        tempo_sigma = F.softplus(vec[:, K + 4]) + 1e-3     # [B]
        return meter_logits, phase_mu, phase_kappa, tempo_mu, tempo_sigma

    def z_features(self, meter_soft, phi, log_tempo):
        # meter_soft [B,K], phi [B], log_tempo [B] -> [B, z_feat_dim]
        return torch.cat([torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1),
                          log_tempo.unsqueeze(-1), meter_soft], dim=-1)

    # ---- prior meter transition (returns log prior over m_t), batched ----
    def meter_prior_logp(self, meter_prev, phi_t, phi_prev, prior_ctx_t):
        feats = torch.cat([meter_prev,
                           torch.cos(phi_t).unsqueeze(-1), torch.sin(phi_t).unsqueeze(-1),
                           torch.cos(phi_prev).unsqueeze(-1), torch.sin(phi_prev).unsqueeze(-1),
                           prior_ctx_t], dim=-1)                       # [B, K+4+hidden]
        Pi = F.softmax(self.meter_prior(feats).reshape(-1, self.K, self.K), dim=2)  # rows: from-meter
        pi_p = torch.bmm(meter_prev.unsqueeze(1), Pi).squeeze(1)       # [B, K]
        return torch.log(pi_p + 1e-9)

    def decode(self, z_feat, prior_ctx_t):
        x = z_feat if self.latent_only else torch.cat([z_feat, prior_ctx_t], dim=-1)
        return self.decoder(x).squeeze(-1)                            # [B] beat logit
