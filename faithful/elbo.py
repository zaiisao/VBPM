"""Algorithm 1 (the SGVB training rollout) and the deploy-path free-run.

Batched port of ``run_algorithm_1`` and ``free_run`` from
``notebooks/build_elbo_notebook.py`` (§5, §10), transcribing *ELBO for DBN* §4 and
Algorithm 1. The objective is the STRICT ELBO:

    L = sum_t BCE(b_t, sigmoid(decoder(z_t, h)))  +  sum_t [KL_meter + KL_phase + KL_tempo]

with beta = 1, a single MC sample over the sampled z_{t-1}, and NOTHING else: no
free-bits, no KL annealing, no latent supervision, no pos_weight, no extra latents.
Anything added here would be a bandage and break faithfulness.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .distributions import (
    TWO_PI, gumbel_softmax, sample_von_mises,
    kl_categorical, kl_von_mises, kl_log_normal,
)


def strict_elbo(model, h: torch.Tensor, b: torch.Tensor, temperature: float = 0.5):
    """Algorithm 1 over a batch. ``h`` is [B,T,n_mels], ``b`` is [B,T] binary beats.

    Returns (loss, info). ``loss`` is the per-sequence negative ELBO averaged over
    the batch; ``info`` holds the breakdown (recon / per-latent KL) for monitoring
    posterior collapse.
    """
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)    # f_phi context, reads (b, h)  [B,T,hidden]
    prior_ctx = model.encode_prior(h)          # f_psi context, reads h       [B,T,hidden]

    kl_m = h.new_zeros(B)
    kl_p = h.new_zeros(B)
    kl_t = h.new_zeros(B)
    z_feats = []                # latent features per step, for the decoder
    post_phase_mu = []          # posterior phase-mean trajectory, for monitoring

    # ---------- t = 1 : initial state (Algorithm 1, lines 7-13) ----------
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_phi_mu, q_phi_k, q_tau_mu, q_tau_s = model.unpack(
        model.post_head(torch.cat([post_ctx[:, 0], z0], dim=-1)))
    p_m, p_phi_mu, p_phi_k, p_tau_mu, p_tau_s = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))

    meter = gumbel_softmax(q_m, temperature)                       # line 9
    phi = sample_von_mises(q_phi_mu, q_phi_k) % TWO_PI             # line 10
    log_tempo = q_tau_mu + q_tau_s * torch.randn_like(q_tau_mu)    # line 11

    kl_m = kl_m + kl_categorical(torch.log_softmax(q_m, -1), torch.log_softmax(p_m, -1))
    kl_p = kl_p + kl_von_mises(q_phi_mu, q_phi_k, p_phi_mu, p_phi_k)
    kl_t = kl_t + kl_log_normal(q_tau_mu, q_tau_s, p_tau_mu, p_tau_s)

    z_feats.append(model.z_features(meter, phi, log_tempo))
    post_phase_mu.append(q_phi_mu)
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo

    # ---------- t = 2..T : transitions (Algorithm 1, lines 14-23) ----------
    for t in range(1, T):
        # posterior reads sampled z_{t-1} (line 15)
        z_prev_feat = model.z_features(meter_prev, phi_prev, log_tempo_prev)
        q_m, q_phi_mu, q_phi_k, q_tau_mu, q_tau_s = model.unpack(
            model.post_head(torch.cat([post_ctx[:, t], z_prev_feat], dim=-1)))

        # prior MEANS are the deterministic bar-pointer dynamics on sampled z_{t-1} (line 16)
        tempo_prev = torch.exp(log_tempo_prev)
        p_phi_mu = (phi_prev + tempo_prev) % TWO_PI                 # mu^p_phi = phi_{t-1}+phidot_{t-1}
        p_phi_k = F.softplus(model.prior_phase_kappa(prior_ctx[:, t]).squeeze(-1)) + 0.01
        p_tau_mu = log_tempo_prev                                   # mu^p_tempo = log phidot_{t-1}
        p_tau_s = F.softplus(model.prior_tempo_sigma(prior_ctx[:, t]).squeeze(-1)) + 1e-3

        # sample current latents from the posterior (lines 17-19)
        meter = gumbel_softmax(q_m, temperature)
        phi = sample_von_mises(q_phi_mu, q_phi_k) % TWO_PI
        log_tempo = q_tau_mu + q_tau_s * torch.randn_like(q_tau_mu)

        # meter prior uses the SAMPLED phi_t (line 21)
        log_pi_p = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t])

        kl_m = kl_m + kl_categorical(torch.log_softmax(q_m, -1), log_pi_p)
        kl_p = kl_p + kl_von_mises(q_phi_mu, q_phi_k, p_phi_mu, p_phi_k)
        kl_t = kl_t + kl_log_normal(q_tau_mu, q_tau_s, p_tau_mu, p_tau_s)

        z_feats.append(model.z_features(meter, phi, log_tempo))
        post_phase_mu.append(q_phi_mu)
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo

    # ---------- decode (lines 24-27) ----------
    beat_logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], dim=1)
    recon = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="none").sum(1)  # [B]
    L_kl = kl_m + kl_p + kl_t                                       # [B]
    loss = (recon + L_kl).mean()                                    # strict ELBO, beta=1

    info = {
        "loss": float(loss), "recon": float(recon.mean()), "kl": float(L_kl.mean()),
        "kl_meter": float(kl_m.mean()), "kl_phase": float(kl_p.mean()), "kl_tempo": float(kl_t.mean()),
        "beat_prob": torch.sigmoid(beat_logits).detach(),
        "post_phase_mu": torch.stack(post_phase_mu, dim=1).detach(),  # [B,T]
    }
    return loss, info


@torch.no_grad()
def free_run(model, h: torch.Tensor, temperature: float = 0.3):
    """Deploy path: roll the PRIOR forward with NO beats (the test-time generative model).

    Returns per-step trajectories for both inference read-outs:
      * ``phase`` / ``phase_mu`` : the stochastic phase sample and the noise-free MEAN
        chain (the phase-wrap read-out detects 2*pi->0 wraps on ``phase_mu``);
      * ``decoder_prob`` : the Bernoulli decoder's beat probability (reads h unless latent-only).
    """
    B, T, _ = h.shape
    prior_ctx = model.encode_prior(h)

    # t = 1: sample from the PRIOR initial state (no posterior -- no beats at deploy time)
    p_m, p_phi_mu, p_phi_k, p_tau_mu, p_tau_s = model.unpack(
        model.prior_init_head(prior_ctx.mean(1)))
    meter = gumbel_softmax(p_m, temperature)
    phi = sample_von_mises(p_phi_mu, p_phi_k) % TWO_PI
    log_tempo = p_tau_mu + p_tau_s * torch.randn_like(p_tau_mu)

    # noise-free mean chain for the phase-wrap read-out
    phi_mu = p_phi_mu % TWO_PI
    log_tempo_mu = p_tau_mu

    z_feats = [model.z_features(meter, phi, log_tempo)]
    phase_traj, phase_mu_traj, log_tempo_traj, meter_traj = [phi], [phi_mu], [log_tempo], [meter.argmax(-1)]
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo

    for t in range(1, T):
        tempo_prev = torch.exp(log_tempo_prev)
        p_phi_mu = (phi_prev + tempo_prev) % TWO_PI
        p_phi_k = F.softplus(model.prior_phase_kappa(prior_ctx[:, t]).squeeze(-1)) + 0.01
        p_tau_mu = log_tempo_prev
        p_tau_s = F.softplus(model.prior_tempo_sigma(prior_ctx[:, t]).squeeze(-1)) + 1e-3

        phi = sample_von_mises(p_phi_mu, p_phi_k) % TWO_PI
        log_tempo = p_tau_mu + p_tau_s * torch.randn_like(p_tau_mu)
        meter = gumbel_softmax(model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t]), temperature)

        # deterministic mean chain (random-walk mean = previous mean -> constant tempo)
        phi_mu = (phi_mu + torch.exp(log_tempo_mu)) % TWO_PI

        z_feats.append(model.z_features(meter, phi, log_tempo))
        phase_traj.append(phi); phase_mu_traj.append(phi_mu)
        log_tempo_traj.append(log_tempo); meter_traj.append(meter.argmax(-1))
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo

    beat_logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], dim=1)
    return {
        "phase": torch.stack(phase_traj, dim=1),            # [B,T] stochastic
        "phase_mu": torch.stack(phase_mu_traj, dim=1),      # [B,T] deterministic mean
        "log_tempo": torch.stack(log_tempo_traj, dim=1),    # [B,T]
        "meter": torch.stack(meter_traj, dim=1),            # [B,T] argmax class
        "decoder_prob": torch.sigmoid(beat_logits),         # [B,T]
    }
