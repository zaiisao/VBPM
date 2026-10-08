"""
vae_dbn_discrete.py -- a DISCRETE reference version of the VAE-DBN, for DEBUGGING.

Why this file exists
--------------------
The continuous VAE-DBN (vae_dbn.py) is trained by an *approximate* objective: a
variational ELBO whose phase factor needs the custom von Mises implicit-reparameterisation
node so that gradients can flow through a *sampled* circular latent. Most "it won't train"
bugs live precisely there: a VonMises `.sample()` used where a reparameterised `.rsample()`
is needed (so no gradient reaches the phase head), a sign error in the ELBO, a decoder that
bypasses the latent, KL not annealed, etc.

This file removes ALL of that machinery. It discretises the latent chain
    z_k = (phase phi_k, velocity v_k, meter m_k)
onto a finite grid, which turns the model into a plain HMM / dynamic Bayesian network whose
latent is a single composite categorical state s_k = (i_phi, i_v, m). For a discrete-state
chain the posterior and the marginal likelihood are available IN CLOSED FORM by the
forward--backward algorithm, so:

  * training maximises the EXACT log-likelihood log p(x, b)  -- no ELBO, no sampling, no
    reparameterisation, no KL term, nothing to anneal;
  * the gradient is the exact gradient of an exact objective, so if the model and data are
    sound it *must* train (loss must decrease), deterministically.

That makes it a debugging ORACLE:

  * If this discrete model trains but your continuous VAE-DBN does not  ==> the bug is in the
    continuous / reparameterisation / ELBO machinery (first suspect: the von Mises gradient,
    i.e. `.sample()` instead of the implicit-reparam `.rsample()`), NOT in the model or data.
  * If even this discrete model cannot fit the data  ==> the bug is in the model structure,
    the emission, or the data generation -- look there first.

It is, deliberately, the discretised bar-pointer model (Whiteley-Cemgil-Godsill; Krebs et al.):
deterministic-ish phase advance by velocity with a von-Mises-shaped spread, a stochastic
velocity random walk, a categorical meter, and a soft observation of the audio x and the beat
labels b. Everything is exact. The latent chain here emits BOTH x (audio) and b (labels), so
no recognition network or conditioning trick is needed; x and b are simply observations.

Run:
    python vae_dbn_discrete.py --mode train     # fit by exact MLE; loss must fall
    python vae_dbn_discrete.py --mode decode     # Viterbi MAP latent path + label accuracy
"""
import math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

PI = math.pi
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = -1e9  # log(0) stand-in


# ---------------------------------------------------------------------------
# Data: reuse the SAME synthetic generator as the continuous model, so the two
# models see identical sequences. We only need (x, b); the exact HMM marginalises
# the latent, so the ground-truth phi/v/m are not used for training.
# ---------------------------------------------------------------------------
from vae_dbn import synth  # x:[B,T,4] audio features, b:[B,T] labels in {0=N,1=B,2=D}


class DiscreteDBN(nn.Module):
    """Discretised (phase x velocity x meter) latent chain, exact-inference HMM.

    State  s = (i_phi, i_v, m),  flattened as  s = i_phi*(Nv*R) + i_v*R + m,  S = Nphi*Nv*R.
    Transition factorises exactly as in the continuous model:
        p(s_k | s_{k-1}) = p(i_phi_k | i_phi_{k-1}, i_v_{k-1}) p(i_v_k | i_v_{k-1}) p(m_k | m_{k-1})
      - phase : von-Mises-shaped around the velocity-driven advance  phi_{k-1}+v_{k-1}*Delta
                (one learnable concentration kappa),
      - velocity : a free Nv x Nv random-walk table,
      - meter : a free R x R table.
    Observations (both soft, hence data has non-zero probability):
      - labels b_k ~ Cat over {N,B,D} with a learnable table over (i_phi, m),
      - audio  x_k ~ Gaussian with a learnable mean table over (i_phi, m) and a learnable scale.
    """
    def __init__(self, Nphi=16, Nv=4, R=3, x_dim=4, vmin=0.05, vmax=0.70, Delta=1.0):
        super().__init__()
        self.Nphi, self.Nv, self.R, self.S = Nphi, Nv, R, Nphi * Nv * R
        self.Delta = Delta
        # ---- fixed bin centres -------------------------------------------------
        self.register_buffer("phi_c", (torch.arange(Nphi) + 0.5) * (2 * PI / Nphi))   # [Nphi] in [0,2pi)
        self.register_buffer("v_c",   torch.linspace(vmin, vmax, Nv))                  # [Nv]
        # ---- transition parameters --------------------------------------------
        self.log_kappa = nn.Parameter(torch.tensor(1.0))        # phase concentration (softplus'd)
        self.vel_T  = nn.Parameter(0.01 * torch.randn(Nv, Nv))  # velocity random-walk logits [v',v]
        self.meter_T = nn.Parameter(0.01 * torch.randn(R, R))   # meter transition logits    [m',m]
        self.init_v = nn.Parameter(torch.zeros(Nv))             # initial velocity logits
        self.init_m = nn.Parameter(torch.zeros(R))              # initial meter logits
        # ---- emission parameters ----------------------------------------------
        self.emit_b = nn.Parameter(0.01 * torch.randn(Nphi, R, 3))   # p(b | i_phi, m) logits
        self.mu_x   = nn.Parameter(0.01 * torch.randn(Nphi, R, x_dim))  # E[x | i_phi, m]
        self.logsig_x = nn.Parameter(torch.zeros(x_dim))             # obs scale (shared over states)

    # ---- transition log-matrix  logT[s', s]  (built from the factors) ----------
    def log_transition(self):
        Nphi, Nv, R = self.Nphi, self.Nv, self.R
        kappa = F.softplus(self.log_kappa) + 1e-3
        adv = self.phi_c[:, None] + self.v_c[None, :] * self.Delta            # [Nphi', Nv'] advance
        diff = self.phi_c[None, None, :] - adv[:, :, None]                    # [Nphi', Nv', Nphi]
        logT_phi = F.log_softmax(kappa * torch.cos(diff), dim=-1)            # normalise over next phase
        logT_v = F.log_softmax(self.vel_T, dim=-1)                           # [Nv', Nv]
        logT_m = F.log_softmax(self.meter_T, dim=-1)                         # [R', R]
        # assemble [Nphi',Nv',R', Nphi,Nv,R] then flatten to [S,S]
        T = (logT_phi[:, :, None, :, None, None]
             + logT_v[None, :, None, None, :, None]
             + logT_m[None, None, :, None, None, :])
        return T.reshape(self.S, self.S)                                     # [S, S]

    # ---- initial state log-prob  logpi[s] --------------------------------------
    def log_init(self):
        lp_phi = torch.full((self.Nphi,), -math.log(self.Nphi), device=self.v_c.device)  # uniform phase
        lp_v = F.log_softmax(self.init_v, dim=-1)
        lp_m = F.log_softmax(self.init_m, dim=-1)
        return (lp_phi[:, None, None] + lp_v[None, :, None] + lp_m[None, None, :]).reshape(self.S)

    # ---- per-frame emission log-prob  logE[B, S] -------------------------------
    def log_emission(self, x_k, b_k=None):
        """x_k:[B,x_dim], b_k:[B] or None (exclude labels -> predictive, audio-only)."""
        B = x_k.size(0)
        # audio: Gaussian  logp(x_k | i_phi, m)  -> [B, Nphi, R]
        sig = F.softplus(self.logsig_x) + 1e-3
        d = (x_k[:, None, None, :] - self.mu_x[None]) / sig                   # [B,Nphi,R,x_dim]
        logp_x = (-0.5 * (d ** 2) - torch.log(sig) - 0.5 * math.log(2 * PI)).sum(-1)  # [B,Nphi,R]
        logE = logp_x
        if b_k is not None:                                                  # labels: categorical
            logp_b = F.log_softmax(self.emit_b, dim=-1)                       # [Nphi,R,3]
            sel = logp_b.permute(2, 0, 1)[b_k]                               # [B,Nphi,R]
            logE = logE + sel
        # broadcast over velocity (emission is independent of i_v), flatten to [B,S]
        return logE[:, :, None, :].expand(B, self.Nphi, self.Nv, self.R).reshape(B, self.S)

    # ---- EXACT forward algorithm: returns log p(observations) per sequence -----
    def forward_ll(self, x, b=None):
        """x:[B,T,x_dim], b:[B,T] or None. Returns loglik [B] (exact)."""
        B, T = x.size(0), x.size(1)
        logT = self.log_transition()                                         # [S,S]
        a = self.log_init()[None] + self.log_emission(x[:, 0], None if b is None else b[:, 0])
        for k in range(1, T):
            # a_k(s) = logE_k(s) + logsumexp_{s'} [ a_{k-1}(s') + logT(s', s) ]
            a = self.log_emission(x[:, k], None if b is None else b[:, k]) \
                + torch.logsumexp(a[:, :, None] + logT[None], dim=1)
        return torch.logsumexp(a, dim=1)                                     # [B]

    # ---- forward-backward posterior marginals gamma_k(s) -----------------------
    @torch.no_grad()
    def posterior_marginals(self, x, b=None):
        B, T, S = x.size(0), x.size(1), self.S
        logT = self.log_transition()
        logE = torch.stack([self.log_emission(x[:, k], None if b is None else b[:, k])
                            for k in range(T)], 1)                            # [B,T,S]
        a = torch.empty(B, T, S, device=x.device)
        a[:, 0] = self.log_init()[None] + logE[:, 0]
        for k in range(1, T):
            a[:, k] = logE[:, k] + torch.logsumexp(a[:, k-1][:, :, None] + logT[None], dim=1)
        bta = torch.zeros(B, T, S, device=x.device)
        for k in range(T - 2, -1, -1):
            bta[:, k] = torch.logsumexp(logT[None] + (logE[:, k+1] + bta[:, k+1])[:, None, :], dim=2)
        g = a + bta
        return g - torch.logsumexp(g, dim=2, keepdim=True)                   # log-marginals [B,T,S]

    # ---- Viterbi MAP latent path ----------------------------------------------
    @torch.no_grad()
    def viterbi(self, x, b=None):
        B, T, S = x.size(0), x.size(1), self.S
        logT = self.log_transition()
        d = self.log_init()[None] + self.log_emission(x[:, 0], None if b is None else b[:, 0])
        bp = torch.empty(B, T, S, dtype=torch.long, device=x.device)
        for k in range(1, T):
            m = d[:, :, None] + logT[None]                                   # [B,S',S]
            d_best, bp[:, k] = m.max(dim=1)
            d = d_best + self.log_emission(x[:, k], None if b is None else b[:, k])
        s = torch.empty(B, T, dtype=torch.long, device=x.device)
        s[:, -1] = d.argmax(1)
        for k in range(T - 2, -1, -1):
            s[:, k] = bp[torch.arange(B), k + 1, s[:, k + 1]]
        iphi = s // (self.Nv * self.R); iv = (s // self.R) % self.Nv; m = s % self.R
        return iphi, iv, m

    # ---- predict labels from AUDIO ONLY (x), score vs ground truth -------------
    @torch.no_grad()
    def predict_labels(self, x):
        g = self.posterior_marginals(x, b=None).exp()                        # p(s_k | x)  [B,T,S]
        g = g.view(x.size(0), x.size(1), self.Nphi, self.Nv, self.R).sum(3)  # marginalise velocity -> [B,T,Nphi,R]
        pb = F.softmax(self.emit_b, dim=-1)                                  # [Nphi,R,3]
        p_label = torch.einsum('btpr,prc->btc', g, pb)                       # [B,T,3]
        return p_label.argmax(-1), p_label


def train(steps=400, lr=0.05, log_every=20, seed=0):
    torch.manual_seed(seed)
    x, b = synth(); x, b = x.to(DEV), b.to(DEV)          # x = log-mel-spectrogram [B,T,n_mels]
    model = DiscreteDBN(x_dim=x.size(-1)).to(DEV)        # Gaussian obs over the mel features
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"device: {DEV} | x {tuple(x.shape)} b {tuple(b.shape)} | states S={model.S} "
          f"(Nphi={model.Nphi} x Nv={model.Nv} x R={model.R})")
    first = None
    for t in range(1, steps + 1):
        ll = model.forward_ll(x, b)                      # EXACT log p(x,b) per sequence
        loss = -ll.mean()                                # exact NLL -- no ELBO, no sampling
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if t == 1: first = loss.item()
        if t == 1 or t % log_every == 0:
            # Sec-9-style health line. The cheap metrics (NLL, kappa) print every log step; the
            # heavier posterior diagnostics (audio-only predictive accuracy, posterior state-
            # occupancy entropy = collapse watch) are evaluated only at a few checkpoints.
            heavy = t in (1, steps // 2, steps)
            acc, occ_H = float("nan"), float("nan")
            if heavy:
                with torch.no_grad():
                    acc = (model.predict_labels(x)[0] == b).float().mean().item()
                    g = model.posterior_marginals(x, b).exp().mean((0, 1))   # mean state occupancy
                    occ_H = -(g * (g + 1e-12).log()).sum().item()            # entropy over S (nats)
            print(f"step {t:4d} | exact NLL = {loss.item():8.4f} | kappa = "
                  f"{F.softplus(model.log_kappa).item():5.2f}" +
                  (f" | audio-only label acc = {acc:5.3f} | occupancy H = {occ_H:5.2f}/{math.log(model.S):.2f}"
                   if heavy else ""))
    print(f"[done] NLL {first:.3f} -> {loss.item():.3f}  (exact MLE; must decrease)")
    torch.save({"state": model.state_dict()}, "vae_dbn_discrete.pt")
    return model, x, b


def decode(seed=0):
    torch.manual_seed(seed)
    x, b = synth(); x, b = x.to(DEV), b.to(DEV)
    model = DiscreteDBN(x_dim=x.size(-1)).to(DEV)
    try:
        model.load_state_dict(torch.load("vae_dbn_discrete.pt", map_location=DEV)["state"])
        print("[loaded vae_dbn_discrete.pt]")
    except FileNotFoundError:
        print("[no checkpoint -- training first]"); model, x, b = train()
    iphi, iv, m = model.viterbi(x, b)                    # MAP latent path (uses labels too)
    b_hat, _ = model.predict_labels(x)                   # labels from audio only
    acc = (b_hat == b).float().mean().item()
    # meter accuracy: the Viterbi meter should be ~constant per sequence and match nothing we
    # stored, but its entropy across frames tells us whether a single meter was inferred.
    meter_mode = m.mode(dim=1).values
    print(f"Viterbi path: phase/velocity/meter decoded for {x.size(0)} sequences x {x.size(1)} frames")
    print(f"audio-only label accuracy vs ground truth = {acc:5.3f}")
    print(f"per-sequence Viterbi meter (mode over frames): {meter_mode.tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "decode"], default="train")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    if args.mode == "train":
        train(steps=args.steps)
    else:
        decode()
