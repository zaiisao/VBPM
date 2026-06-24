"""Builder for ELBO_for_DBN.ipynb.

Assembles a faithful, from-scratch, human-readable Jupyter notebook that implements the
paper "ELBO for DBN" (Jaehoon Ahn) EXACTLY as written: the three latents (Section 2), the
generation model / prior (Section 3), the ELBO derivation (Section 4), the concrete
distributions + closed-form KLs (Section 5), Algorithm 1 (SGVB training rollout) and
Algorithm 2 (von Mises Best-Fisher sampler + implicit reparameterisation).

Nothing here imports the existing CHART code -- it is a clean reference implementation so
the correspondence to the paper can be checked line-by-line. Run:

    python notebooks/build_elbo_notebook.py        # writes notebooks/ELBO_for_DBN.ipynb
"""
import json
from pathlib import Path

# We build the .ipynb JSON by hand with the standard library so there are NO extra
# dependencies (no nbformat needed). The notebook schema is just a dict of cells.
cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ============================================================================
md(r"""
# A Faithful, Readable Implementation of *ELBO for DBN*

This notebook is a **from-scratch reference implementation** of the variational bar-pointer
model described in `ELBO_for_DBN.pdf`. It deliberately does **not** import any of our
existing training code: every piece is written out plainly so it can be checked against the
paper section by section. The goal is to make the correspondence with the paper obvious.

We implement, in order:

| Paper | Notebook |
|---|---|
| §2 — latent state $z_t=[m_t,\phi_t,\dot\phi_t]$ | the three sampled latents in the rollout |
| §3 / §5 — generation model (prior $p_\psi$) | `prior` heads + deterministic means |
| §5.1–5.3 — posteriors $q_\phi$ & closed-form KLs | `posterior` heads + `kl_*` functions |
| §5.4 — Bernoulli decoder $p_\theta(b_t\mid z_t,h)$ | `decode` |
| §4 — ELBO | the loss assembled in `run_algorithm_1` |
| Algorithm 1 — SGVB rollout | `run_algorithm_1` |
| Algorithm 2 — von Mises sampler | `best_fisher_rejection` + `VonMisesSample` |

Crucially, the objective is the **strict ELBO**: $\mathcal L = \sum_t \text{BCE} + \sum_t \text{KL}$
with $\beta=1$ and **no** free-bits, no latent supervision, no extra regularisers. This is
exactly the paper, so it doubles as the reference the rest of the project should be measured
against.
""")

code(r"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

TWO_PI = 2.0 * math.pi
device = torch.device("cpu")   # the notebook is tiny; CPU keeps it portable
print("torch", torch.__version__, "| device", device)
""")

# ============================================================================
md(r"""
## 1. Math helpers — Bessel functions and the von Mises density

The von Mises KL (§5.2) needs $\log I_0(\kappa)$ and $A(\kappa)\triangleq I_1(\kappa)/I_0(\kappa)$,
the mean resultant length. We use the *exponentially-scaled* Bessel functions
`i0e(x)=e^{-x}I_0(x)` and `i1e(x)=e^{-x}I_1(x)` for numerical stability, so

$$\log I_0(\kappa) = \log(\text{i0e}(\kappa)) + \kappa,\qquad A(\kappa)=\frac{\text{i1e}(\kappa)}{\text{i0e}(\kappa)}.$$

The zero-mean von Mises density (used by the implicit-reparam backward, §5.2) is
$p(z\mid 0,\kappa)=\dfrac{e^{\kappa\cos z}}{2\pi I_0(\kappa)}=\dfrac{e^{\kappa(\cos z-1)}}{2\pi\,\text{i0e}(\kappa)}.$
""")

code(r"""
def log_i0(kappa):
    # log I0(kappa), numerically stable via the exponentially-scaled Bessel i0e
    return torch.log(torch.special.i0e(kappa)) + kappa

def A_kappa(kappa):
    # mean resultant length A(kappa) = I1(kappa) / I0(kappa)
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)

def von_mises_pdf(z, kappa):
    # density of a ZERO-MEAN von Mises at angle z
    return torch.exp(kappa * (torch.cos(z) - 1.0)) / (TWO_PI * torch.special.i0e(kappa))

def von_mises_cdf(z, kappa, n_steps=100):
    # F(z | 0, kappa) = integral_{-pi}^{z} p(t | 0, kappa) dt, via the trapezoid rule.
    # Works for z, kappa of the same (arbitrary) shape; integrates along a new last axis.
    lower = -math.pi
    frac = torch.linspace(0.0, 1.0, n_steps, device=z.device, dtype=z.dtype)  # [n]
    z_e = z.unsqueeze(-1)
    k_e = kappa.unsqueeze(-1)
    t = lower + frac * (z_e - lower)                                # integration grid
    pdf = torch.exp(k_e * (torch.cos(t) - 1.0)) / (TWO_PI * torch.special.i0e(k_e))
    weights = torch.ones_like(pdf)
    weights[..., 0] = 0.5
    weights[..., -1] = 0.5
    step = (z_e - lower) / (n_steps - 1)
    return (pdf * weights).sum(-1) * step.squeeze(-1)

# quick sanity check: A(kappa) increases toward 1 as kappa grows
for k in [0.1, 1.0, 5.0, 20.0]:
    kk = torch.tensor(k)
    print(f"kappa={k:5.1f}  log I0={log_i0(kk):.4f}  A(kappa)={A_kappa(kk):.4f}")
""")

# ============================================================================
md(r"""
## 2. Algorithm 2 — the von Mises sampler (Best–Fisher) with implicit reparameterisation

The phase latent is drawn from a von Mises distribution. Because the standard rejection
sampler is not differentiable, the paper (Algorithm 2) uses **implicit reparameterisation**
gradients (Figurnov et al. 2018):

- **Forward** (lines 1–16): draw $z\sim\mathrm{vM}(0,\kappa)$ with the Best–Fisher rejection
  sampler, then return $\hat\phi=\mu+z$.
- **Backward** (lines 18–24): $\dfrac{\partial\hat\phi}{\partial\mu}=1$ and
  $\dfrac{\partial\hat\phi}{\partial\kappa}=-\dfrac{\partial F(z\mid\kappa)/\partial\kappa}{p(z\mid 0,\kappa)}$,
  where $\partial F/\partial\kappa$ is obtained by auto-differentiating the CDF.

We implement the rejection loop exactly as Algorithm 2, vectorised so all time steps in the
batch are drawn together (entries re-draw until accepted).
""")

code(r"""
def best_fisher_rejection(kappa, max_iter=100):
    # Algorithm 2, forward pass (lines 1-16). Samples z ~ vM(0, kappa) elementwise.
    shape = kappa.shape
    k = kappa.reshape(-1).clamp(min=1e-3)

    tau = 1.0 + torch.sqrt(1.0 + 4.0 * k * k)        # line 1
    rho = (tau - torch.sqrt(2.0 * tau)) / (2.0 * k)  # line 2
    r = (1.0 + rho * rho) / (2.0 * rho)              # line 3

    z = torch.zeros_like(k)
    accepted = torch.zeros_like(k, dtype=torch.bool)
    for _ in range(max_iter):
        u1 = torch.rand_like(k)
        u2 = torch.rand_like(k)
        u3 = torch.rand_like(k)
        c = torch.cos(math.pi * u1)                  # line 6
        f = (1.0 + r * c) / (r + c)                  # line 7
        accept = (k * (r - f) + torch.log(f) - torch.log(r)) >= torch.log(u2)   # line 9
        sign = torch.where(u3 > 0.5, 1.0, -1.0)      # lines 11-15
        angle = sign * torch.acos(torch.clamp(f, -1.0, 1.0))
        newly = accept & (~accepted)
        z = torch.where(newly, angle, z)
        accepted = accepted | accept
        if bool(accepted.all()):
            break
    return z.reshape(shape)


class VonMisesSample(torch.autograd.Function):
    # Algorithm 2 wrapped as an autograd Function: forward = rejection sampler,
    # backward = implicit reparameterisation gradients.
    @staticmethod
    def forward(ctx, mu, kappa):
        z = best_fisher_rejection(kappa)             # z ~ vM(0, kappa)
        phi = mu + z                                 # line 16
        ctx.save_for_backward(z, kappa)
        return phi

    @staticmethod
    def backward(ctx, grad_phi):
        z, kappa = ctx.saved_tensors
        # dF(z|kappa)/dkappa via autograd on the CDF (line 20)
        with torch.enable_grad():
            k = kappa.detach().clone().requires_grad_(True)
            cdf = von_mises_cdf(z.detach(), k)
            (dF_dk,) = torch.autograd.grad(cdf.sum(), k)
        p = von_mises_pdf(z, kappa)                  # line 19
        dphi_dkappa = -dF_dk / (p + 1e-12)           # line 21
        grad_mu = grad_phi * 1.0                     # line 22 (dphi/dmu = 1)
        grad_kappa = grad_phi * dphi_dkappa          # line 23
        return grad_mu, grad_kappa


def sample_von_mises(mu, kappa):
    return VonMisesSample.apply(mu, kappa)

# sanity check: empirical mean angle ~ mu, and gradient flows through kappa
mu_test = torch.tensor(1.0)
kap_test = torch.tensor(4.0, requires_grad=True)
samples = torch.stack([sample_von_mises(mu_test, kap_test) for _ in range(2000)])
print(f"target mu=1.000  empirical circular mean={torch.atan2(torch.sin(samples).mean(), torch.cos(samples).mean()):.3f}")
(sample_von_mises(mu_test, kap_test)).backward()
print("d(sample)/d(kappa) is finite:", torch.isfinite(kap_test.grad).item())
""")

# ============================================================================
md(r"""
## 3. The closed-form KL divergences (§5.1–5.3)

These are copied verbatim from the paper:

- **Meter (Categorical, §5.1):** $\;D_{KL}=\sum_k \pi^q_k\log\dfrac{\pi^q_k}{\pi^p_k}$
- **Phase (von Mises, §5.2):** $\;D_{KL}=\log\dfrac{I_0(\kappa^p)}{I_0(\kappa^q)}+A(\kappa^q)\big[\kappa^q-\kappa^p\cos(\mu^q-\mu^p)\big]$
- **Tempo (Log-Normal, §5.3):** $\;D_{KL}=\log\dfrac{\sigma^p}{\sigma^q}+\dfrac{\sigma^{q2}+(\mu^q-\mu^p)^2}{2\sigma^{p2}}-\dfrac12$
""")

code(r"""
def kl_categorical(log_q, log_p):
    # KL( Cat(q) || Cat(p) ) ; inputs are LOG-probabilities, summed over the K classes
    q = log_q.exp()
    return (q * (log_q - log_p)).sum(-1)

def kl_von_mises(mu_q, kappa_q, mu_p, kappa_p):
    return (log_i0(kappa_p) - log_i0(kappa_q)
            + A_kappa(kappa_q) * (kappa_q - kappa_p * torch.cos(mu_q - mu_p)))

def kl_log_normal(mu_q, sigma_q, mu_p, sigma_p):
    # Log-Normal KL reduces to the Gaussian KL in log-space (§5.3)
    return (torch.log(sigma_p / sigma_q)
            + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2) - 0.5)

# sanity: KL(p || p) == 0 for each
print("KL meter (q==p):", float(kl_categorical(torch.log_softmax(torch.tensor([1.,0.,2.,0.]),0),
                                                torch.log_softmax(torch.tensor([1.,0.,2.,0.]),0))))
print("KL phase (q==p):", float(kl_von_mises(torch.tensor(1.0), torch.tensor(3.0),
                                             torch.tensor(1.0), torch.tensor(3.0))))
print("KL tempo (q==p):", float(kl_log_normal(torch.tensor(0.5), torch.tensor(0.2),
                                              torch.tensor(0.5), torch.tensor(0.2))))
""")

# ============================================================================
md(r"""
## 4. The model: priors $p_\psi$, posteriors $q_\phi$, decoder $p_\theta$ (§3, §5)

Each sub-network maps directly onto a function named in the paper:

- `encode_posterior` is the inference network's shared read of $b_{1:T}$ and $h_{1:T}$
  (the $f_\phi$ context). The per-step posterior head also takes the **sampled** $\hat z_{t-1}$.
- `encode_prior` is the generative read of $h_{1:T}$ (the $f_\psi$ context).
- `prior_init_head` is $f^{\text{init}}_\psi(h_{1:T})$ for $t=1$.
- `prior_phase_kappa` is $\kappa^p_{\phi,t}=f^\phi_\psi(h)$; the phase prior **mean** is the
  deterministic bar-pointer advance $\mu^p_{\phi,t}=\phi_{t-1}+\dot\phi_{t-1}$ (no network).
- `prior_tempo_sigma` is $\sigma^p_{\dot\phi,t}=f^{\dot\phi}_\psi(h)$; the tempo prior **mean**
  is the random walk $\mu^p_{\dot\phi,t}=\log\dot\phi_{t-1}$ (no network).
- `meter_prior_logp` is the $K\times K$ transition matrix $f^m_\psi(m_{t-1},\phi_t,\phi_{t-1},h)$.
- `decode` is the Bernoulli decoder $\hat b_t=\sigma(\mathrm{NN}_\theta(z_t,h_{1:T}))$ (§5.4).

The latent feature vector handed to the posterior head and decoder is
$[\cos\phi,\ \sin\phi,\ \log\dot\phi,\ \text{onehot}(m)]$.
""")

code(r"""
def gumbel_softmax(logits, temperature):
    # Gumbel-Softmax relaxation of a Categorical (§5.1)
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    return F.softmax((logits + g) / temperature, dim=-1)


class BarPointerVAE(nn.Module):
    def __init__(self, h_dim, hidden=32, num_meters=4):
        super().__init__()
        self.K = num_meters
        self.hidden = hidden
        z_feat_dim = 3 + num_meters     # cos phi, sin phi, log tempo, onehot(meter)
        param_dim = num_meters + 2 + 1 + 1 + 1   # meter logits | phase(u,v) | phase log-kappa | tempo mu | tempo log-sigma

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
        self.prior_phase_kappa = nn.Linear(hidden, 1)     # f^phi_psi(h)
        self.prior_tempo_sigma = nn.Linear(hidden, 1)     # f^phidot_psi(h)
        self.meter_prior = nn.Sequential(                 # f^m_psi(m_{t-1}, phi_t, phi_{t-1}, h)
            nn.Linear(num_meters + 4 + hidden, hidden), nn.Tanh(), nn.Linear(hidden, num_meters * num_meters))

        # decoder NN_theta(z_t, h)
        self.decoder = nn.Sequential(
            nn.Linear(z_feat_dim + hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    # ---- shared sequence encoders ----
    def encode_posterior(self, h, b):
        x = torch.cat([h, b.unsqueeze(-1)], dim=-1).unsqueeze(0)   # [1, T, h_dim+1]
        out, _ = self.post_gru(x)
        return torch.tanh(self.post_ctx(out[0]))                   # [T, hidden]

    def encode_prior(self, h):
        out, _ = self.prior_gru(h.unsqueeze(0))
        return torch.tanh(self.prior_ctx(out[0]))                  # [T, hidden]

    # ---- unpack a raw parameter vector into named distribution params ----
    def unpack(self, vec):
        K = self.K
        meter_logits = vec[:K]
        u, v = vec[K], vec[K + 1]
        phase_mu = torch.atan2(v, u) % TWO_PI
        phase_kappa = F.softplus(vec[K + 2]) + 0.01
        tempo_mu = vec[K + 3]
        tempo_sigma = F.softplus(vec[K + 4]) + 1e-3
        return meter_logits, phase_mu, phase_kappa, tempo_mu, tempo_sigma

    def z_features(self, meter_soft, phi, log_tempo):
        return torch.cat([torch.cos(phi).reshape(1), torch.sin(phi).reshape(1),
                          log_tempo.reshape(1), meter_soft.reshape(self.K)])

    # ---- prior meter transition (returns log prior over m_t) ----
    def meter_prior_logp(self, meter_prev, phi_t, phi_prev, prior_ctx_t):
        feats = torch.cat([meter_prev.reshape(self.K),
                           torch.cos(phi_t).reshape(1), torch.sin(phi_t).reshape(1),
                           torch.cos(phi_prev).reshape(1), torch.sin(phi_prev).reshape(1),
                           prior_ctx_t])
        Pi = F.softmax(self.meter_prior(feats).reshape(self.K, self.K), dim=1)  # rows: from-meter
        pi_p = meter_prev.reshape(1, self.K) @ Pi          # mix by the (soft) previous meter
        return torch.log(pi_p.reshape(self.K) + 1e-9)

    def decode(self, z_feat, prior_ctx_t):
        return self.decoder(torch.cat([z_feat, prior_ctx_t])).reshape(())   # scalar beat logit

print("model defined")
""")

# ============================================================================
md(r"""
## 5. The ELBO and Algorithm 1 (the training rollout)

The ELBO derivation in §4 collapses to a sum of a reconstruction term and per-step KL terms
(one per latent, for $t=1$ and for the transitions $t\ge2$):

$$\mathcal L=\underbrace{\sum_{t}\!-\big[b_t\log\hat b_t+(1-b_t)\log(1-\hat b_t)\big]}_{\text{reconstruction (BCE)}}
\;+\;\sum_{t}\Big[\underbrace{D_{KL}^{m}}_{\text{meter}}+\underbrace{D_{KL}^{\phi}}_{\text{phase}}+\underbrace{D_{KL}^{\dot\phi}}_{\text{tempo}}\Big].$$

`run_algorithm_1` implements Algorithm 1 step for step. The key faithful details:
- the posterior at $t$ reads the **sampled** $\hat z_{t-1}$ (line 15);
- the prior phase mean is $\phi_{t-1}+\dot\phi_{t-1}$ and the prior tempo mean is $\log\dot\phi_{t-1}$,
  both from the **sampled** previous state (line 16);
- the meter prior is evaluated **after** sampling $\hat\phi_t$ (line 21), because a bar boundary
  depends on the current phase.

It returns the loss plus a dictionary of intermediate values so we can print/plot them.
""")

code(r"""
def run_algorithm_1(model, h, b, temperature=0.5, verbose=False):
    T = h.shape[0]
    post_ctx = model.encode_posterior(h, b)    # f_phi context, reads (b, h)
    prior_ctx = model.encode_prior(h)          # f_psi context, reads h

    L_kl = h.new_zeros(())
    kl_running = {"meter": 0.0, "phase": 0.0, "tempo": 0.0}
    z_feats = []                # latent features per step, for the decoder
    post_phase_mu = []          # posterior phase mean trajectory, for plotting

    # ---------- t = 1 : initial state (Algorithm 1, lines 7-13) ----------
    q_vec = model.post_head(torch.cat([post_ctx[0], model.z0]))
    q_m, q_phi_mu, q_phi_k, q_tau_mu, q_tau_s = model.unpack(q_vec)
    p_vec = model.prior_init_head(prior_ctx.mean(0))
    p_m, p_phi_mu, p_phi_k, p_tau_mu, p_tau_s = model.unpack(p_vec)

    meter = gumbel_softmax(q_m, temperature)                 # line 9
    phi = sample_von_mises(q_phi_mu, q_phi_k) % TWO_PI        # line 10
    log_tempo = q_tau_mu + q_tau_s * torch.randn(())         # line 11

    kld_m = kl_categorical(torch.log_softmax(q_m, 0), torch.log_softmax(p_m, 0))
    kld_p = kl_von_mises(q_phi_mu, q_phi_k, p_phi_mu, p_phi_k)
    kld_t = kl_log_normal(q_tau_mu, q_tau_s, p_tau_mu, p_tau_s)
    L_kl = L_kl + kld_m + kld_p + kld_t
    kl_running["meter"] += float(kld_m); kl_running["phase"] += float(kld_p); kl_running["tempo"] += float(kld_t)

    z_feats.append(model.z_features(meter, phi, log_tempo))
    post_phase_mu.append(float(q_phi_mu))
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo

    if verbose:
        print("t=1  (initial state)")
        print(f"   posterior: phase_mu={float(q_phi_mu):.3f} kappa={float(q_phi_k):.2f} "
              f"tempo_mu={float(q_tau_mu):.3f} sigma={float(q_tau_s):.3f}")
        print(f"   sampled  : meter={meter.detach().numpy().round(2)} phi={float(phi):.3f} "
              f"log_tempo={float(log_tempo):.3f}")
        print(f"   KL       : meter={float(kld_m):.4f} phase={float(kld_p):.4f} tempo={float(kld_t):.4f}")

    # ---------- t = 2..T : transitions (Algorithm 1, lines 14-23) ----------
    for t in range(1, T):
        # posterior reads sampled z_{t-1} (line 15)
        z_prev_feat = model.z_features(meter_prev, phi_prev, log_tempo_prev)
        q_vec = model.post_head(torch.cat([post_ctx[t], z_prev_feat]))
        q_m, q_phi_mu, q_phi_k, q_tau_mu, q_tau_s = model.unpack(q_vec)

        # prior MEANS are the deterministic bar-pointer dynamics on sampled z_{t-1} (line 16)
        tempo_prev = torch.exp(log_tempo_prev)
        p_phi_mu = (phi_prev + tempo_prev) % TWO_PI         # mu^p_phi = phi_{t-1} + phidot_{t-1}
        p_phi_k = F.softplus(model.prior_phase_kappa(prior_ctx[t]).reshape(())) + 0.01
        p_tau_mu = log_tempo_prev                           # mu^p_tempo = log phidot_{t-1}
        p_tau_s = F.softplus(model.prior_tempo_sigma(prior_ctx[t]).reshape(())) + 1e-3

        # sample current latents from the posterior (lines 17-19)
        meter = gumbel_softmax(q_m, temperature)
        phi = sample_von_mises(q_phi_mu, q_phi_k) % TWO_PI
        log_tempo = q_tau_mu + q_tau_s * torch.randn(())

        # meter prior uses the SAMPLED phi_t (line 21)
        log_pi_p = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[t])

        kld_m = kl_categorical(torch.log_softmax(q_m, 0), log_pi_p)
        kld_p = kl_von_mises(q_phi_mu, q_phi_k, p_phi_mu, p_phi_k)
        kld_t = kl_log_normal(q_tau_mu, q_tau_s, p_tau_mu, p_tau_s)
        L_kl = L_kl + kld_m + kld_p + kld_t
        kl_running["meter"] += float(kld_m); kl_running["phase"] += float(kld_p); kl_running["tempo"] += float(kld_t)

        z_feats.append(model.z_features(meter, phi, log_tempo))
        post_phase_mu.append(float(q_phi_mu))
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo

        if verbose and t == T // 2:
            print(f"t={t}  (a transition step)")
            print(f"   prior   : phase_mu={float(p_phi_mu):.3f} (=phi_prev+tempo_prev) kappa={float(p_phi_k):.2f}")
            print(f"   posterior: phase_mu={float(q_phi_mu):.3f} kappa={float(q_phi_k):.2f}")
            print(f"   KL       : meter={float(kld_m):.4f} phase={float(kld_p):.4f} tempo={float(kld_t):.4f}")

    # ---------- decode (lines 24-27) ----------
    beat_logits = torch.stack([model.decode(z_feats[t], prior_ctx[t]) for t in range(T)])
    recon = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="sum")
    loss = recon + L_kl                                     # line 28 (strict ELBO, beta=1)

    info = {
        "loss": float(loss), "recon": float(recon), "kl": float(L_kl),
        "kl_meter": kl_running["meter"], "kl_phase": kl_running["phase"], "kl_tempo": kl_running["tempo"],
        "beat_prob": torch.sigmoid(beat_logits).detach().numpy(),
        "post_phase_mu": np.array(post_phase_mu),
    }
    return loss, info

print("run_algorithm_1 defined")
""")

# ============================================================================
md(r"""
## 6. A toy "song" to make the values concrete

To exercise the full machinery we synthesise one short periodic sequence: beats every
`frames_per_beat` frames, and an observation $h$ that is a noisy Gaussian bump at each beat
(a stand-in for a frozen frontend's onset feature). This lets us watch the latent learn the
phase and the decoder reconstruct the beats.
""")

code(r"""
def make_toy_song(T=96, frames_per_beat=12, h_dim=8, noise=0.4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    beat_frames = torch.arange(0, T, frames_per_beat)
    b = torch.zeros(T)
    b[beat_frames] = 1.0
    idx = torch.arange(T).float()
    bump = torch.zeros(T)
    for f in beat_frames:
        bump = bump + torch.exp(-0.5 * ((idx - f) / 1.5) ** 2)
    h = bump.unsqueeze(-1).repeat(1, h_dim) + noise * torch.randn(T, h_dim, generator=gen)
    return h, b, beat_frames

h, b, beat_frames = make_toy_song()
print("h shape", tuple(h.shape), "| #beats", int(b.sum().item()), "| frames/beat = 12 -> ~0.52 rad/frame")

fig, ax = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
ax[0].imshow(h.T, aspect="auto", origin="lower", cmap="magma"); ax[0].set_ylabel("h channels")
ax[0].set_title("Observation h (noisy onset bumps)")
ax[1].plot(b.numpy(), "g"); ax[1].set_ylabel("beat"); ax[1].set_xlabel("frame")
ax[1].set_title("Ground-truth beats b")
plt.tight_layout(); plt.show()
""")

# ============================================================================
md(r"""
## 7. One forward pass — printing every intermediate value

Before any training, run Algorithm 1 once with `verbose=True`. This prints the posterior and
prior parameters, the sampled latents, and the per-step KL — the exact quantities the paper
defines — followed by the full ELBO breakdown.
""")

code(r"""
model = BarPointerVAE(h_dim=h.shape[-1], hidden=32, num_meters=4).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {n_params}\n")

loss, info = run_algorithm_1(model, h, b, temperature=0.7, verbose=True)
print("\nELBO breakdown (untrained):")
print(f"   reconstruction (BCE) = {info['recon']:.3f}")
print(f"   KL meter             = {info['kl_meter']:.3f}")
print(f"   KL phase             = {info['kl_phase']:.3f}")
print(f"   KL tempo             = {info['kl_tempo']:.3f}")
print(f"   TOTAL  L = -ELBO     = {info['loss']:.3f}")
""")

# ============================================================================
md(r"""
## 8. Training (Algorithm 1, lines 3–31)

A single `AdamW` optimiser updates **all** parameters $\theta,\phi,\psi$ jointly (line 30).
The objective is the **strict ELBO** — $\beta=1$, no free-bits, no auxiliary supervision.
We overfit the one toy sequence and watch $\mathcal L$, the reconstruction, and the KL.
""")

code(r"""
model = BarPointerVAE(h_dim=h.shape[-1], hidden=32, num_meters=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

n_steps = 200
history = {"loss": [], "recon": [], "kl": []}
for step in range(1, n_steps + 1):
    temperature = 1.0 + (0.3 - 1.0) * (step / n_steps)   # anneal Gumbel temperature 1.0 -> 0.3
    optimizer.zero_grad()
    loss, info = run_algorithm_1(model, h, b, temperature=temperature)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    history["loss"].append(info["loss"]); history["recon"].append(info["recon"]); history["kl"].append(info["kl"])
    if step % 20 == 0 or step == 1:
        print(f"step {step:3d} | L={info['loss']:7.2f} | recon={info['recon']:7.2f} | "
              f"KL={info['kl']:6.2f} (m={info['kl_meter']:.2f} phi={info['kl_phase']:.2f} tau={info['kl_tempo']:.2f})")
""")

# ============================================================================
md(r"""
## 9. Results — did the latent learn the structure?

We plot (1) the loss curves, (2) the learned posterior phase mean $\mu^q_{\phi,t}$ against the
ground-truth beats — a correct fit looks like a sawtooth that wraps $2\pi\!\to\!0$ at each beat —
and (3) the decoder's beat probability against the ground truth. Finally we print the
converged ELBO breakdown.
""")

code(r"""
loss, info = run_algorithm_1(model, h, b, temperature=0.3)

fig, ax = plt.subplots(3, 1, figsize=(11, 8))
ax[0].plot(history["loss"], label="L = -ELBO")
ax[0].plot(history["recon"], label="reconstruction")
ax[0].plot(history["kl"], label="KL")
ax[0].set_title("Training curves"); ax[0].set_xlabel("step"); ax[0].legend()

ax[1].plot(info["post_phase_mu"], "purple", label=r"posterior $\mu^q_\phi$ (mod $2\pi$)")
for f in beat_frames.numpy():
    ax[1].axvline(f, color="g", alpha=0.4)
ax[1].set_title("Learned phase vs GT beats (green) — should wrap once per beat"); ax[1].legend()

ax[2].plot(info["beat_prob"], "b", label="decoder P(beat)")
ax[2].plot(b.numpy(), "g", alpha=0.5, label="GT beat")
ax[2].set_title("Reconstruction"); ax[2].set_xlabel("frame"); ax[2].legend()
plt.tight_layout(); plt.show()

print("Converged ELBO breakdown:")
print(f"   reconstruction (BCE) = {info['recon']:.3f}")
print(f"   KL meter / phase / tempo = {info['kl_meter']:.3f} / {info['kl_phase']:.3f} / {info['kl_tempo']:.3f}")
print(f"   TOTAL L = {info['loss']:.3f}")
""")

# ============================================================================
md(r"""
## 10. Does it actually work? Deploy-path evaluation + a real DBN baseline

Sections 7–9 only exercised the **teacher-forced** path: the latent was inferred by $q_\phi$
*with access to the beats $b$*, and the decoder also sees $h$. A low reconstruction there can be
achieved by the decoder reading $h$ and ignoring the latent, so it is **not** evidence the model
works. The honest tests are:

1. **Free-running the prior** (the deploy path): roll $p_\psi$ forward with **no beats**, reading
   beats off (a) the deterministic phase wraps and (b) the decoder. This is what the model does at
   test time.
2. **Comparison with a real DBN** — the discrete bar-pointer model with exact **Viterbi** decoding
   (Krebs/madmom-style), the established method this variational model is meant to succeed. On a
   clean signal the DBN is the bar to clear.

Every method is scored with a simple $\pm2$-frame F-measure against the ground-truth beats.
""")

code(r"""
# ---- a shared 1-D beat activation derived from h (what peak-pick and the DBN observe) ----
act_1d = h.mean(dim=-1)
act_1d = (act_1d - act_1d.min()) / (act_1d.max() - act_1d.min() + 1e-9)
act_1d_np = act_1d.numpy()
gt_beat_frames = beat_frames.numpy()

def beat_f_measure(pred_frames, gt_frames, tol=2):
    # self-contained tolerance-window F-measure (a +/- tol-frame match, like mir_eval's window)
    pred = sorted(int(x) for x in pred_frames)
    gt = [int(x) for x in gt_frames]
    used = set(); tp = 0
    for pb in pred:
        for i, gb in enumerate(gt):
            if i not in used and abs(pb - gb) <= tol:
                used.add(i); tp += 1; break
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

def peak_pick(activation, threshold=0.5):
    # beats = local maxima above a threshold
    frames = []
    for t in range(1, len(activation) - 1):
        if activation[t] >= threshold and activation[t] >= activation[t-1] and activation[t] >= activation[t+1]:
            frames.append(t)
    return np.array(frames)

print("shared 1-D activation ready | #GT beats =", len(gt_beat_frames))
""")

code(r"""
# ============ a real discrete bar-pointer DBN with Viterbi (Krebs/madmom-style) ============
# State = (beat-period n in frames, position p within the beat, p in 0..n-1). The pointer advances
# one position per frame; a beat occurs at p=0. At a beat boundary the tempo may change with
# probability proportional to exp(-lambda * |n'/n - 1|) (the madmom transition). Decoding is exact
# MAP via Viterbi. This is the established model the paper's variational version is meant to succeed.

def build_dbn(min_period=6, max_period=20):
    states = []
    for n in range(min_period, max_period + 1):
        for p in range(n):
            states.append((n, p))
    index = {s: i for i, s in enumerate(states)}
    periods = list(range(min_period, max_period + 1))
    return states, index, periods

def dbn_viterbi(activation, states, index, periods, lam=100.0):
    T = len(activation); S = len(states); NEG = -1e9
    log_beat = np.log(np.clip(activation, 1e-6, 1 - 1e-6))
    log_nonbeat = np.log(np.clip(1 - activation, 1e-6, 1 - 1e-6))

    # tempo-transition log-probs at a beat boundary, normalised over target periods per source
    log_tempo_trans = {}
    for n_src in periods:
        w = {n_dst: math.exp(-lam * abs(n_dst / n_src - 1.0)) for n_dst in periods}
        Z = sum(w.values())
        log_tempo_trans[n_src] = {n_dst: math.log(w[n_dst] / Z) for n_dst in periods}

    # predecessors of each state: list of (prev_state_index, log_transition_prob)
    preds = [[] for _ in range(S)]
    for (n, p) in states:
        s = index[(n, p)]
        if p == 0:                                    # beat onset: from the last position of any period
            for n_src in periods:
                preds[s].append((index[(n_src, n_src - 1)], log_tempo_trans[n_src][n]))
        else:                                         # mid-beat: deterministic +1 advance
            preds[s].append((index[(n, p - 1)], 0.0))

    delta = np.full((T, S), NEG); back = np.zeros((T, S), dtype=int)
    for s, (n, p) in enumerate(states):
        delta[0, s] = log_beat[0] if p == 0 else log_nonbeat[0]
    for t in range(1, T):
        for s, (n, p) in enumerate(states):
            obs = log_beat[t] if p == 0 else log_nonbeat[t]
            best, best_prev = NEG, 0
            for (prev, lt) in preds[s]:
                val = delta[t - 1, prev] + lt
                if val > best:
                    best, best_prev = val, prev
            delta[t, s] = best + obs; back[t, s] = best_prev
    s = int(np.argmax(delta[T - 1])); path = [s]
    for t in range(T - 1, 0, -1):
        s = back[t, s]; path.append(s)
    path = path[::-1]
    return np.array([t for t in range(T) if states[path[t]][1] == 0])

states, index, periods = build_dbn()
dbn_beats = dbn_viterbi(act_1d_np, states, index, periods, lam=100.0)
print(f"DBN: {len(states)} states | decoded {len(dbn_beats)} beats")
""")

code(r"""
# ============ the variational model's DEPLOY path: free-run the prior (NO beats) ============
@torch.no_grad()
def free_run(model, h, temperature=0.3):
    T = h.shape[0]
    prior_ctx = model.encode_prior(h)
    # t = 1: sample from the PRIOR initial state (no posterior -- there are no beats at deploy time)
    p_m, p_phi_mu, p_phi_k, p_tau_mu, p_tau_s = model.unpack(model.prior_init_head(prior_ctx.mean(0)))
    meter = gumbel_softmax(p_m, temperature)
    phi = sample_von_mises(p_phi_mu, p_phi_k) % TWO_PI
    log_tempo = p_tau_mu + p_tau_s * torch.randn(())
    z_feats = [model.z_features(meter, phi, log_tempo)]
    wrap_beats = []
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
    for t in range(1, T):
        tempo_prev = torch.exp(log_tempo_prev)
        # deterministic bar-pointer wrap detection (paper p.6): a beat when the predicted mean crosses 2*pi
        if float(phi_prev + tempo_prev) >= TWO_PI:
            wrap_beats.append(t)
        p_phi_mu = (phi_prev + tempo_prev) % TWO_PI
        p_phi_k = F.softplus(model.prior_phase_kappa(prior_ctx[t]).reshape(())) + 0.01
        p_tau_mu = log_tempo_prev
        p_tau_s = F.softplus(model.prior_tempo_sigma(prior_ctx[t]).reshape(())) + 1e-3
        phi = sample_von_mises(p_phi_mu, p_phi_k) % TWO_PI
        log_tempo = p_tau_mu + p_tau_s * torch.randn(())
        log_pi_p = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[t])
        meter = gumbel_softmax(log_pi_p, temperature)
        z_feats.append(model.z_features(meter, phi, log_tempo))
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
    beat_logits = torch.stack([model.decode(z_feats[t], prior_ctx[t]) for t in range(T)])
    return np.array(wrap_beats), torch.sigmoid(beat_logits).numpy()

freerun_wrap_beats, freerun_decoder_prob = free_run(model, h, temperature=0.3)
freerun_decoder_beats = peak_pick(freerun_decoder_prob, threshold=0.5)
print(f"free-run: {len(freerun_wrap_beats)} phase-wrap beats, {len(freerun_decoder_beats)} decoder beats")
""")

code(r"""
# ---- side-by-side comparison: every method vs the ground-truth beats ----
peak_beats = peak_pick(act_1d_np, threshold=0.5)
_, post_info = run_algorithm_1(model, h, b, temperature=0.3)             # teacher-forced (sees beats)
post_decoder_beats = peak_pick(post_info["beat_prob"], threshold=0.5)

results = {
    "peak-pick on h (discriminative)":            peak_beats,
    "DBN (Viterbi) -- the real baseline":         dbn_beats,
    "variational FREE-RUN: phase-wrap":           freerun_wrap_beats,
    "variational FREE-RUN: decoder":              freerun_decoder_beats,
    "variational posterior decoder (cheats: sees beats)": post_decoder_beats,
}

print(f"{'method':52s} {'#beats':>6} {'beat-F':>8}")
print("-" * 70)
for name, frames in results.items():
    print(f"{name:52s} {len(frames):6d} {beat_f_measure(frames, gt_beat_frames):8.3f}")

fig, ax = plt.subplots(figsize=(11, 3.2))
for f in gt_beat_frames:
    ax.axvline(f, color="green", alpha=0.3)
rows = list(results.keys())
colors = ["tab:blue", "tab:red", "tab:purple", "tab:orange", "tab:gray"]
for i, (name, frames) in enumerate(results.items()):
    ax.scatter(frames, [i] * len(frames), c=colors[i], s=22)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
ax.set_xlabel("frame   (green lines = GT beats)")
ax.set_title("Beats recovered per method vs ground truth")
plt.tight_layout(); plt.show()
""")

# ============================================================================
md(r"""
## 11. How to read the comparison

- **DBN (Viterbi)** is the discrete bar-pointer model the paper's variational version is meant to
  succeed. On a clean signal it should be near-perfect — it is the bar to clear.
- **peak-pick** is the trivial discriminative baseline.
- The two **variational FREE-RUN** rows use **no beats** at test time — they are the real measure of
  whether the generative model learned the structure (phase-wrap = the bar-pointer dynamics;
  decoder = the learned emission).
- The **posterior decoder** row is shown only for contrast: it saw the beats through $q_\phi$, so a
  high score there is the decoder-shortcut number, **not** evidence the model works.

Honest reading: if the free-run rows match the DBN, the variational model earns its place. If they
fall well short while the posterior row looks good, the model is collapsing onto the shortcut — the
same failure as the production runs, now visible side-by-side against a real DBN.
""")

# ============================================================================
md(r"""
## 12. Faithfulness checklist

Every element of the paper is present, in one readable place:

| Paper element | Where, in this notebook |
|---|---|
| Latent $z_t=[m_t,\phi_t,\dot\phi_t]$ (§2) | sampled in `run_algorithm_1` |
| Tempo prior = log-space random walk, $\mu^p=\log\dot\phi_{t-1}$ (§3, §5.3) | `p_tau_mu = log_tempo_prev` |
| Phase prior mean $=\phi_{t-1}+\dot\phi_{t-1}$, learned $\kappa$ (§5.2) | `p_phi_mu`, `prior_phase_kappa` |
| Meter prior $f^m_\psi(m_{t-1},\phi_t,\phi_{t-1},h)$ (§3, §5.1) | `meter_prior_logp` |
| Gumbel-Softmax meter sampling (§5.1) | `gumbel_softmax` |
| von Mises Best–Fisher sampler + implicit reparam (Alg. 2) | `best_fisher_rejection`, `VonMisesSample` |
| Log-Normal tempo reparam (§5.3) | `q_tau_mu + q_tau_s * randn` |
| Closed-form KLs (§5.1–5.3) | `kl_categorical`, `kl_von_mises`, `kl_log_normal` |
| Bernoulli decoder $\sigma(\mathrm{NN}_\theta(z_t,h))$ (§5.4) | `decode` |
| Posterior reads $b_{1:T}, \hat z_{t-1}, h$ (Alg. 1 line 15) | `post_head([post_ctx[t], z_prev_feat])` |
| Meter prior after sampling $\hat\phi_t$ (Alg. 1 line 21) | order inside the loop |
| $\mathcal L=\sum\text{BCE}+\sum\text{KL}$, single Adam, $\beta=1$ (Alg. 1 lines 28-30) | `loss = recon + L_kl` |

**No** free-bits, **no** latent supervision, **no** audio-correction of the prior mean,
**no** extra latents, **no** scheduled sampling. This is the strict ELBO exactly as derived —
the reference against which any deviation in the production code can be named and justified.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "ELBO_for_DBN.ipynb"
with open(out, "w") as f:
    json.dump(notebook, f, indent=1)
print("wrote", out)
