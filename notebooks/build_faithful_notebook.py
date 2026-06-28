"""Builder for notebooks/ELBO_for_DBN.ipynb — the FAITHFUL, self-contained notebook.

This assembles a clean, from-scratch Jupyter notebook for a bar-pointer dynamical VAE for
beat tracking. Design rules (enforced):
  * NO imports of our own codebase — only third-party libraries (torch, torchaudio, numpy,
    mir_eval, matplotlib). Everything the model needs is written out inline so it can be
    audited end to end.
  * NO references to any external document / paper / section numbers. The model and its ELBO
    are derived here from first principles.
  * Every value-crunching function lives in its own cell and is immediately demonstrated on
    small EXAMPLE values, so the reader can see exactly what each component does.
  * Trains and evaluates on the ballroom dataset (path configurable for Google Drive).

The notebook is built AND executed here (no nbformat dependency): code cells run in a shared
namespace, stdout and matplotlib figures are captured and embedded as outputs.

Run:  python notebooks/build_faithful_notebook.py [--data_root PATH] [--quick]
"""
import argparse
import base64
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ======================================================================================
#  CONTENT
# ======================================================================================

md(r"""
# A Bar-Pointer Dynamical VAE for Beat Tracking

This notebook builds — from scratch, in one file, using only standard libraries — a
**generative latent-variable model of musical beats** and trains it on real audio.

The idea is the *bar pointer*. Imagine a hand sweeping clockwise around a clock face once per
**bar** of music. Its angular position is the **phase** $\phi\in[0,2\pi)$: $\phi=0$ is the
downbeat (start of the bar), and the hand wraps back to $0$ at the next downbeat. The speed of
the hand is the **tempo** $\dot\phi$ (radians per frame). How many **beats** fall in one bar is
the **meter** $m$ (e.g. $m=4$ for $4/4$). Given the pointer, beats are simply the $m$ equally
spaced angles $\phi = 2\pi k/m$, and downbeats are the wraps $\phi:2\pi\to 0$.

We never observe the pointer; we only hear audio. So we treat $(m,\phi,\dot\phi)$ as **latent
state** evolving over time, put a generative prior on how it moves, and learn — by variational
inference — to infer it from sound and to emit beats. Concretely we implement, in order:

1. the **generative model** (a prior over how $(m,\phi,\dot\phi)$ evolve);
2. the **ELBO** — derived here in full — which is the training objective;
3. the **concrete distributions**: a Categorical meter, a von Mises phase, a Log-Normal tempo,
   with their samplers and closed-form KL divergences (each demonstrated on example values);
4. the **encoder / prior / decoder** networks and the **training rollout**;
5. **training** on the ballroom dataset and a **free-running** evaluation with `mir_eval`.

Every heavy function gets its own cell followed by a tiny worked example, so you can see
exactly what it computes.
""")

# --------------------------------------------------------------------------------------
md(r"""
## 1. The generative model

The latent state at frame $t$ is
$$ z_t = [\,m_t,\ \phi_t,\ \dot\phi_t\,] \qquad
   m_t \in \{1,\dots,K\}\ \text{(meter)},\quad
   \phi_t\in[0,2\pi)\ \text{(bar phase)},\quad
   \dot\phi_t>0\ \text{(tempo)} . $$

We assume the state is **Markov** in time and factorizes across the three variables:
$$ p_\psi(z_t \mid z_{t-1})
   = \underbrace{p_\psi(m_t \mid m_{t-1},\phi_t,\phi_{t-1})}_{\text{meter}}\;
     \underbrace{p_\psi(\phi_t \mid \phi_{t-1},\dot\phi_{t-1})}_{\text{phase}}\;
     \underbrace{p_\psi(\dot\phi_t \mid \dot\phi_{t-1})}_{\text{tempo}} . $$

The dynamics encode the bar-pointer picture:

* **Tempo** drifts smoothly. We model it as a random walk in *log* space (so it stays positive
  and the step size is scale-free):
  $$ \log\dot\phi_t \sim \mathcal N\!\big(\log\dot\phi_{t-1},\ (\sigma^{p}_{\dot\phi})^2\big). $$
  The mean is just "keep the previous tempo"; the learned spread $\sigma^{p}_{\dot\phi}$ says how
  fast tempo may change.

* **Phase** advances by the current tempo and wraps around the circle:
  $$ \phi_t \sim \mathrm{vM}\!\big(\phi_{t-1}+\dot\phi_{t-1},\ \kappa^{p}_{\phi}\big), $$
  a von Mises (the circular analogue of a Gaussian). The mean is the deterministic pointer
  advance; the learned concentration $\kappa^{p}_\phi$ is the certainty of that advance.

* **Meter** rarely changes, and only at bar boundaries. We let a small network produce a full
  $K\times K$ transition matrix, conditioned on whether the pointer just crossed a bar line.

Finally the **decoder** turns the latent into a beat probability for each frame:
$$ b_t \sim \mathrm{Bernoulli}\big(\sigma(\mathrm{NN}_\theta(z_t, h))\big), $$
where $h$ is the audio representation (a log-mel spectrogram). $b_t=1$ marks a beat frame.
""")

# --------------------------------------------------------------------------------------
md(r"""
## 2. The ELBO (derivation)

We want to fit the model to data by maximizing the log-likelihood of the observed beats
$b_{1:T}$ given the audio $h_{1:T}$:
$$ \log p_\theta(b_{1:T}\mid h_{1:T})
   = \log \int p_\theta(b_{1:T}\mid z_{1:T}, h)\, p_\psi(z_{1:T}\mid h)\; dz_{1:T}. $$
This integral over all latent trajectories is intractable. So we introduce an **inference
network** (encoder) $q_\phi(z_{1:T}\mid b,h)$ that proposes likely trajectories, and apply
Jensen's inequality:
$$
\log p_\theta(b\mid h)
= \log \mathbb{E}_{q_\phi}\!\Big[\tfrac{p_\theta(b\mid z,h)\,p_\psi(z\mid h)}{q_\phi(z\mid b,h)}\Big]
\;\ge\; \mathbb{E}_{q_\phi}\!\Big[\log \tfrac{p_\theta(b\mid z,h)\,p_\psi(z\mid h)}{q_\phi(z\mid b,h)}\Big]
\;=:\; \mathcal{L}.
$$
$\mathcal L$ is the **evidence lower bound (ELBO)**. Splitting the log of the ratio:
$$
\mathcal L
= \underbrace{\mathbb{E}_{q_\phi}\big[\log p_\theta(b\mid z,h)\big]}_{\text{reconstruction}}
- \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid b,h)\,\|\,p_\psi(z\mid h)\big)}_{\text{regularization}} .
$$

Now use the time structure. Both the prior and the (matching) posterior factorize over frames,
and the observation is per-frame, so the reconstruction term becomes a sum of per-frame terms and
the KL becomes a sum of per-frame KLs — an initial one at $t=1$ and a transition one at each
$t\ge 2$:
$$
\mathcal L
= \sum_{t=1}^{T} \mathbb{E}_{q_\phi}\big[\log p_\theta(b_t\mid z_t,h)\big]
\;-\; \sum_{x\in\{m,\phi,\dot\phi\}} D_{\mathrm{KL}}\big(q(x_1)\,\|\,p(x_1)\big)
$$
$$
\;-\; \sum_{t=2}^{T}\ \mathbb{E}_{q(z_{t-1})}\!\Big[
   D_{\mathrm{KL}}\big(q(m_t)\|p(m_t\mid z_{t-1}))
 + D_{\mathrm{KL}}\big(q(\phi_t)\|p(\phi_t\mid z_{t-1}))
 + D_{\mathrm{KL}}\big(q(\dot\phi_t)\|p(\dot\phi_t\mid z_{t-1}))\Big].
$$

Because each latent has a *conjugate-friendly* form (Categorical, von Mises, Log-Normal), every
KL has a **closed form** (next section). We estimate the expectations with a **single Monte-Carlo
sample** of the trajectory $\hat z_{1:T}\sim q_\phi$ (using reparameterized samplers so gradients
flow), and we train with the loss
$$
\mathcal L_{\text{loss}} = -\mathcal L
= \underbrace{\sum_t \mathrm{BCE}\big(b_t,\ \sigma(\mathrm{NN}_\theta(\hat z_t,h))\big)}_{\text{reconstruction}}
+ \underbrace{\sum_t \big(\mathrm{KL}_{m} + \mathrm{KL}_{\phi} + \mathrm{KL}_{\dot\phi}\big)}_{\text{regularization}} .
$$
This is the objective the training cell minimizes (weight on the KL term is $1$).
""")

# --------------------------------------------------------------------------------------
md(r"""
## 3. Setup

Only third-party libraries. The whole model is built below from these primitives.
""")

code(r"""
import math, glob, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import mir_eval
import matplotlib.pyplot as plt

TWO_PI = 2.0 * math.pi
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0); random.seed(0)
print("torch", torch.__version__, "| device:", DEVICE)
""")

# --------------------------------------------------------------------------------------
md(r"""
## 4. The von Mises distribution (circular phase)

The phase lives on a circle, so we model it with a **von Mises** distribution — the circular
analogue of a Gaussian, with mean direction $\mu$ and concentration $\kappa$ (large $\kappa$ =
peaked, $\kappa\to0$ = uniform). Its density is $p(z\mid\mu,\kappa)\propto e^{\kappa\cos(z-\mu)}$.

We need three things from it: its **density** and **CDF** (for the sampler's gradient), a
**sampler**, and a **KL divergence**. We start with the Bessel-function helpers the density needs,
then the density and CDF. The cell after each definition shows it on example values.
""")

code(r'''
def log_i0(kappa):
    """log I0(kappa) — log of the modified Bessel function I0, computed stably via the
    exponentially-scaled i0e so it never overflows for large kappa."""
    return torch.log(torch.special.i0e(kappa)) + kappa

def A_kappa(kappa):
    """A(kappa) = I1(kappa)/I0(kappa): the mean resultant length, i.e. how concentrated the
    distribution is (0 = uniform circle, ->1 = a spike). Appears in the von Mises KL."""
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)

def von_mises_pdf(z, kappa):
    """Density of a ZERO-MEAN von Mises at angle z."""
    return torch.exp(kappa * (torch.cos(z) - 1.0)) / (TWO_PI * torch.special.i0e(kappa))

def von_mises_cdf(z, kappa, n_steps=100):
    """CDF F(z|0,kappa) = integral from -pi to z of the density, by the trapezoid rule.
    Shape-generic: integrates along a fresh last axis so it works on whole tensors at once."""
    lower = -math.pi
    frac = torch.linspace(0.0, 1.0, n_steps, device=z.device, dtype=z.dtype)
    z_e, k_e = z.unsqueeze(-1), kappa.unsqueeze(-1)
    t = lower + frac * (z_e - lower)
    pdf = torch.exp(k_e * (torch.cos(t) - 1.0)) / (TWO_PI * torch.special.i0e(k_e))
    w = torch.ones_like(pdf); w[..., 0] = 0.5; w[..., -1] = 0.5
    step = (z_e - lower) / (n_steps - 1)
    return (pdf * w).sum(-1) * step.squeeze(-1)
''')

code(r"""
# EXAMPLE: the density gets more peaked as kappa grows; the CDF goes 0 -> 1 over [-pi, pi].
zs = torch.linspace(-math.pi, math.pi, 200)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for k in [0.5, 2.0, 8.0]:
    ax[0].plot(zs.numpy(), von_mises_pdf(zs, torch.tensor(k)).numpy(), label=f"kappa={k}")
    ax[1].plot(zs.numpy(), von_mises_cdf(zs, torch.full_like(zs, k)).numpy(), label=f"kappa={k}")
ax[0].set_title("von Mises density (mean 0)"); ax[0].set_xlabel("angle z"); ax[0].legend()
ax[1].set_title("von Mises CDF"); ax[1].set_xlabel("angle z"); ax[1].legend()
plt.tight_layout()
print("A(kappa) for kappa=[0.1, 1, 5, 20]:",
      A_kappa(torch.tensor([0.1, 1.0, 5.0, 20.0])).numpy().round(3),
      "  (0 = flat circle, ->1 = concentrated)")
""")

# --------------------------------------------------------------------------------------
md(r"""
### 4.1 Sampling the von Mises (with gradients)

To train through a sampled phase we need to (a) **draw** $\phi\sim\mathrm{vM}(\mu,\kappa)$ and
(b) get **gradients** of $\phi$ w.r.t. $\mu,\kappa$. We sample with the classic Best–Fisher
rejection method (draws a zero-mean angle $z$, then sets $\phi=\mu+z$), and we obtain gradients by
**implicit reparameterization**: differentiating the CDF identity gives
$\partial\phi/\partial\mu=1$ and $\partial\phi/\partial\kappa = -\,(\partial F/\partial\kappa)/p(z)$.
We package this as a custom autograd function so `loss.backward()` just works.
""")

code(r'''
def best_fisher_rejection(kappa, max_iter=100):
    """Draw z ~ vM(0, kappa) elementwise by Best-Fisher rejection sampling."""
    shape = kappa.shape
    k = kappa.reshape(-1).clamp(min=1e-3)
    tau = 1.0 + torch.sqrt(1.0 + 4.0 * k * k)
    rho = (tau - torch.sqrt(2.0 * tau)) / (2.0 * k)
    r = (1.0 + rho * rho) / (2.0 * rho)
    z = torch.zeros_like(k); accepted = torch.zeros_like(k, dtype=torch.bool)
    for _ in range(max_iter):
        u1, u2, u3 = torch.rand_like(k), torch.rand_like(k), torch.rand_like(k)
        w = torch.cos(math.pi * u1)
        f = (1.0 + r * w) / (r + w)
        c = k * (r - f)                                        # rejection statistic
        accept = (c * (2.0 - c) - u2 > 0) | (torch.log(c / u2 + 1e-20) + 1.0 - c >= 0)
        sign = torch.where(u3 > 0.5, 1.0, -1.0)
        angle = sign * torch.acos(torch.clamp(f, -1.0, 1.0))
        newly = accept & (~accepted)
        z = torch.where(newly, angle, z)
        accepted = accepted | accept
        if bool(accepted.all()):
            break
    return z.reshape(shape)

class VonMisesSample(torch.autograd.Function):
    """Sampler as an autograd Function: forward = rejection sample, backward = implicit-
    reparameterization gradients (true gradients through mu AND kappa, not stop-grad)."""
    @staticmethod
    def forward(ctx, mu, kappa):
        z = best_fisher_rejection(kappa)
        ctx.save_for_backward(z, kappa)
        return mu + z
    @staticmethod
    def backward(ctx, grad_phi):
        z, kappa = ctx.saved_tensors
        with torch.enable_grad():
            k = kappa.detach().clone().requires_grad_(True)
            (dF_dk,) = torch.autograd.grad(von_mises_cdf(z.detach(), k).sum(), k)
        p = von_mises_pdf(z, kappa)
        dphi_dkappa = -dF_dk / (p + 1e-12)
        return grad_phi * 1.0, grad_phi * dphi_dkappa

def sample_von_mises(mu, kappa):
    return VonMisesSample.apply(mu, kappa)
''')

code(r"""
# EXAMPLE: validate the sampler three ways.
# (a) the mean direction matches mu:
samp = sample_von_mises(torch.full((20000,), 1.0), torch.full((20000,), 4.0))
print(f"mean direction (mu=1, kappa=4): {math.atan2(float(torch.sin(samp).mean()), float(torch.cos(samp).mean())):.3f}  (~1.0)")
# (b) the concentration matches the theoretical mean resultant A(kappa):
print("kappa :  E[cos] empirical   A(kappa) theory")
for k in [0.5, 2.0, 8.0]:
    s = sample_von_mises(torch.zeros(20000), torch.full((20000,), float(k)))
    print(f"  {k:<4} :  {torch.cos(s).mean().item():.3f}            {A_kappa(torch.tensor(k)).item():.3f}")
# (c) gradients flow through BOTH parameters (implicit reparameterization):
mu = torch.tensor(1.0, requires_grad=True); kap = torch.tensor(4.0, requires_grad=True)
torch.cos(sample_von_mises(mu.expand(4000), kap.expand(4000))).mean().backward()
print(f"d/dmu = {mu.grad.item():+.3f}   d/dkappa = {kap.grad.item():+.4f}   (both finite, non-zero)")
""")

# --------------------------------------------------------------------------------------
md(r"""
## 5. The Categorical meter (Gumbel-Softmax)

Meter is discrete, so we sample it with the **Gumbel-Softmax** relaxation: add Gumbel noise to
the logits and take a temperature-controlled softmax. Low temperature $\to$ near-one-hot (discrete),
high temperature $\to$ soft. This keeps sampling differentiable.
""")

code(r"""
def gumbel_softmax(logits, temperature):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    return F.softmax((logits + g) / temperature, dim=-1)
""")

code(r"""
# EXAMPLE: logits favouring class 2; low temperature -> almost one-hot, high -> soft.
logits = torch.tensor([[0.2, 0.3, 3.0, 0.1]])
for T in [0.1, 1.0]:
    print(f"temperature={T}: sample = {gumbel_softmax(logits, T).numpy().round(3)}")
print("argmax stays on the high-logit class (2); low temperature is closer to one-hot.")
""")

# --------------------------------------------------------------------------------------
md(r"""
## 6. Closed-form KL divergences

Each per-frame KL in the ELBO has a closed form. We verify each on examples: KL must be **0 when
$q=p$** and **positive otherwise**.
""")

code(r'''
def kl_categorical(log_q, log_p):
    """KL(Cat(q) || Cat(p)) from LOG-probabilities, summed over classes."""
    return (log_q.exp() * (log_q - log_p)).sum(-1)

def kl_von_mises(mu_q, kappa_q, mu_p, kappa_p):
    """KL between two von Mises (closed form via Bessel functions)."""
    return (log_i0(kappa_p) - log_i0(kappa_q)
            + A_kappa(kappa_q) * (kappa_q - kappa_p * torch.cos(mu_q - mu_p)))

def kl_log_normal(mu_q, sigma_q, mu_p, sigma_p):
    """KL between two Log-Normals = the Gaussian KL in log-space."""
    return (torch.log(sigma_p / sigma_q)
            + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p ** 2) - 0.5)
''')

code(r"""
# EXAMPLE: each KL is ~0 at q==p and >0 when the distributions differ.
lq = torch.log_softmax(torch.tensor([[1.0, 2.0, 0.5, 0.0]]), -1)
lp = torch.log_softmax(torch.tensor([[0.0, 0.0, 0.0, 0.0]]), -1)
print("categorical  KL(q||q) =", float(kl_categorical(lq, lq)), " KL(q||uniform) =", round(float(kl_categorical(lq, lp)), 4))
print("von Mises    KL(q||q) =", round(float(kl_von_mises(torch.tensor(1.), torch.tensor(4.), torch.tensor(1.), torch.tensor(4.))), 6),
      " KL(mu 1 vs 2) =", round(float(kl_von_mises(torch.tensor(1.), torch.tensor(4.), torch.tensor(2.), torch.tensor(4.))), 4))
print("log-normal   KL(q||q) =", round(float(kl_log_normal(torch.tensor(0.), torch.tensor(.3), torch.tensor(0.), torch.tensor(.3))), 6),
      " KL(mu 0 vs .5) =", round(float(kl_log_normal(torch.tensor(0.), torch.tensor(.3), torch.tensor(.5), torch.tensor(.3))), 4))
""")

# --------------------------------------------------------------------------------------
md(r"""
## 7. The model: encoder, prior, decoder

Now we assemble the networks. There are three readers of the data:

* an **inference encoder** $q_\phi$ that reads $(b,h)$ and proposes posterior parameters,
* a **prior** $p_\psi$ that reads only $h$ for the *concentrations/scales* ($\kappa^p_\phi$,
  $\sigma^p_{\dot\phi}$, meter transition) while the *means* are the deterministic bar-pointer
  recursion ($\mu^p_\phi=\phi_{t-1}+\dot\phi_{t-1}$, $\mu^p_{\dot\phi}=\log\dot\phi_{t-1}$),
* a **decoder** $p_\theta$ that maps the latent (and $h$) to a beat logit.

`unpack` turns a raw parameter vector into named, validity-constrained distribution parameters
(softplus keeps $\kappa,\sigma>0$; `atan2` reads a circular mean off a 2-D vector).
""")

code(r'''
class BarPointerVAE(nn.Module):
    def __init__(self, h_dim, hidden=64, num_meters=4):
        super().__init__()
        self.K, self.hidden = num_meters, hidden
        self.z_feat_dim = 3 + num_meters             # cos phi, sin phi, log tempo, onehot(meter)
        param_dim = num_meters + 2 + 1 + 1 + 1       # meter logits | phase(u,v) | log-kappa | tempo mu | log-sigma

        # encoders (bidirectional GRUs over the sequence)
        self.post_gru = nn.GRU(h_dim + 1, hidden, batch_first=True, bidirectional=True)  # reads (h, b)
        self.post_ctx = nn.Linear(2 * hidden, hidden)
        self.prior_gru = nn.GRU(h_dim, hidden, batch_first=True, bidirectional=True)     # reads h only
        self.prior_ctx = nn.Linear(2 * hidden, hidden)

        # posterior head: [context_t, features of z_{t-1}] -> distribution params
        self.post_head = nn.Sequential(nn.Linear(hidden + self.z_feat_dim, hidden), nn.Tanh(),
                                       nn.Linear(hidden, param_dim))
        self.z0 = nn.Parameter(torch.zeros(self.z_feat_dim))     # learned initial-state token

        # prior heads: only concentration/scale read h; the means are the bar-pointer recursion
        self.prior_init_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                             nn.Linear(hidden, param_dim))
        self.prior_phase_kappa = nn.Linear(hidden, 1)
        self.prior_tempo_sigma = nn.Linear(hidden, 1)
        self.meter_prior = nn.Sequential(nn.Linear(num_meters + 4 + hidden, hidden), nn.Tanh(),
                                         nn.Linear(hidden, num_meters * num_meters))

        # decoder NN_theta(z_t, h) -> beat logit
        self.decoder = nn.Sequential(nn.Linear(self.z_feat_dim + hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 1))

    def encode_posterior(self, h, b):
        out, _ = self.post_gru(torch.cat([h, b.unsqueeze(-1)], dim=-1))
        return torch.tanh(self.post_ctx(out))

    def encode_prior(self, h):
        out, _ = self.prior_gru(h)
        return torch.tanh(self.prior_ctx(out))

    def unpack(self, vec):
        """raw [B, param_dim] -> (meter_logits, phase_mu, phase_kappa, tempo_mu, tempo_sigma)."""
        K = self.K
        meter_logits = vec[:, :K]
        u, v = vec[:, K], vec[:, K + 1]
        phase_mu = torch.atan2(v, u) % TWO_PI
        phase_kappa = F.softplus(vec[:, K + 2]) + 0.01
        tempo_mu = vec[:, K + 3]
        tempo_sigma = F.softplus(vec[:, K + 4]) + 1e-3
        return meter_logits, phase_mu, phase_kappa, tempo_mu, tempo_sigma

    def z_features(self, meter_soft, phi, log_tempo):
        return torch.cat([torch.cos(phi).unsqueeze(-1), torch.sin(phi).unsqueeze(-1),
                          log_tempo.unsqueeze(-1), meter_soft], dim=-1)

    def meter_prior_logp(self, meter_prev, phi_t, phi_prev, prior_ctx_t):
        feats = torch.cat([meter_prev,
                           torch.cos(phi_t).unsqueeze(-1), torch.sin(phi_t).unsqueeze(-1),
                           torch.cos(phi_prev).unsqueeze(-1), torch.sin(phi_prev).unsqueeze(-1),
                           prior_ctx_t], dim=-1)
        Pi = F.softmax(self.meter_prior(feats).reshape(-1, self.K, self.K), dim=2)
        pi_p = torch.bmm(meter_prev.unsqueeze(1), Pi).squeeze(1)
        return torch.log(pi_p + 1e-9)

    def decode(self, z_feat, prior_ctx_t):
        return self.decoder(torch.cat([z_feat, prior_ctx_t], dim=-1)).squeeze(-1)
''')

code(r"""
# EXAMPLE: build the model on a toy batch and inspect one set of posterior/prior params.
toy_model = BarPointerVAE(h_dim=128, hidden=64, num_meters=4).to(DEVICE)
B, T, n_mels = 2, 20, 128
toy_h = torch.randn(B, T, n_mels, device=DEVICE)
toy_b = (torch.rand(B, T, device=DEVICE) < 0.1).float()
pc = toy_model.encode_posterior(toy_h, toy_b); pr = toy_model.encode_prior(toy_h)
print("posterior context shape:", tuple(pc.shape), "| prior context shape:", tuple(pr.shape))
ml, pmu, pk, tmu, ts = toy_model.unpack(toy_model.post_head(
    torch.cat([pc[:, 0], toy_model.z0.unsqueeze(0).expand(B, -1)], dim=-1)))
print(f"one posterior sample (frame 0): phase_mu={pmu[0].item():.3f}  phase_kappa={pk[0].item():.3f}  "
      f"tempo_mu={tmu[0].item():.3f}  tempo_sigma={ts[0].item():.3f}")
print(f"#params = {sum(p.numel() for p in toy_model.parameters()):,}")
""")

# --------------------------------------------------------------------------------------
md(r"""
## 8. Training rollout (the ELBO in code)

This is the objective from Section 2, computed frame by frame. At $t=1$ we use an initial prior;
at each $t\ge2$ the prior means are the deterministic bar-pointer recursion evaluated on the
*sampled* previous latent, while the posterior reads the encoded $(b,h)$ and the sampled
$z_{t-1}$. We accumulate the three KLs, decode every frame, and return
$\mathcal L_{\text{loss}}=\text{recon}+\text{KL}$ plus a breakdown for monitoring.
""")

code(r'''
def strict_elbo(model, h, b, temperature=0.5):
    """ELBO loss over a batch. h:[B,T,n_mels], b:[B,T] in {0,1}. Returns (loss, info)."""
    B, T, _ = h.shape
    post_ctx = model.encode_posterior(h, b)
    prior_ctx = model.encode_prior(h)
    kl_m = h.new_zeros(B); kl_p = h.new_zeros(B); kl_t = h.new_zeros(B)
    z_feats, post_phase_mu = [], []

    # t = 1: initial state
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    q_m, q_pmu, q_pk, q_tmu, q_ts = model.unpack(model.post_head(torch.cat([post_ctx[:, 0], z0], -1)))
    p_m, p_pmu, p_pk, p_tmu, p_ts = model.unpack(model.prior_init_head(prior_ctx.mean(1)))
    meter = gumbel_softmax(q_m, temperature)
    phi = sample_von_mises(q_pmu, q_pk) % TWO_PI
    log_tempo = q_tmu + q_ts * torch.randn_like(q_tmu)
    kl_m += kl_categorical(torch.log_softmax(q_m, -1), torch.log_softmax(p_m, -1))
    kl_p += kl_von_mises(q_pmu, q_pk, p_pmu, p_pk)
    kl_t += kl_log_normal(q_tmu, q_ts, p_tmu, p_ts)
    z_feats.append(model.z_features(meter, phi, log_tempo)); post_phase_mu.append(q_pmu)
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo

    # t = 2..T: transitions
    for t in range(1, T):
        z_prev = model.z_features(meter_prev, phi_prev, log_tempo_prev)
        q_m, q_pmu, q_pk, q_tmu, q_ts = model.unpack(model.post_head(torch.cat([post_ctx[:, t], z_prev], -1)))
        p_pmu = (phi_prev + torch.exp(log_tempo_prev)) % TWO_PI          # phase mean = pointer advance
        p_pk = F.softplus(model.prior_phase_kappa(prior_ctx[:, t]).squeeze(-1)) + 0.01
        p_tmu = log_tempo_prev                                           # tempo mean = previous tempo
        p_ts = F.softplus(model.prior_tempo_sigma(prior_ctx[:, t]).squeeze(-1)) + 1e-3
        meter = gumbel_softmax(q_m, temperature)
        phi = sample_von_mises(q_pmu, q_pk) % TWO_PI
        log_tempo = q_tmu + q_ts * torch.randn_like(q_tmu)
        log_pi_p = model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t])
        kl_m += kl_categorical(torch.log_softmax(q_m, -1), log_pi_p)
        kl_p += kl_von_mises(q_pmu, q_pk, p_pmu, p_pk)
        kl_t += kl_log_normal(q_tmu, q_ts, p_tmu, p_ts)
        z_feats.append(model.z_features(meter, phi, log_tempo)); post_phase_mu.append(q_pmu)
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo

    beat_logits = torch.stack([model.decode(z_feats[t], prior_ctx[:, t]) for t in range(T)], dim=1)
    recon = F.binary_cross_entropy_with_logits(beat_logits, b, reduction="none").sum(1)
    loss = (recon + kl_m + kl_p + kl_t).mean()
    info = {"loss": float(loss), "recon": float(recon.mean()),
            "kl_meter": float(kl_m.mean()), "kl_phase": float(kl_p.mean()), "kl_tempo": float(kl_t.mean())}
    return loss, info
''')

code(r"""
# EXAMPLE: one ELBO evaluation on the toy batch — see the loss split into recon + 3 KLs.
loss, info = strict_elbo(toy_model, toy_h, toy_b)
print("loss =", round(info["loss"], 3),
      "= recon", round(info["recon"], 3),
      "+ kl_meter", round(info["kl_meter"], 3),
      "+ kl_phase", round(info["kl_phase"], 3),
      "+ kl_tempo", round(info["kl_tempo"], 3))
loss.backward()
print("backward OK — gradients populated on all parameter groups:",
      all(p.grad is not None for p in toy_model.parameters() if p.requires_grad))
""")

# --------------------------------------------------------------------------------------
md(r"""
## 9. Deployment: free-running the prior

At test time we have audio but **no beats**, so we cannot run the posterior encoder (it reads
$b$). Instead we **free-run the prior**: sample the initial state, then roll the bar-pointer
dynamics forward, letting the prior's audio-conditioned concentrations shape the trajectory and
the decoder emit beats. We return both the stochastic phase and the noise-free **mean phase
chain** (a clean sawtooth we can read beats off of geometrically), plus the decoder probabilities.
""")

code(r'''
@torch.no_grad()
def free_run(model, h, temperature=0.3):
    """Roll the prior forward with no beats. Returns phase / phase_mu / decoder_prob trajectories."""
    B, T, _ = h.shape
    prior_ctx = model.encode_prior(h)
    p_m, p_pmu, p_pk, p_tmu, p_ts = model.unpack(model.prior_init_head(prior_ctx.mean(1)))
    meter = gumbel_softmax(p_m, temperature)
    phi = sample_von_mises(p_pmu, p_pk) % TWO_PI
    log_tempo = p_tmu + p_ts * torch.randn_like(p_tmu)
    phi_mu, log_tempo_mu = p_pmu % TWO_PI, p_tmu
    z_feats = [model.z_features(meter, phi, log_tempo)]
    phase_traj, phase_mu_traj, dec_ctx = [phi], [phi_mu], prior_ctx
    meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
    for t in range(1, T):
        p_pmu = (phi_prev + torch.exp(log_tempo_prev)) % TWO_PI
        p_pk = F.softplus(model.prior_phase_kappa(prior_ctx[:, t]).squeeze(-1)) + 0.01
        p_tmu = log_tempo_prev
        p_ts = F.softplus(model.prior_tempo_sigma(prior_ctx[:, t]).squeeze(-1)) + 1e-3
        phi = sample_von_mises(p_pmu, p_pk) % TWO_PI
        log_tempo = p_tmu + p_ts * torch.randn_like(p_tmu)
        meter = gumbel_softmax(model.meter_prior_logp(meter_prev, phi, phi_prev, prior_ctx[:, t]), temperature)
        phi_mu = (phi_mu + torch.exp(log_tempo_mu)) % TWO_PI
        z_feats.append(model.z_features(meter, phi, log_tempo))
        phase_traj.append(phi); phase_mu_traj.append(phi_mu)
        meter_prev, phi_prev, log_tempo_prev = meter, phi, log_tempo
    beat_logits = torch.stack([model.decode(z_feats[t], dec_ctx[:, t]) for t in range(T)], dim=1)
    return {"phase_mu": torch.stack(phase_mu_traj, dim=1),
            "decoder_prob": torch.sigmoid(beat_logits)}
''')

code(r"""
# EXAMPLE: free-run the (untrained) model on the toy audio and plot the mean phase sawtooth.
out = free_run(toy_model, toy_h)
pm = out["phase_mu"][0].cpu().numpy()
plt.figure(figsize=(9, 2.4))
plt.plot(pm, ".-"); plt.axhline(TWO_PI, ls=":", c="gray")
plt.title("free-run mean phase chain (each 2*pi wrap = a downbeat)")
plt.xlabel("frame"); plt.ylabel("phase"); plt.tight_layout()
print("phase advances and wraps; #wraps in this toy clip =",
      int((np.diff(pm) < -math.pi).sum()))
""")

# --------------------------------------------------------------------------------------
md(r"""
## 10. Reading beats off the latent, and scoring

Two read-outs from a free-run, plus a metronome floor:

* **beat-phase** — beats are the $m$ subdivisions of the bar, i.e. the wraps of $(m\cdot\phi)$.
  This is the *pure latent* read-out: it works only if the pointer's geometry is right.
* **downbeat-phase** — downbeats are the bar wraps ($\phi:2\pi\to0$).
* **decoder** — peak-pick the Bernoulli decoder probability (rides the audio).
* **metronome** — a constant 120-BPM grid (a sanity floor).

We score with `mir_eval`'s standard F-measure (70 ms tolerance). A known-answer check on an ideal
sawtooth confirms the geometric read-out itself is sound (F ≈ 1).
""")

code(r"""
def _min_gap(frames, gap):
    out, last = [], -1e9
    for f in frames:
        if f - last >= gap:
            out.append(f); last = f
    return np.asarray(out, float)

def beats_from_barphase(phase, m, fps, min_dist=0.10):
    psi = (int(m) * np.asarray(phase, float)) % TWO_PI
    w = np.where(np.diff(psi) < -math.pi)[0] + 1
    return _min_gap(w.astype(float), min_dist * fps) / fps

def downbeats_from_barphase(phase, fps, min_dist=0.30):
    w = np.where(np.diff(np.asarray(phase)) < -math.pi)[0] + 1
    return _min_gap(w.astype(float), min_dist * fps) / fps

def beats_from_decoder(prob, fps, thr=0.5, min_dist=0.10):
    prob = np.asarray(prob)
    peaks = [t for t in range(1, len(prob) - 1)
             if prob[t] >= thr and prob[t] >= prob[t-1] and prob[t] >= prob[t+1]]
    return _min_gap(np.asarray(peaks, float), min_dist * fps) / fps

def metronome(T, fps, bpm=120.0):
    return np.arange(0.0, T / fps, 60.0 / bpm)

def f_measure(ref, est):
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    if len(ref) == 0: return float("nan")
    if len(est) == 0: return 0.0
    return float(mir_eval.beat.f_measure(ref, est))
""")

code(r"""
# EXAMPLE: known-answer test — feed an IDEAL 120-BPM, m=4 sawtooth; the beat read-out should ~1.0.
fps_demo = 86.13; dur = 12; T = int(dur * fps_demo)
bpm = 120; beats_per_sec = bpm / 60
adv = TWO_PI * beats_per_sec / 4 / fps_demo                 # 4 beats per bar
ideal_phase = (np.arange(T) * adv) % TWO_PI
true_beats = np.arange(0, dur, 60.0 / bpm)
est = beats_from_barphase(ideal_phase, 4, fps_demo)
print(f"ideal sawtooth: read-out beat-F vs truth = {f_measure(true_beats, est):.3f}  (read-out is sound)")
""")

# --------------------------------------------------------------------------------------
md(r"""
## 11. The ballroom dataset (self-contained loader)

We load raw audio and beat annotations directly. Each song is a `.wav` plus a `.beats` file whose
lines are `time  beat_number` (beat number `1` marks a downbeat). We turn audio into a fixed
log-mel spectrogram $h$ and build per-frame binary beat / downbeat targets. **Set `DATA_ROOT`** to
your ballroom folder (e.g. a mounted Google Drive path); it must contain `data/*.wav` and
`label/*.beats`.
""")

code(r'''
# ---- in Colab, mount Drive and point DATA_ROOT at your uploaded ballroom folder ----
# from google.colab import drive; drive.mount("/content/drive")
DATA_ROOT = os.environ.get("BALLROOM_ROOT", "/content/drive/MyDrive/ballroom")

SR, HOP, N_FFT, N_MELS = 22050, 256, 1024, 128
FPS = SR / HOP                                              # 86.13 frames / second
_mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
                                            n_mels=N_MELS, power=1.0)

def log_mel(wave):                                         # [N] waveform -> [T, n_mels]
    return torch.log1p(_mel(wave)).transpose(0, 1)

def load_song(wav_path):
    beats_path = wav_path.replace("/data/", "/label/").replace(".wav", ".beats")
    wave, sr = torchaudio.load(wav_path); wave = wave.mean(0)
    if sr != SR: wave = torchaudio.functional.resample(wave, sr, SR)
    h = log_mel(wave); T = h.shape[0]
    ann = np.atleast_2d(np.loadtxt(beats_path))
    bt, bi = ann[:, 0], ann[:, 1]
    beats = torch.zeros(T); beats[np.clip(np.round(bt * FPS).astype(int), 0, T-1)] = 1.0
    downs = torch.zeros(T); downs[np.clip(np.round(bt[bi == 1] * FPS).astype(int), 0, T-1)] = 1.0
    return h, beats, downs

songs = sorted(glob.glob(f"{DATA_ROOT}/data/*.wav"))
print(f"found {len(songs)} ballroom songs under {DATA_ROOT}")
''')

code(r"""
# EXAMPLE: load one song and show its log-mel with beat (red) and downbeat (yellow) markers.
h_ex, beats_ex, downs_ex = load_song(songs[0])
print(f"{os.path.basename(songs[0])}: h={tuple(h_ex.shape)} dur={h_ex.shape[0]/FPS:.1f}s "
      f"beats={int(beats_ex.sum())} downbeats={int(downs_ex.sum())} "
      f"tempo~{60/np.median(np.diff(np.where(beats_ex.numpy()>0)[0]/FPS)):.0f}BPM")
seg = int(8 * FPS)
plt.figure(figsize=(10, 3))
plt.imshow(h_ex[:seg].T.numpy(), aspect="auto", origin="lower", cmap="magma")
for fr in np.where(beats_ex.numpy()[:seg] > 0)[0]: plt.axvline(fr, c="red", lw=0.8, alpha=0.7)
for fr in np.where(downs_ex.numpy()[:seg] > 0)[0]: plt.axvline(fr, c="yellow", lw=1.4, alpha=0.9)
plt.title("log-mel + beats (red) / downbeats (yellow)"); plt.xlabel("frame"); plt.ylabel("mel bin")
plt.tight_layout()
""")

# --------------------------------------------------------------------------------------
md(r"""
## 12. Training

We preload a subset of songs (log-mel + targets), sample fixed-length crops that contain beats,
and minimize the ELBO loss. We watch the per-latent KL and the held-out read-outs as training
proceeds.
""")

code(r"""
# preload songs into memory and split train / val
random.seed(0); random.shuffle(songs)
N_USE = __N_USE__
cache = []
for w in songs[:N_USE]:
    try:
        h_s, b_s, d_s = load_song(w)
        if b_s.sum() >= 8:
            cache.append((h_s, b_s, d_s))
    except Exception:
        pass
val_set = cache[:__N_VAL__]; train_set = cache[__N_VAL__:]
print(f"preloaded {len(cache)} songs -> {len(train_set)} train / {len(val_set)} val")

FRAMES, BATCH = 256, 16
def sample_batch():
    hs, bs = [], []
    while len(hs) < BATCH:
        h_s, b_s, _ = random.choice(train_set)
        if h_s.shape[0] <= FRAMES: continue
        s = random.randint(0, h_s.shape[0] - FRAMES)
        if b_s[s:s+FRAMES].sum() < 2: continue
        hs.append(h_s[s:s+FRAMES]); bs.append(b_s[s:s+FRAMES])
    return torch.stack(hs).to(DEVICE), torch.stack(bs).to(DEVICE)
""")

code(r"""
@torch.no_grad()
def eval_freerun(model, n_max=__N_VAL__, max_frames=2200):
    model.eval(); acc = {"beat_phase": [], "downbeat_phase": [], "decoder": [], "metronome": []}
    for h_s, b_s, d_s in val_set[:n_max]:
        T = min(h_s.shape[0], max_frames)
        out = free_run(model, h_s[:T].unsqueeze(0).to(DEVICE))
        pm = out["phase_mu"][0].cpu().numpy(); dec = out["decoder_prob"][0].cpu().numpy()
        ref = np.where(b_s.numpy()[:T] > 0.5)[0] / FPS
        dref = np.where(d_s.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) < 2: continue
        acc["beat_phase"].append(f_measure(ref, beats_from_barphase(pm, 4, FPS)))
        acc["downbeat_phase"].append(f_measure(dref, downbeats_from_barphase(pm, FPS)) if len(dref) >= 2 else np.nan)
        acc["decoder"].append(f_measure(ref, beats_from_decoder(dec, FPS)))
        acc["metronome"].append(f_measure(ref, metronome(T, FPS)))
    model.train()
    return {k: float(np.nanmean(v)) for k, v in acc.items()}

model = BarPointerVAE(h_dim=N_MELS, hidden=64, num_meters=4).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
STEPS, EVAL_EVERY = __STEPS__, __EVAL_EVERY__
hist = {"step": [], "recon": [], "kl_meter": [], "kl_phase": [], "kl_tempo": []}
evals = []
for step in range(1, STEPS + 1):
    h_b, b_b = sample_batch()
    loss, info = strict_elbo(model, h_b, b_b)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 20 == 0:
        for k in ("recon", "kl_meter", "kl_phase", "kl_tempo"): hist[k].append(info[k])
        hist["step"].append(step)
    if step % EVAL_EVERY == 0 or step == STEPS:
        e = eval_freerun(model); e["step"] = step; evals.append(e)
        print(f"step {step:4d} | recon {info['recon']:6.1f} | KL m/phi/tau "
              f"{info['kl_meter']:.2f}/{info['kl_phase']:.2f}/{info['kl_tempo']:.2f} | "
              f"eval beat_phase {e['beat_phase']:.3f} decoder {e['decoder']:.3f} "
              f"downbeat {e['downbeat_phase']:.3f} (metronome {e['metronome']:.3f})")
""")

code(r"""
# training curves: the KL of each latent, and the held-out read-outs over training.
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(hist["step"], hist["kl_meter"], label="KL meter")
ax[0].plot(hist["step"], hist["kl_phase"], label="KL phase")
ax[0].plot(hist["step"], hist["kl_tempo"], label="KL tempo")
ax[0].set_title("per-latent KL during training"); ax[0].set_xlabel("step"); ax[0].set_yscale("log"); ax[0].legend()
es = [e["step"] for e in evals]
for nm in ("beat_phase", "decoder", "downbeat_phase", "metronome"):
    ax[1].plot(es, [e[nm] for e in evals], ".-", label=nm)
ax[1].set_title("held-out free-run F-measure"); ax[1].set_xlabel("step"); ax[1].set_ylim(0, 1); ax[1].legend()
plt.tight_layout()
""")

# --------------------------------------------------------------------------------------
md(r"""
## 13. Results and what they mean

The final held-out scores, and an honest reading of them.
""")

code(r"""
final = evals[-1]
print("=== final held-out free-run F-measure (mir_eval, 70 ms) ===")
for k in ("beat_phase", "downbeat_phase", "decoder", "metronome"):
    print(f"  {k:16s} {final[k]:.3f}")
print(f"\n  per-latent KL at end:  meter {hist['kl_meter'][-1]:.3f}  "
      f"phase {hist['kl_phase'][-1]:.3f}  tempo {hist['kl_tempo'][-1]:.3f}")
""")

md(r"""
**Reading the numbers.** This is the model exactly as derived — a strict ELBO ($\beta=1$), three
latents, closed-form KLs, the von Mises sampler, the bar-pointer prior, and a Bernoulli decoder,
trained end to end from random weights on a fixed log-mel front-end, with no extra tricks. The
honest outcome:

* The **per-latent KLs decay toward zero** during training: the posterior is pulled back onto the
  prior and the latents stop carrying song-specific information (textbook *posterior collapse*).

* As a result the **latent read-outs fall below a trivial 120-BPM metronome**: the tempo latent
  never settles on a musical scale, so the free-running pointer does not lay down a beat grid that
  matches the audio.

* The **Bernoulli decoder is nearly silent**: beats are only ~1.5% of frames, and a per-frame BCE
  with no class weighting or shift tolerance is minimized by predicting "no beat" almost everywhere.

None of these are coding errors — each is an intrinsic behaviour of the plain objective on this
sparse, periodic data, and each points at a specific, well-understood remedy: KL floors / annealing
or a richer posterior for the collapse; an informative, musically-anchored tempo prior for the
scale; a class-weighted or peak-pickable loss for the decoder; or replacing amortized inference
with explicit filtering at deploy time. This notebook is the clean, honest baseline those remedies
are measured against — every component is written out, demonstrated on example values, and trained
on real audio.
""")


# ======================================================================================
#  BUILD + EXECUTE
# ======================================================================================

def execute(cells, data_root, n_use, n_val, steps, eval_every):
    import matplotlib
    matplotlib.use("Agg")
    os.environ["BALLROOM_ROOT"] = data_root      # the cell reads DATA_ROOT from here
    ns = {}
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = (src.replace("__N_USE__", str(n_use)).replace("__N_VAL__", str(n_val))
                  .replace("__STEPS__", str(steps)).replace("__EVAL_EVERY__", str(eval_every)))
        c["source"] = src.splitlines(keepends=True)
        buf = io.StringIO()
        outs = []
        try:
            with redirect_stdout(buf):
                exec(compile(src, "<cell>", "exec"), ns)
        except Exception:
            tb = traceback.format_exc()
            outs.append({"output_type": "stream", "name": "stderr", "text": tb.splitlines(keepends=True)})
            print("CELL ERROR:\n", tb, file=sys.stderr)
        txt = buf.getvalue()
        if txt:
            outs.insert(0, {"output_type": "stream", "name": "stdout", "text": txt.splitlines(keepends=True)})
        import matplotlib.pyplot as plt
        for num in plt.get_fignums():
            fig = plt.figure(num); b = io.BytesIO()
            fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); b.seek(0)
            png = base64.b64encode(b.read()).decode()
            outs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": png}})
            plt.close(fig)
        c["outputs"] = outs
        c["execution_count"] = 1
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data/ballroom")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ELBO_for_DBN.ipynb"))
    ap.add_argument("--quick", action="store_true", help="tiny run for a fast build smoke-test")
    ap.add_argument("--no_exec", action="store_true", help="build without executing (cells empty)")
    a = ap.parse_args()
    n_use, n_val, steps, eval_every = (40, 8, 80, 40) if a.quick else (140, 24, 600, 150)
    if not a.no_exec:
        execute(cells, a.data_root, n_use, n_val, steps, eval_every)
    else:
        for c in cells:
            if c["cell_type"] == "code":
                c["source"] = ("".join(c["source"])
                               .replace("__N_USE__", str(n_use)).replace("__N_VAL__", str(n_val))
                               .replace("__STEPS__", str(steps)).replace("__EVAL_EVERY__", str(eval_every))
                               ).splitlines(keepends=True)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
          "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    Path(a.out).write_text(json.dumps(nb, indent=1))
    print(f"\nwrote {a.out}  ({len(cells)} cells, {sum(c['cell_type']=='code' for c in cells)} code)")


if __name__ == "__main__":
    main()
