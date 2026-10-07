"""
VAE-DBN: a structured (sequential) conditional VAE with a Markov latent chain.

Latent per frame k:  z_k = (phi_k, m_k)
    phi_k in S^1  : bar phase         -> von Mises        (implicit reparam), advancing by omega(h_k) + delta(h_k, phi_{k-1})
    m_k   in {0..R-1} : meter class   -> Categorical      (Gumbel-softmax)
Covariate x (audio features) conditions every factor and is never inferred.
Observation b_{1:T} (beat labels) is the data; only the encoder sees it.

This is a compact but faithful implementation on SYNTHETIC sequences, enough to verify
the whole pipeline (von Mises reparam inside a chain, the per-frame closed-form KLs, and
the CVAE/GSNN hybrid objective). It overfits one batch, which certifies the gradients.

    python vae_dbn.py            # single-batch overfit sanity check
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F

PI = math.pi
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# von Mises implicit reparameterisation (batched; forward bisection,
# backward = the implicit reparam gradient dS/dkappa, derived by differentiating F_kappa(S)=eps)
# ======================================================================
def _i0(k): return torch.special.i0(k)
def A_vm(k): return torch.special.i1e(k) / torch.special.i0e(k)   # I1/I0, stable

class VonMisesInvCDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kappa, eps, n=256, iters=30):
        dt, dev = kappa.dtype, kappa.device
        shape = torch.broadcast_shapes(kappa.shape, eps.shape)
        k, e = kappa.expand(shape), eps.expand(shape)
        lo = torch.full(shape, -PI, dtype=dt, device=dev)
        hi = torch.full(shape,  PI, dtype=dt, device=dev)
        grid = torch.linspace(0, 1, n, dtype=dt, device=dev)
        Z = 2 * PI * _i0(k)
        for _ in range(iters):                          # batched bisection of F_k(phi)=e
            mid = 0.5 * (lo + hi)
            pts = (-PI) + (mid[..., None] + PI) * grid
            dens = torch.exp(k[..., None] * torch.cos(pts))
            step = (mid + PI) / (n - 1)
            Fmid = step * (dens.sum(-1) - 0.5 * (dens[..., 0] + dens[..., -1])) / Z
            go = Fmid < e                               # root is to the right of mid
            lo = torch.where(go, mid, lo)
            hi = torch.where(go, hi, mid)
        phi = 0.5 * (lo + hi)
        ctx.save_for_backward(k, phi); ctx.n = n
        return phi

    @staticmethod
    def backward(ctx, g):
        k, phi = ctx.saved_tensors; n = ctx.n
        dt, dev = k.dtype, k.device
        A = A_vm(k); grid = torch.linspace(0, 1, n, dtype=dt, device=dev)
        pts = (-PI) + (phi[..., None] + PI) * grid
        q0 = torch.exp(k[..., None] * torch.cos(pts)) / (2 * PI * _i0(k)[..., None])
        integ = (torch.cos(pts) - A[..., None]) * q0
        step = (phi + PI) / (n - 1)
        num = step * (integ.sum(-1) - 0.5 * (integ[..., 0] + integ[..., -1]))
        q0p = torch.exp(k * torch.cos(phi)) / (2 * PI * _i0(k))
        dS = -num / q0p                                         # implicit reparam gradient dS/dkappa
        return g * dS, None, None, None

def vm_sample(mu, kappa, eps):                        # full vM(mu,kappa) draw, eps~U(0,1)
    return torch.remainder(mu + VonMisesInvCDF.apply(kappa, eps), 2 * PI)


# ======================================================================
# Closed-form KL divergences
# ======================================================================
def kl_vm(mu_q, k_q, mu_p, k_p):                      # Eq. 9
    return torch.log(_i0(k_p) / _i0(k_q)) + A_vm(k_q) * (k_q - k_p * torch.cos(mu_q - mu_p))
def kl_vm_uniform(mu_q, k_q):                         # Eq. 10 (k_p = 0)
    return k_q * A_vm(k_q) - torch.log(_i0(k_q))
def kl_gauss(mu_q, logs_q, mu_p, logs_p):
    return (logs_p - logs_q + (torch.exp(2*logs_q) + (mu_q-mu_p)**2)/(2*torch.exp(2*logs_p)) - 0.5)
def kl_cat(logit_q, logit_p):
    lq, lp = F.log_softmax(logit_q, -1), F.log_softmax(logit_p, -1)
    return (lq.exp() * (lq - lp)).sum(-1)


def test_vonmises_reparam(kappas=(0.5, 2.0, 5.0, 10.0), n_eps=6, h=1e-5, tol=1e-4, verbose=True):
    """Certify the VonMisesInvCDF node against an INDEPENDENT oracle.

    (1) Backward gradient dS/dkappa (the implicit reparam gradient) vs central finite differences
        of the CDF inverse computed with scipy (brentq on a scipy-quad CDF). That oracle never
        uses the implicit formula, so the agreement is a genuine, non-circular check.
    (2) Derivative w.r.t. the mean, which enters ONLY as an additive shift (a rotation of the
        circle): d(phi_bar)/d(mu) = 1 exactly.
    (3) Bessel-ratio identity A(kappa)=E_vM[cos], i.e. integral_{-pi}^{pi}(cos t - A) q0 dt = 0,
        isolating the A(kappa) computation.
    Run with: python vae_dbn.py --mode test
    """
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import i0 as si0, i0e, i1e
    q0_np = lambda p, k: np.exp(k*np.cos(p)) / (2*np.pi*si0(k))
    F_np = lambda p, k: quad(lambda t: q0_np(t, k), -np.pi, p)[0]
    Sinv = lambda e, k: brentq(lambda p: F_np(p, k) - e, -np.pi, np.pi, xtol=1e-13)

    torch.manual_seed(0); eps = torch.rand(n_eps, dtype=torch.float64); ok = True
    print("[vM reparam test] (1) dS/dkappa: autograd (implicit formula) vs scipy finite-diff oracle")
    for kv in kappas:
        kappa = torch.full((n_eps,), kv, dtype=torch.float64, requires_grad=True)
        VonMisesInvCDF.apply(kappa, eps, 2048, 50).sum().backward()       # high-res forward/backward
        ad = kappa.grad.numpy()
        fd = np.array([(Sinv(float(e), kv+h) - Sinv(float(e), kv-h)) / (2*h) for e in eps])
        err = float(np.max(np.abs(ad - fd))); ok &= err < tol
        if verbose: print(f"    kappa={kv:5.1f} max|autograd - finite_diff| = {err:.2e}")
    # (2) derivative w.r.t. the mean (mu enters only as an additive shift -> should be exactly 1)
    mu = torch.zeros(n_eps, dtype=torch.float64, requires_grad=True)
    vm_sample(mu, torch.full((n_eps,), 2.0, dtype=torch.float64), eps).sum().backward()
    e2 = float((mu.grad - 1.0).abs().max()); ok &= e2 < 1e-6
    print(f"[vM reparam test] (2) d(phi_bar)/d(mu): max|.-1| = {e2:.2e}")
    # (3) A(kappa) consistency
    print("[vM reparam test] (3) A(kappa) identity: |integral (cos - A) q0| should be ~0")
    for kv in kappas:
        A = i1e(kv)/i0e(kv)
        val = abs(quad(lambda t: (np.cos(t) - A)*q0_np(t, kv), -np.pi, np.pi)[0]); ok &= val < 1e-9
        if verbose: print(f"    kappa={kv:5.1f} |integral| = {val:.2e}")
    print(f"[vM reparam test] {'PASS' if ok else 'FAIL'}")
    return ok


# ======================================================================
# Model
# ======================================================================
class VAEDBN(nn.Module):
    def __init__(self, x_dim=4, n_meter=3, hid=64, ctx=64, kmin=1.0, Delta=1.0):
        super().__init__()
        self.R, self.Delta, self.kmin = n_meter, Delta, kmin
        prevdim = 2 + n_meter                         # [cos phi, sin phi, onehot(m)]
        # generative backbone over x (bidirectional over the covariate is allowed)
        self.backbone = nn.GRU(x_dim, hid, batch_first=True, bidirectional=True)
        self.hb = nn.Linear(2*hid, hid)
        # inference encoder over [embed(b), x]  (bidirectional context)
        self.b_emb = nn.Embedding(3, 8)
        self.encoder = nn.GRU(8 + x_dim, ctx, batch_first=True, bidirectional=True)
        self.hc = nn.Linear(2*ctx, ctx)
        # posterior heads: from context c_k and previous latent
        self.post = nn.Sequential(nn.Linear(ctx + prevdim, hid), nn.ReLU())
        self.post_phase = nn.Linear(hid, 3)           # (a1,a2,u)
        self.post_meter = nn.Linear(hid, n_meter)
        # prior heads: from backbone h_k and previous latent
        self.pri = nn.Sequential(nn.Linear(hid + prevdim, hid), nn.ReLU())
        self.pri_kappa = nn.Linear(hid, 1)            # phase concentration (T-b)
        self.pri_phase0 = nn.Linear(hid, 3)
        self.pri_delta = nn.Linear(hid, 1)
        self.pri_omega = nn.Linear(hid, 1)
        self.pri_meter = nn.Linear(hid, n_meter)
        # emission p(b_k | z_k)
        self.emit = nn.Sequential(nn.Linear(2 + n_meter, hid), nn.ReLU(),
                                  nn.Linear(hid, 3))

    def feats(self, phi, m_oh):
        return torch.cat([torch.cos(phi)[:,None], torch.sin(phi)[:,None], m_oh], -1)

    def omega(self, hp):
        """Phase advance per frame as a deterministic function of the audio."""
        return F.softplus(self.pri_omega(hp)).squeeze(-1)

    def backbone_feats(self, x):
        h,_ = self.backbone(x); return torch.tanh(self.hb(h))     # [B,T,hid]
    def context(self, b, x):
        e = torch.cat([self.b_emb(b), x], -1); c,_ = self.encoder(e); return torch.tanh(self.hc(c))

    def phase_params(self, raw):
        mu = torch.atan2(raw[:,1], raw[:,0]); kappa = F.softplus(raw[:,2]) + self.kmin
        return mu, kappa

    def rollout(self, x, h, c=None, b=None, tau=0.5, use_post=True, diag=None):
        """Roll the chain. use_post=True -> CVAE branch (z~posterior, accumulate KL).
           use_post=False -> GSNN branch (z~prior, emission only).
           If diag is a dict, fill it with per-factor KLs and posterior-parameter stats
           (kappa_q, meter entropy) for debugging."""
        B, T = x.size(0), x.size(1)
        emis = x.new_zeros(B); kl = x.new_zeros(B)
        klph = x.new_zeros(B); klme = x.new_zeros(B)
        kap_q_acc, ent_acc = [], []
        phi_p = x.new_zeros(B)
        m_p = x.new_full((B, self.R), 1.0/self.R)
        for k in range(T):
            prev = self.feats(phi_p, m_p)
            # ---- prior factors (condition on previous latent) ----
            hp = self.pri(torch.cat([h[:,k], prev], -1))
            if k == 0:
                mu_pha_p, kap_p = self.phase_params(self.pri_phase0(hp))
            else:
                mu_pha_p = torch.remainder(phi_p + self.omega(hp)*self.Delta + self.pri_delta(hp).squeeze(-1), 2*PI)
                kap_p = F.softplus(self.pri_kappa(hp)).squeeze(-1) + self.kmin
            mlog_p = self.pri_meter(hp)
            # ---- choose source of z: posterior or prior ----
            if use_post:
                hq = self.post(torch.cat([c[:,k], prev], -1))
                mu_pha_q, kap_q = self.phase_params(self.post_phase(hq))
                mlog_q = self.post_meter(hq)
                phi = vm_sample(mu_pha_q, kap_q, torch.rand(B, device=x.device))
                m   = F.gumbel_softmax(mlog_q, tau=tau, hard=False)
                # per-frame KL (closed form), split per factor for diagnostics
                klp = kl_vm(mu_pha_q, kap_q, mu_pha_p, kap_p)
                klph = klph + klp
                klme = klme + kl_cat(mlog_q, mlog_p)
                if diag is not None:
                    kap_q_acc.append(kap_q.detach())
                    rho = F.softmax(mlog_q, -1); ent_acc.append((-(rho*torch.log(rho+1e-9)).sum(-1)).detach())
            else:
                phi = vm_sample(mu_pha_p, kap_p, torch.rand(B, device=x.device))
                m   = F.gumbel_softmax(mlog_p, tau=tau, hard=False)
            # ---- emission (frames 1..T-1 carry a label here; frame 0 included for simplicity) ----
            logit = self.emit(self.feats(phi, m))
            if b is not None:
                emis = emis + F.cross_entropy(logit, b[:,k], reduction="none") * (-1.0)  # +log p
            phi_p, m_p = phi, m                              # advance chain (sampled previous state)
        kl = klph + klme
        if diag is not None and use_post:
            diag.update(kl_phase=klph.mean(), kl_meter=klme.mean(),
                        kappa_q=torch.stack(kap_q_acc), meter_entropy=torch.stack(ent_acc))
        return emis, kl


def hybrid_loss(model, x, b, alpha=0.7, beta=1.0, tau=0.5, diag=None):
    """L = alpha*L_CVAE + (1-alpha)*L_GSNN ; returned as a minimisation loss.
       Pass diag={} to collect debugging diagnostics (per-factor KL, kappa_q, ...)."""
    h = model.backbone_feats(x)
    c = model.context(b, x)
    emis_q, kl = model.rollout(x, h, c=c, b=b, tau=tau, use_post=True, diag=diag)   # CVAE branch
    L_cvae = -(emis_q) + beta * kl                                                    # -(emission) + KL
    emis_p, _ = model.rollout(x, h, c=None, b=b, tau=tau, use_post=False)             # GSNN branch
    L_gsnn = -(emis_p)
    loss = (alpha * L_cvae + (1 - alpha) * L_gsnn).mean()
    return loss, (-emis_q).mean(), kl.mean(), (-emis_p).mean()


# ======================================================================
# Synthetic data: a phase advancing at a per-sequence velocity; labels at phase landmarks
# ======================================================================
def synth(B=16, T=32, R=3, kappa_true=20.0, Delta=1.0, seed=0, return_latents=False):
    """Ancestral sample FROM the model's own generative process, so the data actually
    exercises the stochastic factors:
      * meter   m ~ Categorical over R classes (sets beats-per-bar) -- a real categorical latent;
      * velocity v: Gaussian random walk (slowly varying tempo), reparam of the Gaussian factor;
      * phase   phi_k ~ vM(phi_{k-1} + v_{k-1}*Delta, kappa_true) -- a VON MISES draw around the
                velocity-driven advance, NOT a deterministic cumsum;
      * labels  b_k ~ Cat(.) sampled from a ground-truth emission peaked at the meter-dependent
                beat landmarks (downbeat at phase 0) -- so the labels, too, are ancestral-sampled.
    The audio features x carry (cos phi, sin phi), a weak meter cue, and noise."""
    torch.manual_seed(seed)
    bpb = torch.tensor([2, 3, 4])[:R]                      # beats per bar for meters 0,1,2
    m = torch.randint(0, R, (B,))                          # one meter per sequence (categorical)
    phi = torch.empty(B, T); v = torch.empty(B, T)
    cur_phi = 2*PI*torch.rand(B) - PI
    cur_v = 0.2 + 0.3*torch.rand(B)
    for k in range(T):
        phi[:, k] = cur_phi; v[:, k] = cur_v
        adv = cur_phi + cur_v*Delta
        cur_phi = torch.distributions.VonMises(adv, kappa_true).sample()   # <-- von Mises noise
        cur_phi = torch.remainder(cur_phi + PI, 2*PI) - PI                 # wrap to (-pi,pi]
        cur_v = (cur_v + 0.03*torch.randn(B)).clamp(0.1, 0.6)               # Gaussian random walk
    phw = torch.remainder(phi, 2*PI)                                      # phase in [0,2pi)
    # labels b_k ~ Cat( emission(phi_k, m) ): a ground-truth categorical emission whose logits are
    # Gaussian bumps around the meter-dependent beat landmarks (downbeat at 0, beats at 2pi*j/bpb).
    # Sampling here (rather than thresholding) completes the ancestral sample: z chain AND labels.
    b = torch.zeros(B, T, dtype=torch.long); w, sharp = 0.3, 8.0
    for i in range(B):
        beats = 2*PI*torch.arange(int(bpb[m[i]]))/int(bpb[m[i]])
        for k in range(T):
            d = torch.remainder(phw[i, k] - beats + PI, 2*PI) - PI         # signed dist to each beat
            gD = torch.exp(-(d[0]/w)**2)                                    # closeness to downbeat (idx 0)
            gB = torch.exp(-(d[1:].abs().min()/w)**2) if d.numel() > 1 else d.new_zeros(())
            logits = torch.stack([d.new_zeros(()), sharp*gB, sharp*gD])     # [N, B, D] emission logits
            b[i, k] = torch.distributions.Categorical(logits=logits).sample()
    mcue = (m.float()/max(R-1, 1))[:, None].expand(B, T)
    x = torch.stack([torch.cos(phw) + 0.1*torch.randn(B, T),
                     torch.sin(phw) + 0.1*torch.randn(B, T),
                     mcue + 0.1*torch.randn(B, T),
                     0.1*torch.randn(B, T)], -1)
    if return_latents:
        return x, b, (phi, v, m)
    return x, b


# ======================================================================
# Training with full debug logging (single-batch overfit sanity check)
# ======================================================================
def train(x, b, steps=200, lr=3e-3, alpha=0.7, beta=1.0, tau=0.5, lr_=None,
          log_name="vae_dbn_run", log_every=5):
    """Overfit one batch while logging a wide set of health metrics to <log_name>.csv/.log.
    Logged each step: losses (total / -logp_q / per-factor KL / GSNN), gradient norms per
    submodule (catches a dead von Mises path or a detached sample), posterior-parameter
    stats (kappa_q, meter entropy), update/param ratio, and NaN/Inf + health warnings."""
    from train_logger import TrainLogger, HealthMonitor, grad_norms, tensor_stats, finite_check, update_param_ratio
    model = VAEDBN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = TrainLogger(log_name)
    health = HealthMonitor(loss_key="loss", kl_key="kl", kappa_key="kappa_q/mean")
    groups = {"backbone": model.backbone, "encoder": model.encoder, "post_heads":
              list(model.post.parameters()) + list(model.post_phase.parameters())
              + list(model.post_meter.parameters()),
              "prior_heads": list(model.pri.parameters()) + list(model.pri_kappa.parameters())
              + list(model.pri_phase0.parameters()) + list(model.pri_omega.parameters()) + list(model.pri_delta.parameters()) + list(model.pri_meter.parameters()),
              "emit": model.emit, "total": model}
    print("device:", DEV, "| x", tuple(x.shape), "b", tuple(b.shape), "| logging ->", f"{log_name}.csv/.log")
    for t in range(1, steps + 1):
        diag = {}
        loss, rec_q, kl, rec_p = hybrid_loss(model, x, b, alpha=alpha, beta=beta, tau=tau, diag=diag)
        opt.zero_grad(); loss.backward()
        gn = grad_norms(groups)                                    # BEFORE step (grads live now)
        opt.step()
        if t == 1 or t % log_every == 0:
            m = {"loss": round(loss.item(), 4), "recon": round(rec_q.item(), 4),
                 "kl": round(kl.item(), 4), "gsnn_recon": round(rec_p.item(), 4),
                 "kl_phase": round(float(diag["kl_phase"]), 4),
                 "kl_meter": round(float(diag["kl_meter"]), 4),
                 "upd/param": update_param_ratio(model, lr),
                 "alpha": alpha, "beta": beta, "tau": round(tau, 3)}
            m.update(gn)
            m.update(tensor_stats("kappa_q", diag["kappa_q"]))
            m.update(tensor_stats("meter_entropy", diag["meter_entropy"]))
            warns = finite_check(loss=loss, kl=kl, kappa_q=diag["kappa_q"]) + health.check(m)
            log.log(t, m, warns)
    csvp, logp = log.close()
    torch.save({"state": model.state_dict()}, "vae_dbn.pt")
    print(f"[done] wrote {csvp}, {logp}, and vae_dbn.pt")
    return model


# ======================================================================
# Inference (mirrors Algorithm 2: posterior inference + predictive prior rollout)
# ======================================================================
def load_model(ckpt="vae_dbn.pt"):
    ck = torch.load(ckpt, map_location=DEV)
    model = VAEDBN().to(DEV); model.load_state_dict(ck["state"]); model.eval()
    return model

@torch.no_grad()
def encode_path(model, x, b, sample=False):
    """POSTERIOR INFERENCE: given (x,b), the encoder produces the posterior q(z_k|.) per frame.
    The real inference output is that DISTRIBUTION's parameters per frame -- returned as `params`
    (mu_phi, kappa, rho) -- plus a latent `path` taken as the posterior MODE (or a
    reparameterised sample if sample=True). z comes from the ENCODER, not the prior."""
    h = model.backbone_feats(x); c = model.context(b, x)
    B, T = x.size(0), x.size(1)
    phi_p = x.new_zeros(B); m_p = x.new_full((B, model.R), 1.0/model.R)
    path, params = [], []
    for k in range(T):
        prev = model.feats(phi_p, m_p)
        hq = model.post(torch.cat([c[:, k], prev], -1))
        mu_q, kap_q = model.phase_params(model.post_phase(hq))
        rho_q = F.softmax(model.post_meter(hq), -1)
        params.append(dict(mu_phi=mu_q, kappa=kap_q, rho=rho_q))
        if sample:
            phi = vm_sample(mu_q, kap_q, torch.rand(B, device=x.device))
            m   = torch.multinomial(rho_q, 1).squeeze(-1)
        else:                                          # posterior mode
            phi, m = mu_q, rho_q.argmax(-1)
        m_oh = F.one_hot(m, model.R).float()
        path.append((phi, m)); phi_p, m_p = phi, m_oh
    return path, params

@torch.no_grad()
def predict_labels(model, x, N=64):
    """PREDICTIVE INFERENCE (audio only): roll the PRIOR chain N times, average the per-frame
    label posteriors -> marginal p(b_k|x); argmax gives the predicted beats/downbeats. Uses no
    encoder and no labels -- the test-time generative path (cf. CVAE `sample`)."""
    h = model.backbone_feats(x); B, T = x.size(0), x.size(1)
    prob = x.new_zeros(B, T, 3)
    for _ in range(N):
        phi_p = x.new_zeros(B); m_p = x.new_full((B, model.R), 1.0/model.R)
        for k in range(T):
            prev = model.feats(phi_p, m_p)
            hp = model.pri(torch.cat([h[:, k], prev], -1))
            if k == 0:
                mu_p, kap_p = model.phase_params(model.pri_phase0(hp))
            else:
                mu_p = torch.remainder(phi_p + model.omega(hp)*model.Delta + model.pri_delta(hp).squeeze(-1), 2*PI)
                kap_p = F.softplus(model.pri_kappa(hp)).squeeze(-1) + model.kmin
            phi = vm_sample(mu_p, kap_p, torch.rand(B, device=x.device))
            m = torch.multinomial(F.softmax(model.pri_meter(hp), -1), 1).squeeze(-1)
            m_oh = F.one_hot(m, model.R).float()
            prob[:, k] += F.softmax(model.emit(model.feats(phi, m_oh)), -1)
            phi_p, m_p = phi, m_oh
    prob /= N
    return prob.argmax(-1), prob                          # b_hat [B,T], p_label [B,T,3]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test", "encode", "predict"], default="train")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.7, help="paper convention: 1=pure CVAE, 0=pure GSNN")
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight (anneal if collapsing)")
    ap.add_argument("--tau", type=float, default=0.5, help="Gumbel-softmax temperature for the meter")
    ap.add_argument("--N", type=int, default=64, help="prior rollouts to average in --mode predict")
    args = ap.parse_args()
    print("device:", DEV)

    if args.mode == "train":
        torch.manual_seed(0); x, b = synth(); x, b = x.to(DEV), b.to(DEV)
        train(x, b, steps=args.steps, alpha=args.alpha, beta=args.beta, tau=args.tau,
              log_name="vae_dbn_run")
    elif args.mode == "test":
        test_vonmises_reparam()                                 # verify the von Mises implicit node (dS/dkappa)
    elif args.mode == "encode":
        # POSTERIOR INFERENCE: recover the posterior q(z|b,x) per frame (its PARAMETERS are the
        # real output; z comes from the ENCODER, not the prior. cf. CVAE --mode recon.
        model = load_model(); x, b = synth(); x, b = x.to(DEV), b.to(DEV)
        path, params = encode_path(model, x, b)
        print(f"encoded {x.size(0)} sequences x {x.size(1)} frames; "
              f"frame-0 posterior kappa_q mean = {params[0]['kappa'].mean():.3f}")
    elif args.mode == "predict":
        # PREDICTIVE INFERENCE (audio only): roll the PRIOR chain and decode label predictions --
        # no encoder, no labels. cf. CVAE --mode sample. (synth b is used only to score accuracy.)
        model = load_model(); x, b = synth(); x = x.to(DEV)
        b_hat, p_label = predict_labels(model, x, N=args.N)
        acc = (b_hat.cpu() == b).float().mean().item()
        print(f"predicted labels for {x.size(0)} sequences; frame accuracy vs ground truth = {acc:.3f}")
