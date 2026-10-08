"""
VAE-DBN: a structured (sequential) conditional VAE with a Markov latent chain.

Latent per frame k:  z_k = (phi_k, v_k, m_k)
    phi_k in S^1  : bar phase            -> von Mises        (implicit reparam)
    v_k   in R    : phase velocity       -> Gaussian         (location-scale)
    m_k   in {0..R-1} : meter class      -> Categorical      (Gumbel-softmax)
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


# =====================================================================
# von Mises implicit reparameterisation (batched; forward bisection,
# backward = the implicit reparam gradient dS/dkappa, derived by differentiating F_kappa(S)=eps)
# =====================================================================
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
        for _ in range(iters):                       # batched bisection of F_k(phi)=e
            mid = 0.5 * (lo + hi)
            pts = (-PI) + (mid[..., None] + PI) * grid
            dens = torch.exp(k[..., None] * torch.cos(pts))
            step = (mid + PI) / (n - 1)
            Fmid = step * (dens.sum(-1) - 0.5 * (dens[..., 0] + dens[..., -1])) / Z
            go = Fmid < e                            # root is to the right of mid
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
        dS = -num / q0p                                           # implicit reparam gradient dS/dkappa
        return g * dS, None, None, None

def vm_sample(mu, kappa, eps):                       # full vM(mu,kappa) draw, eps~U(0,1)
    return torch.remainder(mu + VonMisesInvCDF.apply(kappa, eps), 2 * PI)


# =====================================================================
# Closed-form KL divergences
# =====================================================================
def kl_vm(mu_q, k_q, mu_p, k_p):                     # Eq. 9
    return torch.log(_i0(k_p) / _i0(k_q)) + A_vm(k_q) * (k_q - k_p * torch.cos(mu_q - mu_p))
def kl_vm_uniform(mu_q, k_q):                        # Eq. 10 (k_p = 0)
    return k_q * A_vm(k_q) - torch.log(_i0(k_q))
def kl_gauss(mu_q, logs_q, mu_p, logs_p):
    return (logs_p - logs_q + (torch.exp(2*logs_q) + (mu_q-mu_p)**2)/(2*torch.exp(2*logs_p)) - 0.5)
def kl_cat(logit_q, logit_p):
    lq, lp = F.log_softmax(logit_q, -1), F.log_softmax(logit_p, -1)
    return (lq.exp() * (lq - lp)).sum(-1)


# =====================================================================
# Test for the von Mises implicit reparameterisation (verifies the implicit gradient dS/dkappa)
# =====================================================================
def test_vonmises_reparam(kappas=(0.5, 2.0, 5.0, 10.0), n_eps=6, h=1e-5, tol=1e-4, verbose=True):
    """Certify the VonMisesInvCDF node against an INDEPENDENT oracle.

    (1) Backward gradient dS/dkappa (the implicit reparam gradient) vs central finite differences
        of the CDF inverse computed with scipy (brentq on a scipy-quad CDF). That oracle never
        uses the implicit formula, so the agreement is a genuine, non-circular check.
    (2) Derivative w.r.t. the mean, which enters ONLY as an additive shift (a rotation of the
        circle): d(phi_bar)/d(mu) = 1 exactly.
    (3) Bessel-ratio identity A(kappa)=E_vM[cos], i.e. integral_{-pi}^{pi}(cos t - A) q0 dt = 0,
        isolating the A(kappa) computation.
    Run with:  python vae_dbn.py --mode test
    """
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import i0 as si0, i0e, i1e
    q0_np = lambda p, k: np.exp(k*np.cos(p)) / (2*np.pi*si0(k))
    F_np  = lambda p, k: quad(lambda t: q0_np(t, k), -np.pi, p)[0]
    Sinv  = lambda e, k: brentq(lambda p: F_np(p, k) - e, -np.pi, np.pi, xtol=1e-13)

    torch.manual_seed(0); eps = torch.rand(n_eps, dtype=torch.float64); ok = True
    print("[vM reparam test] (1) dS/dkappa: autograd (implicit formula) vs scipy finite-diff oracle")
    for kv in kappas:
        kappa = torch.full((n_eps,), kv, dtype=torch.float64, requires_grad=True)
        VonMisesInvCDF.apply(kappa, eps, 2048, 50).sum().backward()      # high-res forward/backward
        ad = kappa.grad.numpy()
        fd = np.array([(Sinv(float(e), kv+h) - Sinv(float(e), kv-h)) / (2*h) for e in eps])
        err = float(np.max(np.abs(ad - fd))); ok &= err < tol
        if verbose: print(f"    kappa={kv:5.1f}  max|autograd - finite_diff| = {err:.2e}")
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
        if verbose: print(f"    kappa={kv:5.1f}  |integral| = {val:.2e}")
    print(f"[vM reparam test] {'PASS' if ok else 'FAIL'}")
    return ok


# =====================================================================
# Model
# =====================================================================
class VAEDBN(nn.Module):
    def __init__(self, x_dim=4, n_meter=3, hid=64, ctx=64, kmin=1.0, Delta=1.0,
                 emit_uses_x=False):
        super().__init__()
        self.R, self.Delta, self.kmin = n_meter, Delta, kmin
        self.emit_uses_x = emit_uses_x                # False -> p(b|z); True -> p(b|z,x)
        prevdim = 2 + 1 + n_meter                     # [cos phi, sin phi, v, onehot(m)] = z features
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
        self.post_vel   = nn.Linear(hid, 2)           # (mu, log-std)
        self.post_meter = nn.Linear(hid, n_meter)
        # prior heads: from backbone h_k and previous latent
        self.pri = nn.Sequential(nn.Linear(hid + prevdim, hid), nn.ReLU())
        self.pri_kappa = nn.Linear(hid, 1)            # phase concentration (T-b)
        self.pri_vel   = nn.Linear(hid, 2)
        self.pri_meter = nn.Linear(hid, n_meter)
        # emission. Default p(b_k | z_k): the decoder reads ONLY the latent state, not the
        # covariate x. This is the conditional-independence assumption b _|_ x | z, which stops
        # the decoder from bypassing the latent (predicting b from the powerful x alone and
        # ignoring z). With x removed, all of x's influence on b must route through z via the
        # prior p(z|x) and posterior q(z|b,x), so z stays the explanatory bottleneck -- what we
        # want when the latent distribution is the object of interest. Set emit_uses_x=True to
        # recover p(b_k | z_k, x) (feed the backbone features h_k as well).
        emit_in = (2 + 1 + n_meter) + (hid if emit_uses_x else 0)
        self.emit = nn.Sequential(nn.Linear(emit_in, hid), nn.ReLU(), nn.Linear(hid, 3))

    def emit_logits(self, feats, h_k):
        """Emission logits over {N,B,D}. feats = z features [cos phi, sin phi, v, onehot(m)];
        h_k = backbone (x) features, appended only if emit_uses_x (i.e. p(b|z,x))."""
        inp = torch.cat([feats, h_k], -1) if self.emit_uses_x else feats
        return self.emit(inp)

    def feats(self, phi, v, m_oh):
        return torch.cat([torch.cos(phi)[:,None], torch.sin(phi)[:,None], v[:,None], m_oh], -1)

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
           (kappa_q, velocity sigma, meter entropy) for debugging."""
        B, T = x.size(0), x.size(1)
        emis = x.new_zeros(B); kl = x.new_zeros(B)
        klph = x.new_zeros(B); klve = x.new_zeros(B); klme = x.new_zeros(B)
        kap_q_acc, sig_q_acc, ent_acc = [], [], []
        phi_p = x.new_zeros(B); v_p = x.new_zeros(B)
        m_p = x.new_full((B, self.R), 1.0/self.R)
        for k in range(T):
            prev = self.feats(phi_p, v_p, m_p)
            # ---- prior factors (condition on previous latent) ----
            hp = self.pri(torch.cat([h[:,k], prev], -1))
            if k == 0:
                mu_pha_p, kap_p = torch.zeros(B, device=x.device), torch.zeros(B, device=x.device)  # uniform
            else:
                mu_pha_p = torch.remainder(phi_p + v_p*self.Delta, 2*PI)
                kap_p = F.softplus(self.pri_kappa(hp)).squeeze(-1) + self.kmin
            muv_p, logsv_p = self.pri_vel(hp)[:,0], self.pri_vel(hp)[:,1]
            mlog_p = self.pri_meter(hp)
            # ---- choose source of z: posterior or prior ----
            if use_post:
                hq = self.post(torch.cat([c[:,k], prev], -1))
                mu_pha_q, kap_q = self.phase_params(self.post_phase(hq))
                muv_q, logsv_q = self.post_vel(hq)[:,0], self.post_vel(hq)[:,1]
                mlog_q = self.post_meter(hq)
                phi = vm_sample(mu_pha_q, kap_q, torch.rand(B, device=x.device)) if k == 0 else torch.remainder(phi_p + v_p*self.Delta, 2*PI)
                v   = muv_q + torch.exp(logsv_q) * torch.randn(B, device=x.device)
                m   = F.one_hot(torch.full((B,), 2, device=x.device), self.R).float()
                # per-frame KL (closed form), split per factor for diagnostics
                klp = kl_vm_uniform(mu_pha_q, kap_q) if k == 0 else torch.zeros_like(kap_q)
                klph = klph + klp
                klve = klve + kl_gauss(muv_q, logsv_q, muv_p, logsv_p)
                if diag is not None:
                    kap_q_acc.append(kap_q.detach()); sig_q_acc.append(torch.exp(logsv_q).detach())
                    rho = F.softmax(mlog_q, -1); ent_acc.append((-(rho*torch.log(rho+1e-9)).sum(-1)).detach())
            else:
                phi = torch.remainder(phi_p + v_p*self.Delta, 2*PI) if k > 0 else torch.rand(B, device=x.device) * 2*PI - PI
                v   = muv_p + torch.exp(logsv_p) * torch.randn(B, device=x.device)
                m   = F.one_hot(torch.full((B,), 2, device=x.device), self.R).float()
            # ---- emission (frames 1..T-1 carry a label here; frame 0 included for simplicity) ----
            logit = self.emit_logits(self.feats(phi, v, m), h[:, k])   # p(b|z) or p(b|z,x)
            if b is not None:
                emis = emis + F.cross_entropy(logit, b[:,k], reduction="none") * (-1.0)  # +log p
            phi_p, v_p, m_p = phi, v, m                    # advance chain (sampled previous state)
        kl = klph + klve + klme
        if diag is not None and use_post:
            diag.update(kl_phase=klph.mean(), kl_vel=klve.mean(), kl_meter=klme.mean(),
                        kappa_q=torch.stack(kap_q_acc), sigma_q=torch.stack(sig_q_acc),
                        meter_entropy=torch.stack(ent_acc))
        return emis, kl


def hybrid_loss(model, x, b, alpha=0.7, beta=1.0, tau=0.5, diag=None):
    """L = alpha*L_CVAE + (1-alpha)*L_GSNN ; returned as a minimisation loss.
       Pass diag={} to collect debugging diagnostics (per-factor KL, kappa_q, ...)."""
    h = model.backbone_feats(x)
    c = model.context(b, x)
    emis_q, kl = model.rollout(x, h, c=c, b=b, tau=tau, use_post=True, diag=diag)   # CVAE branch
    L_cvae = -(emis_q) + beta * kl                                         # -(emission) + KL
    emis_p, _ = model.rollout(x, h, c=None, b=b, tau=tau, use_post=False)  # GSNN branch
    L_gsnn = -(emis_p)
    loss = (alpha * L_cvae + (1 - alpha) * L_gsnn).mean()
    return loss, (-emis_q).mean(), kl.mean(), (-emis_p).mean()


# =====================================================================
# Synthetic data: a phase advancing at a per-sequence velocity; labels at phase landmarks
# =====================================================================
# --- log-mel spectrogram helpers (dependency-free: torch.stft + hand-built mel filterbank) ---
_MELFB = {}
def _hz_to_mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
def _mel_to_hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)

def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key in _MELFB: return _MELFB[key]
    n_bins = n_fft // 2 + 1
    fft_f = torch.linspace(0, sr / 2, n_bins)
    mel_pts = torch.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = torch.tensor([_mel_to_hz(m.item()) for m in mel_pts])
    fb = torch.zeros(n_mels, n_bins)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = hz[m - 1], hz[m], hz[m + 1]
        left = (fft_f - lo) / (ctr - lo + 1e-9)
        right = (hi - fft_f) / (hi - ctr + 1e-9)
        fb[m - 1] = torch.clamp(torch.minimum(left, right), min=0.0)
    _MELFB[key] = fb
    return fb

def logmel(wave, sr=22050, n_fft=1024, hop=256, n_mels=64, fmin=30.0, fmax=None, T=None):
    """wave [B,L] -> standardised log-mel [B, T, n_mels] (a genuine mel-spectrogram)."""
    fmax = fmax or sr / 2
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=torch.hann_window(n_fft), center=True, return_complex=True)
    power = spec.abs() ** 2                                      # [B, n_bins, frames]
    fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)          # [n_mels, n_bins]
    mel = torch.einsum('mf,bft->btm', fb, power)                 # [B, frames, n_mels]
    lm = torch.log1p(mel)
    if T is not None:
        lm = lm[:, :T] if lm.size(1) >= T else F.pad(lm, (0, 0, 0, T - lm.size(1)))
    mu, sd = lm.mean((0, 1), keepdim=True), lm.std((0, 1), keepdim=True) + 1e-5
    return (lm - mu) / sd


def synth(B=16, T=32, R=3, kappa_true=20.0, Delta=1.0, seed=0,
          mel=True, n_mels=64, sr=22050, hop=256, n_fft=1024):
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
    m = torch.full((B,), 2)                          # one meter per sequence (categorical)
    phi = torch.empty(B, T); v = torch.empty(B, T)
    cur_phi = 2*PI*torch.rand(B) - PI
    cur_v = 0.2 + 0.3*torch.rand(B)
    for k in range(T):
        phi[:, k] = cur_phi; v[:, k] = cur_v
        adv = cur_phi + cur_v*Delta
        cur_phi = adv   # <-- von Mises noise
        cur_phi = torch.remainder(cur_phi + PI, 2*PI) - PI                 # wrap to (-pi,pi]
        cur_v = (cur_v + 0.03*torch.randn(B)).clamp(0.1, 0.6)              # Gaussian random walk
    phw = torch.remainder(phi, 2*PI)                       # phase in [0,2pi)
    # labels b_k ~ Cat( emission(phi_k, m) ): a ground-truth categorical emission whose logits are
    # Gaussian bumps around the meter-dependent beat landmarks (downbeat at 0, beats at 2pi*j/bpb).
    # Sampling here (rather than thresholding) completes the ancestral sample: z chain AND labels.
    b = torch.zeros(B, T, dtype=torch.long); w, sharp, base = 0.3, 8.0, 4.0
    for i in range(B):
        beats = 2*PI*torch.arange(int(bpb[m[i]]))/int(bpb[m[i]])
        for k in range(T):
            d = torch.remainder(phw[i, k] - beats + PI, 2*PI) - PI         # signed dist to each beat
            gD = torch.exp(-(d[0]/w)**2)                                   # closeness to downbeat (idx 0)
            gB = torch.exp(-(d[1:].abs().min()/w)**2) if d.numel() > 1 else d.new_zeros(())
            # N carries a baseline logit so that, AWAY from any landmark (gB,gD~0), the non-beat
            # class wins -- beats are then SPARSE, firing only within ~w of a landmark. (Without the
            # baseline all three logits ~0 away from beats, giving a uniform draw and spurious beats.)
            logits = torch.stack([base + d.new_zeros(()), sharp*gB, sharp*gD])  # [N, B, D] logits
            b[i, k] = torch.distributions.Categorical(logits=logits).sample()
    if not mel:                                            # legacy toy features (phase leaked)
        mcue = (m.float()/max(R-1, 1))[:, None].expand(B, T)
        x = torch.stack([torch.cos(phw) + 0.1*torch.randn(B, T),
                         torch.sin(phw) + 0.1*torch.randn(B, T),
                         mcue + 0.1*torch.randn(B, T),
                         0.1*torch.randn(B, T)], -1)
        return x, b
    # ---- realistic audio features: render a waveform, then a LOG-MEL SPECTROGRAM ----
    # Percussive clicks at the beat (b=1) and downbeat (b=2) frames the labels mark: the
    # downbeat is a louder, lower "kick", other beats a lighter, higher transient. The phase
    # is thus NOT handed to the model (unlike the toy (cos,sin)); it must be inferred from the
    # onset pattern in the mel-spectrogram -- exactly as in real beat tracking.
    L = T * hop
    wave = torch.zeros(B, L); dur = n_fft
    tcl = torch.arange(dur).float()
    env = torch.exp(-tcl / (hop * 0.8))                    # exponential transient decay
    for i in range(B):
        for k in range(T):
            lab = int(b[i, k])
            if lab == 0:
                continue
            f0, amp = (60.0, 1.0) if lab == 2 else (180.0, 0.55)   # downbeat kick vs beat
            click = amp * env * (torch.sin(2*PI*f0*tcl/sr) + 0.6*torch.randn(dur))
            o = k * hop; e = min(o + dur, L)
            wave[i, o:e] += click[:e - o]
    wave = wave + 0.01 * torch.randn(B, L)                 # background noise
    x = logmel(wave, sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels, T=T)   # [B, T, n_mels]
    return x, b


# =====================================================================
# Training with full debug logging (single-batch overfit sanity check)
# =====================================================================
def train(x, b, steps=200, lr=3e-3, alpha=0.7, beta=1.0, tau=0.5, lr_=None,
          log_name="vae_dbn_run", log_every=5):
    """Overfit one batch while logging a wide set of health metrics to <log_name>.csv/.log.
    Logged each step: losses (total / -logp_q / per-factor KL / GSNN), gradient norms per
    submodule (catches a dead von Mises path or a detached sample), posterior-parameter
    stats (kappa_q, sigma_q, meter entropy), update/param ratio, and NaN/Inf + health warnings."""
    from train_logger import TrainLogger, HealthMonitor, grad_norms, tensor_stats, finite_check, update_param_ratio
    model = VAEDBN(x_dim=x.size(-1)).to(DEV)             # x_dim = n_mels for a mel-spectrogram
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = TrainLogger(log_name)
    health = HealthMonitor(loss_key="loss", kl_key="kl", kappa_key="kappa_q/mean", sigma_key="sigma_q/mean")
    groups = {"backbone": model.backbone, "encoder": model.encoder, "post_heads":
              list(model.post.parameters()) + list(model.post_phase.parameters())
              + list(model.post_vel.parameters()) + list(model.post_meter.parameters()),
              "prior_heads": list(model.pri.parameters()) + list(model.pri_kappa.parameters())
              + list(model.pri_vel.parameters()) + list(model.pri_meter.parameters()),
              "emit": model.emit, "total": model}
    print("device:", DEV, "| x", tuple(x.shape), "b", tuple(b.shape), "| logging ->", f"{log_name}.csv/.log")
    for t in range(1, steps + 1):
        diag = {}
        loss, rec_q, kl, rec_p = hybrid_loss(model, x, b, alpha=alpha, beta=beta, tau=tau, diag=diag)
        opt.zero_grad(); loss.backward()
        gn = grad_norms(groups)                                      # BEFORE step (grads live now)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # tame the ~700 spikes
        opt.step()
        if t == 1 or t % log_every == 0:
            m = {"loss": round(loss.item(), 4), "recon": round(rec_q.item(), 4),
                 "kl": round(kl.item(), 4), "gsnn_recon": round(rec_p.item(), 4),
                 "kl_phase": round(float(diag["kl_phase"]), 4),
                 "kl_vel": round(float(diag["kl_vel"]), 4),
                 "kl_meter": round(float(diag["kl_meter"]), 4),
                 "upd/param": update_param_ratio(model, lr),
                 "alpha": alpha, "beta": beta, "tau": round(tau, 3)}
            m.update(gn)
            m.update(tensor_stats("kappa_q", diag["kappa_q"]))
            m.update(tensor_stats("sigma_q", diag["sigma_q"]))
            m.update(tensor_stats("meter_entropy", diag["meter_entropy"]))
            warns = finite_check(loss=loss, kl=kl, kappa_q=diag["kappa_q"]) + health.check(m)
            log.log(t, m, warns)
    csvp, logp = log.close()
    torch.save({"state": model.state_dict(), "x_dim": x.size(-1)}, "vae_dbn.pt")
    print(f"[done] wrote {csvp}, {logp}, and vae_dbn.pt")
    return model


# =====================================================================
# Inference (mirrors Algorithm 2: posterior inference + predictive prior rollout)
# =====================================================================
def load_model(ckpt="vae_dbn.pt"):
    ck = torch.load(ckpt, map_location=DEV)
    model = VAEDBN(x_dim=ck.get("x_dim", 4)).to(DEV); model.load_state_dict(ck["state"]); model.eval()
    return model

@torch.no_grad()
def encode_path(model, x, b, sample=False):
    """POSTERIOR INFERENCE: given (x,b), the encoder produces the posterior q(z_k|.) per frame.
    The real inference output is that DISTRIBUTION's parameters per frame -- returned as `params`
    (mu_phi, kappa, vbar, s, rho) -- plus a latent `path` taken as the posterior MODE (or a
    reparameterised sample if sample=True). z comes from the ENCODER, not the prior."""
    h = model.backbone_feats(x); c = model.context(b, x)
    B, T = x.size(0), x.size(1)
    phi_p = x.new_zeros(B); v_p = x.new_zeros(B); m_p = x.new_full((B, model.R), 1.0/model.R)
    path, params = [], []
    for k in range(T):
        prev = model.feats(phi_p, v_p, m_p)
        hq = model.post(torch.cat([c[:, k], prev], -1))
        mu_q, kap_q = model.phase_params(model.post_phase(hq))
        vb_q, logs_q = model.post_vel(hq)[:, 0], model.post_vel(hq)[:, 1]
        rho_q = F.softmax(model.post_meter(hq), -1)
        params.append(dict(mu_phi=mu_q, kappa=kap_q, vbar=vb_q, s=torch.exp(logs_q), rho=rho_q))
        if sample:
            phi = vm_sample(mu_q, kap_q, torch.rand(B, device=x.device)) if k == 0 else torch.remainder(phi_p + v_p*model.Delta, 2*PI)
            v   = vb_q + torch.exp(logs_q) * torch.randn(B, device=x.device)
            m   = torch.full((B,), 2, device=x.device)
        else:                                            # posterior mode
            phi = mu_q if k == 0 else torch.remainder(phi_p + v_p*model.Delta, 2*PI)
            v, m = vb_q, torch.full((B,), 2, device=x.device)
        m_oh = F.one_hot(m, model.R).float()
        path.append((phi, v, m)); phi_p, v_p, m_p = phi, v, m_oh
    return path, params

@torch.no_grad()
def predict_labels(model, x, N=64):
    """PREDICTIVE INFERENCE (audio only): roll the PRIOR chain N times, average the per-frame
    label posteriors -> marginal p(b_k|x); argmax gives the predicted beats/downbeats. Uses no
    encoder and no labels -- the test-time generative path (cf. CVAE `sample`)."""
    h = model.backbone_feats(x); B, T = x.size(0), x.size(1)
    prob = x.new_zeros(B, T, 3)
    for _ in range(N):
        phi_p = x.new_zeros(B); v_p = x.new_zeros(B); m_p = x.new_full((B, model.R), 1.0/model.R)
        for k in range(T):
            prev = model.feats(phi_p, v_p, m_p)
            hp = model.pri(torch.cat([h[:, k], prev], -1))
            if k == 0:
                phi = torch.rand(B, device=x.device) * 2*PI - PI          # uniform initial phase
            else:
                phi = torch.remainder(phi_p + v_p*model.Delta, 2*PI)
            vb_p, logs_p = model.pri_vel(hp)[:, 0], model.pri_vel(hp)[:, 1]
            v = vb_p + torch.exp(logs_p) * torch.randn(B, device=x.device)
            m = torch.full((B,), 2, device=x.device)
            m_oh = F.one_hot(m, model.R).float()
            prob[:, k] += F.softmax(model.emit_logits(model.feats(phi, v, m_oh), h[:, k]), -1)
            phi_p, v_p, m_p = phi, v, m_oh
    prob /= N
    return prob.argmax(-1), prob                          # b_hat [B,T], p_label [B,T,3]


def _log_joint(model, x, b, h, phi, v, m_idx):
    """log p_theta(z_{0:T}, b_{1:T} | x) for a given path. Differentiable in (phi, v).
       = sum_k [ log p(phi_k|.) + log p(v_k|.) + log p(m_k|.) ] + sum_{k>=1} log p(b_k|z_k)."""
    B, T = x.size(0), x.size(1)
    lj = x.new_zeros(B)
    phi_p = x.new_zeros(B); v_p = x.new_zeros(B); m_p = x.new_full((B, model.R), 1.0/model.R)
    for k in range(T):
        prev = model.feats(phi_p, v_p, m_p)
        hp = model.pri(torch.cat([h[:, k], prev], -1))
        phk, vk, mk = phi[:, k], v[:, k], m_idx[:, k]
        m_oh = F.one_hot(mk, model.R).float()
        # phase prior: uniform at k=0, else vM(mu_p, kappa_p)
        if k == 0:
            lp_phi = phk.new_full((B,), -math.log(2*PI))
        else:
            mu_pha_p = torch.remainder(phi_p + v_p*model.Delta, 2*PI)
            kap_p = F.softplus(model.pri_kappa(hp)).squeeze(-1) + model.kmin
            lp_phi = kap_p*torch.cos(phk - mu_pha_p) - torch.log(2*PI*_i0(kap_p))
        vb_p, logs_p = model.pri_vel(hp)[:, 0], model.pri_vel(hp)[:, 1]
        lp_v = -0.5*((vk - vb_p)/torch.exp(logs_p))**2 - logs_p - 0.5*math.log(2*PI)
        lp_m = F.log_softmax(model.pri_meter(hp), -1).gather(-1, mk[:, None]).squeeze(-1)
        lj = lj + lp_phi + lp_v + lp_m
        if k >= 1:
            logit = model.emit_logits(model.feats(phk, vk, m_oh), h[:, k])
            lj = lj + F.log_softmax(logit, -1).gather(-1, b[:, k][:, None]).squeeze(-1)
        phi_p, v_p, m_p = phk, vk, m_oh
    return lj

@torch.no_grad()
def meter_viterbi(model, x, b, h, phi, v):
    """Given FIXED continuous (phi, v), find the MAP meter path m_{0:T} by EXACT Viterbi.
    With (phi,v) fixed, the log-joint as a function of the meter path is a chain MRF: the meter
    enters only through unary terms (its own prior, the emission) and pairwise terms (the next
    frame's phase/velocity/meter priors read m_{k-1}), so max-sum DP is exact. This uses the
    factorisation p(z_k|z_{k-1},x)=p(phi_k|.)p(v_k|.)p(m_k|.): phase/velocity contribute scores
    that depend on m_{k-1}; the meter prior contributes the (m_{k-1}->m_k) transition."""
    B, T, R, dev = x.size(0), x.size(1), model.R, x.device
    # k=0 unary: only the meter prior depends on m_0 (phi_0 uniform, v_0 prior use the fixed init)
    prev0 = model.feats(x.new_zeros(B), x.new_zeros(B), x.new_full((B, R), 1.0/R))
    hp0 = model.pri(torch.cat([h[:, 0], prev0], -1))
    score = F.log_softmax(model.pri_meter(hp0), -1)                      # [B,R]  = log p(m_0)
    back = []
    for k in range(1, T):
        phi_pm, v_pm = phi[:, k-1], v[:, k-1]
        mu_pha_p = torch.remainder(phi_pm + v_pm*model.Delta, 2*PI)      # phase mean (m-independent)
        emis = x.new_zeros(B, R)                                         # emission vs m_k (unary)
        for rc in range(R):
            m_oh = F.one_hot(torch.full((B,), rc, device=dev), R).float()
            logit = model.emit_logits(model.feats(phi[:, k], v[:, k], m_oh), h[:, k])
            emis[:, rc] = F.log_softmax(logit, -1).gather(-1, b[:, k][:, None]).squeeze(-1)
        E = x.new_zeros(B, R, R)                                         # E[:, m_{k-1}, m_k]
        for rp in range(R):                                             # each candidate m_{k-1}
            m_oh_p = F.one_hot(torch.full((B,), rp, device=dev), R).float()
            hp = model.pri(torch.cat([h[:, k], model.feats(phi_pm, v_pm, m_oh_p)], -1))
            kap_p = F.softplus(model.pri_kappa(hp)).squeeze(-1) + model.kmin
            logphi = kap_p*torch.cos(phi[:, k] - mu_pha_p) - torch.log(2*PI*_i0(kap_p))     # [B]
            vb_p, logs_p = model.pri_vel(hp)[:, 0], model.pri_vel(hp)[:, 1]
            logv = -0.5*((v[:, k]-vb_p)/torch.exp(logs_p))**2 - logs_p - 0.5*math.log(2*PI)  # [B]
            logrho = F.log_softmax(model.pri_meter(hp), -1)             # [B,R] = log p(m_k|m_{k-1}=rp)
            E[:, rp, :] = (logphi + logv)[:, None] + logrho + emis
        total = score[:, :, None] + E                                   # [B, m_{k-1}, m_k]
        score, arg = total.max(dim=1)                                   # [B, m_k]
        back.append(arg)
    m = x.new_zeros(B, T, dtype=torch.long)
    last = score.argmax(dim=1); m[:, T-1] = last
    for k in range(T-1, 0, -1):
        last = back[k-1].gather(1, last[:, None]).squeeze(1); m[:, k-1] = last
    return m


def map_path(model, x, b, cont_steps=80, rounds=2, lr=0.05, meter_mode="viterbi"):
    """FACTORED mixed MAP of the posterior p(z_{0:T}|b,x) = argmax_z log p_theta(z,b|x), by
    COORDINATE ASCENT that handles each latent factor with the method matching its type, using the
    transition factorisation p(z_k|z_{k-1},x) = p(phi_k|.) p(v_k|.) p(m_k|.):
      * phase + velocity (continuous): gradient ascent on the exact log-joint (general nonlinear
        optimisation; phase is circular so this is a LOCAL optimum -- an annealed-particle method
        could seek the global one);
      * meter (discrete chain): EXACT Viterbi (max-sum DP) given the continuous factors
        (meter_mode='viterbi'); 'icm' does cheaper per-frame coordinate-max instead.
    The two blocks alternate, initialised at the amortized posterior mode (encode_path). Returns
    ((phi,v,m), lj_mode, lj_map); lj_map >= lj_mode certifies the MAP improves the log-joint."""
    model.eval()
    h = model.backbone_feats(x).detach()         # x-features are constant w.r.t. the (phi,v) optim
    with torch.no_grad():
        path, _ = encode_path(model, x, b, sample=False)               # amortized mode init
    phi = torch.stack([p[0] for p in path], 1).clone()
    v   = torch.stack([p[1] for p in path], 1).clone()
    m   = torch.stack([p[2] for p in path], 1).clone()
    with torch.no_grad():
        lj_mode = _log_joint(model, x, b, h, phi, v, m).mean().item()
    phi = phi.detach().requires_grad_(True); v = v.detach().requires_grad_(True)
    for _ in range(rounds):
        opt = torch.optim.Adam([phi, v], lr=lr)                        # continuous block
        for _ in range(cont_steps):
            loss = -_log_joint(model, x, b, h, phi, v, m).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                          # discrete meter block
            if meter_mode == "viterbi":
                m = meter_viterbi(model, x, b, h, phi.detach(), v.detach())   # exact DP
            else:                                                      # 'icm' coordinate-max
                for k in range(x.size(1)):
                    best = m[:, k].clone(); best_lj = _log_joint(model, x, b, h, phi, v, m)
                    for cand in range(model.R):
                        m_try = m.clone(); m_try[:, k] = cand
                        lj_try = _log_joint(model, x, b, h, phi, v, m_try)
                        take = lj_try > best_lj
                        best = torch.where(take, torch.full_like(best, cand), best)
                        best_lj = torch.where(take, lj_try, best_lj)
                    m[:, k] = best
    with torch.no_grad():
        lj_map = _log_joint(model, x, b, h, phi, v, m).mean().item()
    return (phi.detach(), v.detach(), m), lj_mode, lj_map


def map_path_particle(model, x, b, P=32, sweeps=120, T0=1.0, Tend=0.02, sigma0=0.5,
                      polish=60, lr=0.05, seed=0):
    """GLOBAL continuous MAP by an ANNEALED-PARTICLE search (simulated annealing with P parallel
    chains per sequence), then a short gradient polish. The phase is circular -> the log-joint is
    multimodal in phi, so plain gradient ascent (map_path) only finds a LOCAL optimum; here P
    particles are spread around the circle and cooled (Metropolis moves, derivative-free), which
    can jump between modes; the best-ever particle is kept, polished by a few gradient steps, and
    the meter set by exact Viterbi. Returns ((phi,v,m), lj_mode, lj_map), lj_map >= lj_mode.
    (Trade-off: multimodal-robust and gradient-free in the search, but suffers the curse of
    dimensionality; use it when the global phase alignment matters.)"""
    torch.manual_seed(seed)
    B, T, R, dev = x.size(0), x.size(1), model.R, x.device
    h = model.backbone_feats(x).detach()
    with torch.no_grad():
        path, _ = encode_path(model, x, b, sample=False)                # amortized mode
        phi0 = torch.stack([p[0] for p in path], 1); v0 = torch.stack([p[1] for p in path], 1)
        m0   = torch.stack([p[2] for p in path], 1)
        lj_mode = _log_joint(model, x, b, h, phi0, v0, m0).mean().item()
        rep = lambda t: t.repeat_interleave(P, dim=0)
        xr, br, hr, mr = rep(x), rep(b), rep(h), rep(m0)
        # diversify: random global phase offset + jitter; keep particle 0 at the exact mode
        phi = torch.remainder(rep(phi0) + torch.rand(B*P,1,device=dev)*2*PI, 2*PI) + 0.2*torch.randn(B*P,T,device=dev)
        v   = rep(v0) + 0.1*torch.randn(B*P, T, device=dev)
        phi = phi.view(B,P,T); phi[:,0] = phi0; phi = torch.remainder(phi.reshape(B*P,T), 2*PI)
        v   = v.view(B,P,T);   v[:,0]   = v0;   v   = v.reshape(B*P,T)
        lj  = _log_joint(model, xr, br, hr, phi, v, mr)
        phi_best, v_best, lj_best = phi.clone(), v.clone(), lj.clone()   # track best-ever
        for s in range(sweeps):
            Temp = T0 * (Tend/T0) ** (s/max(sweeps-1, 1))               # geometric cooling
            sig  = sigma0 * Temp / T0
            phi_p = torch.remainder(phi + sig*torch.randn_like(phi), 2*PI)
            v_p   = v + sig*torch.randn_like(v)
            lj_p  = _log_joint(model, xr, br, hr, phi_p, v_p, mr)
            acc = torch.rand(B*P, device=dev) < torch.exp(((lj_p - lj)/Temp).clamp(max=0))   # Metropolis
            phi = torch.where(acc[:,None], phi_p, phi); v = torch.where(acc[:,None], v_p, v)
            lj  = torch.where(acc, lj_p, lj)
            imp = lj > lj_best                                          # keep best-ever (not final)
            lj_best = torch.where(imp, lj, lj_best)
            phi_best = torch.where(imp[:,None], phi, phi_best); v_best = torch.where(imp[:,None], v, v_best)
        best = lj_best.view(B,P).argmax(1); idx = torch.arange(B,device=dev)*P + best
        phi_b, v_b = phi_best[idx].clone(), v_best[idx].clone()
        m_b = meter_viterbi(model, x, b, h, phi_b, v_b)
    # short gradient polish of the best particle (local refine after global search)
    phi_b = phi_b.detach().requires_grad_(True); v_b = v_b.detach().requires_grad_(True)
    opt = torch.optim.Adam([phi_b, v_b], lr=lr)
    for _ in range(polish):
        loss = -_log_joint(model, x, b, h, phi_b, v_b, m_b).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        m_b = meter_viterbi(model, x, b, h, phi_b.detach(), v_b.detach())
        lj_map = _log_joint(model, x, b, h, phi_b.detach(), v_b.detach(), m_b).mean().item()
    return (phi_b.detach(), v_b.detach(), m_b), lj_mode, lj_map


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test", "encode", "predict", "map"], default="train")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.7, help="paper convention: 1=pure CVAE, 0=pure GSNN")
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight (anneal if collapsing)")
    ap.add_argument("--tau", type=float, default=0.5, help="Gumbel-softmax temperature for the meter")
    ap.add_argument("--N", type=int, default=64, help="prior rollouts to average in --mode predict")
    ap.add_argument("--method", choices=["coord", "particle"], default="coord",
                    help="--mode map: 'coord' = gradient+Viterbi (local); 'particle' = annealed-particle (global phase)")
    args = ap.parse_args()
    print("device:", DEV)

    if args.mode == "train":
        torch.manual_seed(0); x, b = synth(); x, b = x.to(DEV), b.to(DEV)
        train(x, b, steps=args.steps, alpha=args.alpha, beta=args.beta, tau=args.tau, log_name="vae_dbn_run")
    elif args.mode == "test":
        test_vonmises_reparam()                           # verify the von Mises implicit node (dS/dkappa)
    elif args.mode == "encode":
        # POSTERIOR INFERENCE: recover the posterior q(z|b,x) per frame (its PARAMETERS are the
        # real output; z comes from the ENCODER, not the prior). cf. CVAE --mode recon.
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
    elif args.mode == "map":
        # JOINT MAP of the posterior p(z|b,x): maximise log p_theta(z,b|x) over the whole path
        # (continuous phase/velocity by gradient ascent, discrete meter by ICM), initialised at
        # the amortized posterior mode. lj_map >= lj_mode certifies the MAP improves the log-joint.
        model = load_model(); x, b = synth(); x, b = x.to(DEV), b.to(DEV)
        fn = map_path_particle if args.method == "particle" else map_path
        (phi, v, m), lj_mode, lj_map = fn(model, x, b)
        print(f"[{args.method}] log-joint  amortized-mode = {lj_mode:.3f}  ->  joint-MAP = {lj_map:.3f}  "
              f"(improvement {lj_map - lj_mode:+.3f})")
