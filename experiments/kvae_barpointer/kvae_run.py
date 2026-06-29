"""KVAE-bar-pointer M1: replace the AMORTIZED latent posterior with an EXACT differentiable filter.

This is the pivot after the amortized-encoder wall (experiments/diagram_arch/RESULTS.md): the deploy
geometric pointer never audio-locked because the per-frame KL gradient drags q(z|h) to the prior.
Kalman-VAE (Fraccaro et al. 2017) computes q(z|a) EXACTLY with a Kalman filter/smoother -- it cannot
be dragged to the prior by KL gradients (DEEP_RESEARCH_2 Finding G).

We REUSE the PyTorch port's StateSpaceModel (filter+smoother+mixture-of-K transitions) verbatim
(third_party/kalman-vae, cross-verified vs the official TF in third_party/kvae), and only swap the
conv VAE for an MLP VAE over our [512] Beat-This activations, plus a beat/downbeat read-out on the
filtered latent state.

M1 question: does the EXACT differentiable filter escape the wall?  PASS = beat-F competitive AND the
leak controls (shuffled/zero audio) COLLAPSE (so it genuinely uses audio, not a generic grid).

  enc:  h_t [512] --MLP--> a_t ~ N           (pseudo-observation, the "what")
  SSM:  a_{1:T} --Kalman filter/smooth--> z  (LGSSM dynamics, the "where"; EXACT posterior)
  dec:  a_t --MLP--> reconstruct h_t          (KVAE reconstruction)
  head: z_t --MLP--> Bernoulli(beat, downbeat)
  deploy: enc(h)->a (mean) -> Kalman FILTER on audio -> filtered z -> head -> peak-pick beats
"""
import sys, glob, math, random, importlib.util, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

ROOT = "/home/sogang/jaehoon/CHART"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/third_party/kalman-vae")   # so `import kvae` -> the PyTorch port
from kvae.state_space_model import StateSpaceModel
from kvae.sample_control import SampleControl

# reuse our read-out + metrics from the diagram experiments
s = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(s); s.loader.exec_module(da)
peaks, fmeas = da.peaks, da.fmeas
DEV = da.DEV; FPS = 86.1328125


# ----------------------------------------------------------------------------- MLP VAE (swap for conv)
class MLPEncoder(nn.Module):
    """h_t [h_dim] -> a_t ~ N(mu, sigma)  (the KVAE pseudo-observation)."""
    def __init__(self, h_dim, a_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.fc_mean = nn.Linear(hidden, a_dim)
        self.fc_std = nn.Linear(hidden, a_dim)

    def forward(self, x):
        h = self.net(x)
        return D.Normal(self.fc_mean(h), F.softplus(self.fc_std(h)) + 1e-4)


class MLPDecoder(nn.Module):
    """a_t [a_dim] -> reconstruct h_t [h_dim]  (Gaussian, fixed unit scale)."""
    def __init__(self, a_dim, h_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(a_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, h_dim))

    def forward(self, a):
        return D.Normal(self.net(a), torch.ones_like(self.net(a)))


class KVAEBarPointer(nn.Module):
    def __init__(self, h_dim=512, a_dim=8, z_dim=8, K=5, hidden=256):
        super().__init__()
        self.encoder = MLPEncoder(h_dim, a_dim, hidden)
        self.decoder = MLPDecoder(a_dim, h_dim, hidden)
        self.ssm = StateSpaceModel(a_dim=a_dim, z_dim=z_dim, K=K,
                                   dynamics_parameter_network="lstm",
                                   hidden_dim=64, num_layers=1,
                                   learn_noise_covariance=True, init_noise_scale=1.0)
        self.head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 2))  # beat, downbeat
        self.a_dim, self.z_dim = a_dim, z_dim


# ----------------------------------------------------------------------------- KVAE ELBO (faithful)
def kvae_elbo(model, h, sc, recon_w=0.3, reg_w=1.0, kal_w=1.0):
    """h: (T,B,h_dim) sequence-first. Returns (elbo_to_maximize, smoothed_z_sample, info)."""
    T, B, hd = h.shape
    a_distrib = model.encoder(h.reshape(-1, hd))
    a = a_distrib.rsample().view(T, B, model.a_dim)

    # reconstruction  ln p(h|a)   and regularization  -ln q(a|h)
    recon = model.decoder(a.view(-1, model.a_dim)).log_prob(h.reshape(-1, hd)).view(T, B, hd).sum(-1).mean()
    reg = a_distrib.log_prob(a.view(-1, model.a_dim)).view(T, B, model.a_dim).sum(-1).mean()

    fm, fc, fnm, fnc, matA, matC, _, _ = model.ssm.kalman_filter(a, sample_control=sc)
    sm_m, sm_c, zs, _ = model.ssm.kalman_smooth(a, fm, fc, fnm, fnc, matA, matC, sample_control=sc)

    zs_distrib = D.MultivariateNormal(sm_m.view(T, B, model.z_dim), sm_c.view(T, B, model.z_dim, model.z_dim))
    z = zs_distrib.rsample()

    # ln p(a|z)
    kal_obs = D.MultivariateNormal((matC[:-1] @ z.unsqueeze(-1)).view(-1, model.a_dim),
                                   model.ssm.mat_R).log_prob(a.view(-1, model.a_dim)).view(T, B).mean()
    # ln p(z) = ln p(z_0) + sum_t ln p(z_t|z_{t-1})
    prior_means = torch.cat([model.ssm.initial_state_mean.repeat(1, B, 1),
                             (matA[1:-1] @ z[:-1].unsqueeze(-1)).squeeze(-1)])
    prior_covs = torch.cat([model.ssm.initial_state_covariance.repeat(1, B, 1, 1),
                            model.ssm.mat_Q.repeat(T - 1, B, 1, 1)])
    kal_trans = D.MultivariateNormal(prior_means.view(T, B, model.z_dim),
                                     prior_covs.view(T, B, model.z_dim, model.z_dim)).log_prob(z).mean()
    # ln q(z|a)  (posterior entropy term)
    kal_post = zs_distrib.log_prob(z).mean()

    elbo = recon_w * recon - reg_w * reg + kal_w * (kal_obs + kal_trans - kal_post)
    return elbo, z, {"recon": float(recon), "reg": float(reg),
                     "kal_obs": float(kal_obs), "kal_trans": float(kal_trans), "kal_post": float(kal_post)}


# ----------------------------------------------------------------------------- data
def load(cd, n, seed):
    fs = sorted(glob.glob(f"{cd}/*.pt")); random.Random(seed).shuffle(fs); out = []
    for f in fs[:n]:
        d = torch.load(f, map_location="cpu"); hh = d["activations"].float()
        if hh.shape[0] < 400 or d["beat_targets"].sum() < 8: continue
        out.append((hh, d["beat_targets"].float(), d["downbeat_targets"].float()))
    return out


def batch(songs, frames, bs):
    hs, bb, dd = [], [], []
    while len(hs) < bs:
        hh, b, db = random.choice(songs)
        if hh.shape[0] <= frames: continue
        s0 = random.randint(0, hh.shape[0] - frames); bcrop = b[s0:s0 + frames]
        if bcrop.sum() < 2: continue
        hs.append(hh[s0:s0 + frames]); bb.append(bcrop); dd.append(db[s0:s0 + frames])
    # sequence-first (T,B,*) for the SSM
    H = torch.stack(hs).transpose(0, 1).to(DEV)
    Bt = torch.stack(bb).transpose(0, 1).to(DEV); Dt = torch.stack(dd).transpose(0, 1).to(DEV)
    return H, Bt, Dt


@torch.no_grad()
def evaluate(model, val, h_mode="real", frames=1600):
    """Deploy = Kalman FILTER on audio -> filtered z -> beat head -> peak-pick. Headline beat-F + leak."""
    model.eval()
    sc = SampleControl(encoder="mean", decoder="mean", state_transition="mean", observation="mean")
    bF, dF = [], []
    n = len(val)
    for i, (hh, b, db) in enumerate(val):
        h_use = val[(i + 1) % n][0] if h_mode == "shuffle" else hh
        T = min(h_use.shape[0], b.shape[0], frames)
        h_in = torch.zeros(T, 1, hh.shape[1], device=DEV) if h_mode == "zero" \
            else h_use[:T].unsqueeze(1).to(DEV)
        a = model.encoder(h_in.reshape(-1, hh.shape[1])).mean.view(T, 1, model.a_dim)
        fm, *_ = model.ssm.kalman_filter(a, sample_control=sc)   # FILTER (causal, uses audio)
        prob = torch.sigmoid(model.head(fm.view(T, model.z_dim)))[:, ].cpu().numpy()
        ref = np.where(b.numpy()[:T] > 0.5)[0] / FPS; dref = np.where(db.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) >= 2: bF.append(fmeas(ref, peaks(prob[:, 0])))
        if len(dref) >= 2: dF.append(fmeas(dref, peaks(prob[:, 1])))
    model.train(); m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(bF), m(dF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--a_dim", type=int, default=8); ap.add_argument("--z_dim", type=int, default=8)
    ap.add_argument("--K", type=int, default=5); ap.add_argument("--beat_w", type=float, default=5.0)
    ap.add_argument("--ntrain", type=int, default=200); ap.add_argument("--nval", type=int, default=40)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f"KVAE-bar-pointer M1 | a_dim={args.a_dim} z_dim={args.z_dim} K={args.K} beat_w={args.beat_w} "
          f"| EXACT differentiable Kalman filter (reused) + MLP VAE on [512] acts", flush=True)
    train = load("cache/acts/bt_train_rich", args.ntrain, 1); val = load("cache/acts/bt_val_rich", args.nval, 2)
    print(f"train={len(train)} val={len(val)}", flush=True)

    model = KVAEBarPointer(h_dim=512, a_dim=args.a_dim, z_dim=args.z_dim, K=args.K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sc = SampleControl(encoder="sample", decoder="mean", state_transition="sample", observation="sample")
    pw = torch.tensor([8.0, 20.0], device=DEV)

    for step in range(1, args.steps + 1):
        H, Bt, Dt = batch(train, 256, 16)
        elbo, z, info = kvae_elbo(model, H, sc)
        beat_logits = model.head(z.reshape(-1, model.z_dim)).view(*z.shape[:2], 2)
        bce = F.binary_cross_entropy_with_logits(beat_logits, torch.stack([Bt, Dt], -1), pos_weight=pw)
        loss = -elbo + args.beat_w * bce
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % 200 == 0 or step == args.steps:
            bF, dF = evaluate(model, val, "real")
            print(f"\nstep {step} | elbo {float(elbo):.1f} (recon {info['recon']:.1f} reg {info['reg']:.1f} "
                  f"obs {info['kal_obs']:.1f} trans {info['kal_trans']:.1f} post {info['kal_post']:.1f}) "
                  f"bce {float(bce):.3f}\n  FILTER deploy: beat {bF:.3f} downbeat {dF:.3f}", flush=True)

    bF, dF = evaluate(model, val, "real")
    bFs, dFs = evaluate(model, val, "shuffle"); bFz, dFz = evaluate(model, val, "zero")
    print("\n--- FINAL (Kalman-FILTER deploy on audio; read-out head on filtered z) ---")
    print(f"  real     : beat {bF:.3f}  downbeat {dF:.3f}   <- the deploy number")
    print(f"  shuffled : beat {bFs:.3f}  downbeat {dFs:.3f}   (must COLLAPSE)")
    print(f"  zero     : beat {bFz:.3f}  downbeat {dFz:.3f}   (must COLLAPSE)")
    print("VERDICT: exact differentiable filter -> beat-F competitive AND leak collapses => WALL BROKEN")


if __name__ == "__main__":
    main()
