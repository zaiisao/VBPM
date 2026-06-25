"""ROUTE 1 — trainable bar-pointer STATE-SPACE VAE, deployed by a PARTICLE FILTER (the DBN mechanism).

A REAL VAE (learned generative model + ELBO training + variational posterior), but deployed with
inference instead of open-loop free-run:
  observation o_t = onset envelope (fixed audio transform, like log-mel -> no collusion)
  latent z_t = (phase phi, log-tempo lt); tempo is a PURE LATENT (audio-blind RW mean)
  generative emission p(o_t|z_t): learned bump of phase -> expected onset (high at beat subdivisions)
  TRAIN: amortized ELBO (encoder q(z_t|o, z_{t-1}); reparam vM phase + Gaussian log-tempo)
  DEPLOY: particle filter over the prior, weighted by the LEARNED emission p(o_t|z_t) -> searches
          tempo+phase, the audio eliminates wrong hypotheses; read beats from the filtered phase.
Compare DEPLOY-by-filter vs DEPLOY-by-free-run (same model) and the classic ref (~0.66).
"""
import sys, math, argparse
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import mir_eval
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.data import FPS, N_MELS, LogMel, build_train_loader, iter_val_songs
from faithful.distributions import (TWO_PI, sample_von_mises, kl_von_mises, kl_log_normal, log_i0)

dev = "cuda"; ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"; DS = ["ballroom","beatles","hains","rwc_popular"]
M = 4  # meter fixed for the MVP (focus on tempo+phase, the diagnosed breakdown)


def onset_env(mel):  # [.,T,N] -> [.,T] positive spectral flux, per-seq z-scored
    flux = F.relu(mel[:, 1:] - mel[:, :-1]).sum(-1)
    flux = F.pad(flux, (1, 0))
    return (flux - flux.mean(1, keepdim=True)) / (flux.std(1, keepdim=True) + 1e-6)


class DBNVae(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.enc = nn.GRU(1, hid, batch_first=True, bidirectional=True)
        self.ctx = nn.Linear(2*hid, hid)
        self.head = nn.Linear(hid + 3, 5)          # in: ctx + z_feat(cosphi,sinphi,lt) -> u,v,logkappa,lt_mu,lt_logsig
        self.z0 = nn.Parameter(torch.zeros(3))
        self.kappa_em = nn.Parameter(torch.tensor(2.0))    # emission bump sharpness
        self.w_em = nn.Parameter(torch.tensor(1.0)); self.b_em = nn.Parameter(torch.tensor(0.0))
        self.log_sig_obs = nn.Parameter(torch.tensor(0.0))
        self.log_sig_tempo = nn.Parameter(torch.tensor(math.log(0.02)))
        self.kappa_phi = nn.Parameter(torch.tensor(2.0))   # prior phase concentration (how tightly phi follows phi+phidot)
    def encode(self, o):
        out, _ = self.enc(o.unsqueeze(-1)); return torch.tanh(self.ctx(out))
    def unpack(self, v):
        u, w = v[:, 0], v[:, 1]
        return (torch.atan2(w, u) % TWO_PI, F.softplus(v[:, 2]) + 0.01, v[:, 3], F.softplus(v[:, 4]) + 1e-3)
    def emit_mu(self, phi):                          # expected onset from phase (bump at beat subdivisions)
        return self.w_em * torch.exp((F.softplus(self.kappa_em)) * (torch.cos(M * phi) - 1.0)) + self.b_em
    def zf(self, phi, lt):
        return torch.stack([torch.cos(phi), torch.sin(phi), lt], -1)


def elbo(model, o):
    B, T = o.shape
    qc = model.encode(o); sig_obs = F.softplus(model.log_sig_obs) + 1e-2; sig_t = F.softplus(model.log_sig_tempo) + 1e-3
    kphi = F.softplus(model.kappa_phi) + 0.01            # learned prior phase concentration
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    phm, phk, ltm, lts = model.unpack(model.head(torch.cat([qc[:, 0], z0], -1)))
    phi = sample_von_mises(phm, phk) % TWO_PI; lt = ltm + lts * torch.randn_like(ltm)
    recon = ((o[:, 0] - model.emit_mu(phi))**2 / (2*sig_obs**2) + torch.log(sig_obs))
    klp = kl_von_mises(phm, phk, phm.new_zeros(B), phm.new_full((B,), 0.01))      # ~uniform phase prior at t0
    klt = lt.new_zeros(B)
    pp, lp = phi, lt
    for t in range(1, T):
        phm, phk, ltm, lts = model.unpack(model.head(torch.cat([qc[:, t], model.zf(pp, lp)], -1)))
        ppm = (pp + torch.exp(lp)) % TWO_PI; ptm = lp                              # pure-latent tempo (audio-blind)
        phi = sample_von_mises(phm, phk) % TWO_PI; lt = ltm + lts * torch.randn_like(ltm)
        recon = recon + ((o[:, t] - model.emit_mu(phi))**2 / (2*sig_obs**2) + torch.log(sig_obs))
        klp = klp + kl_von_mises(phm, phk, ppm, kphi.expand(B))
        klt = klt + kl_log_normal(ltm, lts, ptm, lt.new_full((B,), float(sig_t)))
        pp, lp = phi, lt
    return (recon + klp + klt).mean()


@torch.no_grad()
def pf_deploy(model, o, m, K=800):
    T = o.shape[1]; sig_t = float(F.softplus(model.log_sig_tempo) + 1e-3); sig_obs = float(F.softplus(model.log_sig_obs)+1e-2)
    bpm = np.random.uniform(50, 210, K); lt = torch.tensor(np.log(TWO_PI*(bpm/60)/m/FPS), device=dev, dtype=torch.float32)
    phi = torch.rand(K, device=dev) * TWO_PI; oo = o[0]
    mean_phi = np.empty(T)
    ke = float(F.softplus(model.kappa_em)); we = float(model.w_em); be = float(model.b_em)
    for t in range(T):
        if t > 0:
            lt = lt + sig_t * torch.randn(K, device=dev); phi = (phi + torch.exp(lt)) % TWO_PI
        mu = we * torch.exp(ke * (torch.cos(m*phi) - 1.0)) + be
        logw = -((oo[t] - mu)**2) / (2*sig_obs**2)
        w = torch.softmax(logw, 0)
        mean_phi[t] = math.atan2(float((w*torch.sin(phi)).sum()), float((w*torch.cos(phi)).sum())) % TWO_PI
        if 1.0/float((w*w).sum()) < K/2:
            idx = torch.multinomial(w, K, replacement=True); phi, lt = phi[idx], lt[idx]
    return mean_phi


@torch.no_grad()
def free_run(model, o, m):  # open-loop (no filter): init from t=0 posterior, roll prior mean
    qc = model.encode(o); z0 = model.z0.unsqueeze(0).expand(1, -1)
    phm, phk, ltm, lts = model.unpack(model.head(torch.cat([qc[:, 0], z0], -1)))
    phi = float(phm[0]); lt = float(ltm[0]); T = o.shape[1]; ch = [phi]
    for t in range(1, T): phi = (phi + math.exp(lt)) % TWO_PI; ch.append(phi)
    return np.array(ch)


def beats(phase, m):
    psi = (m*np.asarray(phase)) % TWO_PI; w = np.where(np.diff(psi) < -math.pi)[0]+1
    out, last = [], -1e9
    for fr in w:
        if fr-last >= 0.10*FPS: out.append(fr); last = fr
    return np.array(out)/FPS
def f1(ref, est): return 0.0 if len(est)==0 else (float("nan") if len(ref)==0 else float(mir_eval.beat.f_measure(ref, est)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=600); args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    logmel = LogMel().to(dev); model = DBNVae().to(dev); opt = torch.optim.AdamW(model.parameters(), 2e-3)
    loader = build_train_loader(ROOT, DS, 256, 16, examples_per_epoch=1000, num_workers=4, seed=0); di = iter(loader)
    for s in range(1, args.steps+1):
        try: a, b, _ = next(di)
        except StopIteration: di = iter(loader); a, b, _ = next(di)
        o = onset_env(logmel(a.to(dev))[:, :256])
        opt.zero_grad(); loss = elbo(model, o); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if s % 100 == 0 or s == 1: print(f"[dbnvae] s{s} elbo={float(loss):.2f} kem={float(F.softplus(model.kappa_em)):.2f} sigT={float(F.softplus(model.log_sig_tempo)):.4f}", flush=True)
    torch.save(model.state_dict(), "/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-CHART/84e38297-7220-4bbe-b30a-42cd7c5a3087/scratchpad/dbn_vae.pt")
    model.eval(); ff, pf = [], []
    for key, audio, bts, downs, meta in iter_val_songs(ROOT, DS, max_per_dataset=4):
        T = min(len(bts), 1200); ref = np.where(bts.numpy()[:T] > 0.5)[0]/FPS; df = np.where(downs.numpy()[:T] > 0.5)[0]/FPS
        if len(ref) < 8: continue
        m = 4
        if len(df) >= 2:
            bpb = np.median([np.sum((ref>=df[i])&(ref<df[i+1])) for i in range(len(df)-1)]); m = max(2, min(int(round(bpb)) if bpb>0 else 4, 4))
        o = onset_env(logmel(audio.to(dev).unsqueeze(0))[:, :T])
        ff.append(f1(ref, beats(free_run(model, o, m), m)))
        pf.append(f1(ref, beats(pf_deploy(model, o, m), m)))
    print(f"\n=== ROUTE 1: trainable bar-pointer SSM-VAE, ELBO-trained, deployed two ways ===")
    print(f"  DEPLOY by FREE-RUN (open-loop, no inference) F1 = {np.nanmean(ff):.3f}")
    print(f"  DEPLOY by PARTICLE FILTER (inference)        F1 = {np.nanmean(pf):.3f}")
    print(f"  -> filter {'BEATS' if np.nanmean(pf)>np.nanmean(ff) else 'does NOT beat'} free-run "
          f"(delta {np.nanmean(pf)-np.nanmean(ff):+.3f}); refs: VAE free-run ~0.36 | classic tempo+phase ~0.66")


if __name__ == "__main__":
    main()
