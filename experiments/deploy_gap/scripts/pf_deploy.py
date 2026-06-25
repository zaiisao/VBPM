"""ROUTE 2 — trainable-DBN MVP: learned audio activation + PARTICLE FILTER, tempo a PURE LATENT.

The DBN identifies tempo via exact global inference over the activation (forward-backward); the
continuous analogue is a particle filter. We keep the bar-pointer dynamics audio-blind (tempo = pure
latent random walk) and let the FILTER identify tempo by weighting particles against a learned beat
activation a(h). Decisive test: does PF-deploy beat raw peak-picking on the same a(h)? If yes, the
DBN-style inference (tempo consistency) adds value the frontend alone can't.

Stage 1 (NN): tiny BiGRU activation net A(h), BCE to widened beats (end-to-end, from random weights).
Stage 2 (DBN): particle filter over (phi, log_tempo); prior is audio-blind; weight by agreement with a(h).
"""
import sys, math, argparse
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import mir_eval
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.data import FPS, N_MELS, LogMel, build_train_loader, iter_val_songs

dev = "cuda"; TWO_PI = 2 * math.pi
ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]


class ActNet(nn.Module):   # audio -> per-frame beat activation (the "NN" frontend)
    def __init__(self, h_dim=N_MELS, hid=64):
        super().__init__()
        self.gru = nn.GRU(h_dim, hid, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hid, 1)
    def forward(self, h):
        out, _ = self.gru(h)
        return self.head(out).squeeze(-1)   # [B,T] logit


def widen(b, W=6):
    return F.max_pool1d(b.unsqueeze(1), 2 * W + 1, 1, W).squeeze(1)


def f1(ref, est):
    if len(ref) == 0: return float("nan")
    if len(est) == 0: return 0.0
    return float(mir_eval.beat.f_measure(ref, est))


def peak_pick(prob, fps, thr=0.5, mind=0.10):
    pk = [t for t in range(1, len(prob)-1) if prob[t] >= thr and prob[t] >= prob[t-1] and prob[t] >= prob[t+1]]
    out, last = [], -1e9
    for t in pk:
        if t - last >= mind*fps: out.append(t); last = t
    return np.array(out)/fps


@torch.no_grad()
def particle_filter(act, m, K=600, sigma=0.02, kappa=6.0):
    """act: [T] beat activation in [0,1]. State per particle: phi (bar phase), lt (log bar-advance/frame).
    Tempo is a PURE LATENT random walk (audio-blind); the FILTER identifies it by weighting vs act.
    Returns the weighted-circular-mean bar-phase trajectory [T]."""
    T = len(act)
    # init: broad tempo prior covering ~50-210 BPM (bar-advance = m beats); phase uniform
    bpm = np.random.uniform(50, 210, K)
    lt = np.log(TWO_PI * (bpm/60.0) / m / FPS)            # rad/frame bar advance for that beat-BPM
    lt = torch.tensor(lt, device=dev, dtype=torch.float32)
    phi = torch.rand(K, device=dev) * TWO_PI
    a = torch.tensor(act, device=dev, dtype=torch.float32)
    mean_phi = np.empty(T)
    for t in range(T):
        if t > 0:                                          # propagate (audio-blind prior)
            lt = lt + sigma * torch.randn(K, device=dev)
            phi = (phi + torch.exp(lt)) % TWO_PI
        e = torch.exp(kappa * (torch.cos(m * phi) - 1.0))  # predicted beat-prob from phase (bump at subdivisions)
        w = a[t] * e + (1 - a[t]) * (1 - e) + 1e-6         # agreement with observed activation
        w = w / w.sum()
        mean_phi[t] = math.atan2(float((w*torch.sin(phi)).sum()), float((w*torch.cos(phi)).sum())) % TWO_PI
        ess = 1.0 / float((w*w).sum())
        if ess < K/2:                                      # systematic resample
            idx = torch.multinomial(w, K, replacement=True)
            phi = phi[idx]; lt = lt[idx]
    return mean_phi


def beats_from_barphase(phase, m, fps, mind=0.10):
    psi = (m*np.asarray(phase)) % TWO_PI
    w = np.where(np.diff(psi) < -math.pi)[0] + 1
    out, last = [], -1e9
    for f in w:
        if f-last >= mind*fps: out.append(f); last = f
    return np.array(out)/fps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--K", type=int, default=600)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    logmel = LogMel().to(dev); net = ActNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    loader = build_train_loader(ROOT, DS, 256, 16, examples_per_epoch=1000, num_workers=4, seed=0)
    di = iter(loader)
    for s in range(1, args.steps+1):                       # Stage 1: train the activation net
        try: a, b, _ = next(di)
        except StopIteration: di = iter(loader); a, b, _ = next(di)
        h = logmel(a.to(dev))[:, :256]; bt = b[:, :256].to(dev)
        T = min(h.shape[1], bt.shape[1]); h, bt = h[:, :T], bt[:, :T]
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(net(h), widen(bt), reduction="mean")
        loss.backward(); opt.step()
        if s % 100 == 0 or s == 1: print(f"[act] s{s} bce={float(loss):.4f}", flush=True)
    net.eval()
    pk, pf = [], []
    for key, audio, beats, downs, meta in iter_val_songs(ROOT, DS, max_per_dataset=4):
        T = min(len(beats), 1200)
        ref = np.where(beats.numpy()[:T] > 0.5)[0]/FPS
        df = np.where(downs.numpy()[:T] > 0.5)[0]/FPS
        if len(ref) < 8: continue
        m = 4
        if len(df) >= 2:
            bpb = np.median([np.sum((ref>=df[i])&(ref<df[i+1])) for i in range(len(df)-1)]); m = max(2, min(int(round(bpb)) if bpb>0 else 4, 4))
        h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
        with torch.no_grad():
            act = torch.sigmoid(net(h))[0].cpu().numpy()
        pk.append(f1(ref, peak_pick(act, FPS)))            # baseline: raw frontend peak-pick
        mphi = particle_filter(act, m, K=args.K)            # trainable-DBN: PF inference
        pf.append(f1(ref, beats_from_barphase(mphi, m, FPS)))
    print(f"\n=== ROUTE 2 (trainable DBN: activation + particle filter, tempo pure latent) ===")
    print(f"  raw activation peak-pick   F1 = {np.nanmean(pk):.3f}   (frontend alone, no DBN)")
    print(f"  PARTICLE FILTER deploy     F1 = {np.nanmean(pf):.3f}   (DBN inference over activation)")
    print(f"  -> PF {'BEATS' if np.nanmean(pf)>np.nanmean(pk) else 'does NOT beat'} raw peak-pick "
          f"(delta {np.nanmean(pf)-np.nanmean(pk):+.3f}); vs open-loop free-run baseline ~0.33-0.41")


if __name__ == "__main__":
    main()
