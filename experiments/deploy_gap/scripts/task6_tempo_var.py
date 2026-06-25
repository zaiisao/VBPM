"""TASK 6 — TEMPO VARIANCE-GROWTH.

Is the 1e9-BPM blowup just the unbounded log-random-walk prior behaving exactly as defined?
A pure RW log tau_t = log tau_{t-1} + sigma*eps has Var[log tau_t] = sum_{s<=t} sigma_s^2,
growing ~linearly in t. We estimate the empirical Var across N stochastic free-run rollouts at
each frame and overlay the cumulative-sigma^2 prediction. If they match, the divergence is the
prior class being improper for length-T rollout (argues for an OU / mean-reverting tempo prior),
NOT an optimization bug. Runs on BOTH the strict and overshoot checkpoints (test-both).
Saves task6_tempo_var.png.
"""
import sys, math
import numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.elbo import free_run
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs

dev = "cuda"
ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]


def analyze(ckpt, N=64, T=800):
    ck = torch.load(ckpt, map_location=dev); a = ck.get("args", {})
    model = BarPointerVAE(h_dim=N_MELS, hidden=a.get("hidden", 64),
                          num_meters=a.get("num_meters", 4)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    logmel = LogMel().to(dev)
    key, audio, beats, downs, meta = next(iter(iter_val_songs(ROOT, DS, max_per_dataset=1)))
    h = logmel(audio.to(dev).unsqueeze(0))[:, :T]
    Tt = h.shape[1]
    with torch.no_grad():
        # empirical Var[log_tempo_t] across N stochastic rollouts
        lts = []
        for _ in range(N):
            o = free_run(model, h, temperature=0.3)
            lts.append(o["log_tempo"][0, :Tt].cpu().numpy())
        lts = np.stack(lts, 0)                       # [N, T]
        emp_var = lts.var(0)                         # [T]
        # per-frame prior sigma -> cumulative sigma^2 prediction
        pc = model.encode_prior(h)
        sig = (F.softplus(model.prior_tempo_sigma(pc).squeeze(-1)) + 1e-3)[0, :Tt].cpu().numpy()
        cum = np.cumsum(sig ** 2)
        bpm_med = np.median(np.exp(lts), 0)
    return emp_var, cum, sig.mean(), Tt, key


def main():
    cks = {"strict": "/home/sogang/jaehoon/CHART/runs/strict_elbo/final.pt",
           "overshoot": "/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-CHART/84e38297-7220-4bbe-b30a-42cd7c5a3087/scratchpad/runs/os/d4/final.pt"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (name, ck) in zip(axes, cks.items()):
        ev, cum, sbar, T, key = analyze(ck)
        t = np.arange(T)
        ax.plot(t, ev, label="empirical Var[log tau_t]", lw=2)
        ax.plot(t, cum, "--", label="cumulative sum sigma_s^2 (RW prediction)", lw=2)
        ax.plot(t, sbar**2 * t, ":", label=f"t*sigmabar^2 (sigmabar={sbar:.3f})", lw=1.5)
        ax.set_title(f"{name}  (song={key})"); ax.set_xlabel("frame t"); ax.set_ylabel("Var[log tau]")
        ax.legend(fontsize=8)
        print(f"{name}: sigmabar={sbar:.4f}  emp_var[-1]={ev[-1]:.3f}  RW_pred[-1]={cum[-1]:.3f}  "
              f"ratio={ev[-1]/(cum[-1]+1e-9):.2f}")
    fig.suptitle("Tempo log-random-walk variance growth: empirical vs prior prediction")
    fig.tight_layout()
    out = "/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-CHART/84e38297-7220-4bbe-b30a-42cd7c5a3087/scratchpad/task6_tempo_var.png"
    fig.savefig(out, dpi=110); print("saved", out)


if __name__ == "__main__":
    main()
