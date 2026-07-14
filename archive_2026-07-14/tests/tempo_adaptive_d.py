"""Honest per-song read-out adaptation: set the peak-pick min-distance d from EACH SONG'S estimated
beat period (autocorrelation of the activation -- NO ground truth), vs a fixed global d. Tests how
much of the peak-pick baseline's ceiling is just the fixed-d compromise across tempos. Deployable
(the tempo estimate uses only the observable activation). h stays fixed; this isolates d.
"""
import argparse, sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM")
import numpy as np, torch
import mir_eval.beat as mbeat
from scipy.signal import find_peaks
from config import load_config
from data.dataset import load_cached_songs
from train import build_model

FPS = 22050.0 / 256.0


def estimate_beat_period(activation):
    a = activation - activation.mean()
    ac = np.correlate(a, a, mode="full")[len(a) - 1:]
    lag_lo = int(FPS * 60.0 / 215.0)
    lag_hi = int(FPS * 60.0 / 50.0)
    if lag_hi >= len(ac):
        lag_hi = len(ac) - 1
    if lag_hi <= lag_lo:
        return 0.5
    best = lag_lo + int(np.argmax(ac[lag_lo:lag_hi]))
    return best / FPS


def pp(sig, h, d_sec):
    pk, _ = find_peaks(sig, height=h, distance=max(1, int(d_sec * FPS)))
    return pk / FPS


def three(ref, est):
    if len(ref) < 2 or len(est) < 2:
        return None
    r = mbeat.evaluate(np.asarray(ref), np.asarray(est))
    return (r["F-measure"], r["Correct Metric Level Total"], r["Any Metric Level Total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/sogang/jaehoon/VBPM/checkpoints/foldhonest_s0.pt")
    ap.add_argument("--val", default="/home/sogang/jaehoon/VBPM/cache/acts/foldhonest_val_rich")
    ap.add_argument("--n", type=int, default=999)
    ap.add_argument("--maxf", type=int, default=6000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(); m = build_model(cfg).to(args.device)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu")); m.eval()

    arms = {k: [] for k in ["obs_fixed", "obs_adaptive", "filt_fixed", "filt_adaptive"]}
    with torch.no_grad():
        for s in load_cached_songs(args.val, args.n, selection_seed=2):
            n = min(s.features.shape[0], args.maxf)
            rb = np.where(s.beat_targets[:n].numpy() > 0.5)[0] / FPS
            if len(rb) < 4:
                continue
            obs = s.frontend_activations[:n, 0].cpu().numpy()
            rf = m.filter_deploy(s.features[:n].unsqueeze(0).to(args.device),
                                 s.frontend_activations[:n].to(args.device), num_particles=800)
            ba = rf["beat_activation"]
            d_obs = 0.5 * estimate_beat_period(obs)
            d_filt = 0.5 * estimate_beat_period(ba)
            for name, sig, d in [("obs_fixed", obs, 0.10), ("obs_adaptive", obs, d_obs),
                                 ("filt_fixed", ba, 0.10), ("filt_adaptive", ba, d_filt)]:
                t = three(rb, pp(sig, 0.1, d))
                if t:
                    arms[name].append(t)

    n_scored = len(arms["obs_fixed"])
    print(f"\nn={n_scored} | ckpt {args.ckpt.split('/')[-1]}", flush=True)
    print(f"{'read-out':16s} | {'beatF':>6s} {'CMLt':>6s} {'AMLt':>6s}", flush=True)
    for name, v in arms.items():
        a = np.array(v); mn = a.mean(0) if len(a) else [float('nan')] * 3
        print(f"{name:16s} | {mn[0]:6.3f} {mn[1]:6.3f} {mn[2]:6.3f}", flush=True)
    print("TEMPO_ADAPTIVE_D_DONE", flush=True)


if __name__ == "__main__":
    main()
