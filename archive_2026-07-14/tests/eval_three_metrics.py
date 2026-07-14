"""Canonical three-metric comparison (F-measure, CMLt, AMLt) for beat AND downbeat, via mir_eval --
the same trio the Beat This / BeatFM tables report. One consistent harness so close scores can be
compared honestly across arms; the beat-F-only read-outs hid where we actually win (continuity) or
lose (downbeats).

Compares, on the SAME val songs, SAME evidence (frontend act2):
  * vanilla peak-pick        -- frontend activation, height 0.1 / dist 0.1s|0.3s
  * VBPM filter, default     -- Bayesian activation, height 0.1 / dist 0.1s|0.3s
  * VBPM filter, sweep-winner -- Bayesian activation, height 0.15 / dist 0.13s|0.3s

Usage: python tests/eval_three_metrics.py [--ckpt PATH] [--val DIR] [--device cpu|cuda] [--n N]
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


def pp(sig, height, dist_s):
    pk, _ = find_peaks(sig, height=height, distance=max(1, int(dist_s * FPS)))
    return pk / FPS


def three(ref, est):
    if len(ref) < 2 or len(est) < 2:
        return None
    d = mbeat.evaluate(np.asarray(ref), np.asarray(est))
    return (d["F-measure"], d["Correct Metric Level Total"], d["Any Metric Level Total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/sogang/jaehoon/VBPM/checkpoints/foldhonest_s0.pt")
    ap.add_argument("--val", default="/home/sogang/jaehoon/VBPM/cache/acts/foldhonest_val_rich")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=999)
    ap.add_argument("--maxf", type=int, default=6000)
    args = ap.parse_args()

    cfg = load_config(); m = build_model(cfg).to(args.device)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu")); m.eval()

    arms = {"vanilla_peakpick": {"b": [], "d": []},
            "filter_default": {"b": [], "d": []},
            "filter_winner": {"b": [], "d": []}}
    with torch.no_grad():
        for s in load_cached_songs(args.val, args.n, selection_seed=2):
            n = min(s.features.shape[0], args.maxf)
            rb = np.where(s.beat_targets[:n].numpy() > 0.5)[0] / FPS
            rd = np.where(s.downbeat_targets[:n].numpy() > 0.5)[0] / FPS
            if len(rb) < 4:
                continue
            ob = s.frontend_activations[:n, 0].cpu().numpy()
            od = s.frontend_activations[:n, 1].cpu().numpy()
            r = m.filter_deploy(s.features[:n].unsqueeze(0).to(args.device),
                                s.frontend_activations[:n].to(args.device), num_particles=800)
            ba, da = r["beat_activation"], r["downbeat_activation"]
            cand = {"vanilla_peakpick": (pp(ob, 0.1, 0.10), pp(od, 0.1, 0.30)),
                    "filter_default": (pp(ba, 0.1, 0.10), pp(da, 0.1, 0.30)),
                    "filter_winner": (pp(ba, 0.15, 0.13), pp(da, 0.15, 0.30))}
            for name, (pb, pd) in cand.items():
                tb = three(rb, pb)
                if tb:
                    arms[name]["b"].append(tb)
                if len(rd) >= 2:
                    td = three(rd, pd)
                    if td:
                        arms[name]["d"].append(td)

    n_scored = len(arms["filter_default"]["b"])
    print(f"\nn={n_scored} songs | ckpt {args.ckpt.split('/')[-1]}", flush=True)
    print(f"{'arm':20s} | {'beat F':>7s} {'CMLt':>6s} {'AMLt':>6s} | {'db F':>7s} {'CMLt':>6s} {'AMLt':>6s}", flush=True)
    for name, d in arms.items():
        b = np.array(d["b"]); dn = np.array(d["d"])
        bm = b.mean(0) if len(b) else [np.nan] * 3
        dm = dn.mean(0) if len(dn) else [np.nan] * 3
        print(f"{name:20s} | {bm[0]:7.3f} {bm[1]:6.3f} {bm[2]:6.3f} | {dm[0]:7.3f} {dm[1]:6.3f} {dm[2]:6.3f}", flush=True)
    print("THREE_METRICS_DONE", flush=True)


if __name__ == "__main__":
    main()
