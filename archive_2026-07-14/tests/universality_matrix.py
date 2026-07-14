"""Universality matrix (one frontend cell): on the SAME evidence, compare four decoders three ways
(F / CMLt / AMLt, beat + downbeat) -- peak-pick, our particle FILTER (causal/approx), our GRID-VITERBI
(offline/exact), and madmom's DBN (offline/exact, hand-tuned). Tests the SOTA-bar-pointer thesis:
does our LEARNED dynamics, decoded the same way madmom decodes its hand-set one, match or beat it?
Usage: python tests/universality_matrix.py --ckpt ... --val ... [--n N]
"""
import argparse, sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM")
import numpy as np, torch
import mir_eval.beat as mbeat
from scipy.signal import find_peaks
from config import load_config
from data.dataset import load_cached_songs
from train import build_model
from model.grid_decode import grid_viterbi_decode

FPS = 22050.0 / 256.0


def pp(sig, h, d):
    pk, _ = find_peaks(sig, height=h, distance=max(1, int(d * FPS)))
    return pk / FPS


def three(ref, est):
    if len(ref) < 2 or len(est) < 2:
        return None
    r = mbeat.evaluate(np.asarray(ref), np.asarray(est))
    return (r["F-measure"], r["Correct Metric Level Total"], r["Any Metric Level Total"])


def dbn_decode(act, dbn_beat, dbn_down):
    import numpy as np
    b = dbn_beat(act)                       # returns beat times
    d = dbn_down(act)
    return b, d


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

    try:
        from madmom.features.beats import DBNBeatTrackingProcessor
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor
        beat_dbn = DBNBeatTrackingProcessor(fps=FPS, min_bpm=50, max_bpm=215)
        down_dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS, min_bpm=50, max_bpm=215)
        have_dbn = True
    except Exception as e:
        print("no madmom DBN:", e); have_dbn = False

    arms = {k: {"b": [], "d": []} for k in ["peakpick", "filter", "grid_viterbi"] + (["madmom_dbn"] if have_dbn else [])}
    with torch.no_grad():
        for s in load_cached_songs(args.val, args.n, selection_seed=2):
            n = min(s.features.shape[0], args.maxf)
            rb = np.where(s.beat_targets[:n].numpy() > 0.5)[0] / FPS
            rd = np.where(s.downbeat_targets[:n].numpy() > 0.5)[0] / FPS
            if len(rb) < 4:
                continue
            feats = s.features[:n].unsqueeze(0).to(args.device)
            obsT = s.frontend_activations[:n].to(args.device)
            obs = obsT.cpu().numpy()
            cand = {}
            cand["peakpick"] = (pp(obs[:, 0], 0.1, 0.10), pp(obs[:, 1], 0.1, 0.30))
            rf = m.filter_deploy(feats, obsT, num_particles=800)
            cand["filter"] = (pp(rf["beat_activation"], 0.1, 0.10), pp(rf["downbeat_activation"], 0.1, 0.30))
            rg = grid_viterbi_decode(m, feats, obsT)
            cand["grid_viterbi"] = (pp(rg["beat_activation"], 0.5, 0.10), pp(rg["downbeat_activation"], 0.5, 0.30))
            if have_dbn:
                try:
                    bt = beat_dbn(obs[:, 0])            # beat-optimized DBN on the beat channel
                    out = down_dbn(obs)                 # downbeat DBN for the bar-1 positions
                    db = out[out[:, 1] == 1, 0]
                    cand["madmom_dbn"] = (bt, db)
                except Exception:
                    cand["madmom_dbn"] = (np.array([]), np.array([]))
            for name, (pb, pd) in cand.items():
                tb = three(rb, pb)
                if tb:
                    arms[name]["b"].append(tb)
                if len(rd) >= 2:
                    td = three(rd, pd)
                    if td:
                        arms[name]["d"].append(td)

    n_scored = len(arms["filter"]["b"])
    print(f"\nn={n_scored} | ckpt {args.ckpt.split('/')[-1]}", flush=True)
    print(f"{'decoder':14s} | {'beatF':>6s} {'bCMLt':>6s} {'bAMLt':>6s} | {'dbF':>6s} {'dCMLt':>6s} {'dAMLt':>6s}", flush=True)
    for name, d in arms.items():
        b = np.array(d["b"]); dn = np.array(d["d"])
        bm = b.mean(0) if len(b) else [float('nan')] * 3
        dm = dn.mean(0) if len(dn) else [float('nan')] * 3
        print(f"{name:14s} | {bm[0]:6.3f} {bm[1]:6.3f} {bm[2]:6.3f} | {dm[0]:6.3f} {dm[1]:6.3f} {dm[2]:6.3f}", flush=True)
    print("UNIVERSALITY_MATRIX_DONE", flush=True)


if __name__ == "__main__":
    main()
