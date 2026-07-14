"""Does meter co-training lift DOWNBEAT F, and is the lift concentrated on non-4/4 (the DBN's blind
spot)? Compares downbeat F of meterA (meter co-trained) vs control vs madmom DBN, STRATIFIED by the
song's meter (inferred from the downbeat annotations: median beats between consecutive downbeats).
The thesis: our explicit meter latent beats the DBN's fixed-{3,4} assumption, most on non-4/4.
"""
import sys; sys.path.insert(0, "/home/sogang/jaehoon/VBPM")
import numpy as np, torch
import mir_eval.beat as mbeat
from scipy.signal import find_peaks
from config import load_config
from data.dataset import load_cached_songs
from train import build_model
FPS = 22050.0 / 256.0


def pp(sig, h, d):
    pk, _ = find_peaks(sig, height=h, distance=max(1, int(d * FPS)))
    return pk / FPS


def dbF(ref, est):
    if len(ref) < 2 or len(est) < 2:
        return None
    return mbeat.evaluate(np.asarray(ref), np.asarray(est))["F-measure"]


def infer_meter(beat_t, down_t):
    """beats per bar = median beats in [downbeat_i, downbeat_{i+1})."""
    if len(down_t) < 3:
        return 4
    counts = [np.sum((beat_t >= down_t[i]) & (beat_t < down_t[i + 1])) for i in range(len(down_t) - 1)]
    counts = [c for c in counts if c > 0]
    return int(round(np.median(counts))) if counts else 4


cfg = load_config()
ctrl = build_model(cfg).cuda(); ctrl.load_state_dict(torch.load("checkpoints/foldhonest_s0.pt", map_location="cpu")); ctrl.eval()
meterA = build_model(cfg).cuda(); meterA.load_state_dict(torch.load("checkpoints/meterA_s0.pt", map_location="cpu")); meterA.eval()
try:
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    down_dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS, min_bpm=50, max_bpm=215)
    have_dbn = True
except Exception as e:
    print("no dbn", e); have_dbn = False

strata = {}   # meter -> {ctrl:[], meterA:[], dbn:[]}
with torch.no_grad():
    for s in load_cached_songs("cache/acts/foldhonest_val_rich", 999, selection_seed=2):
        n = min(s.features.shape[0], 6000)
        rb = np.where(s.beat_targets[:n].numpy() > 0.5)[0] / FPS
        rd = np.where(s.downbeat_targets[:n].numpy() > 0.5)[0] / FPS
        if len(rb) < 4 or len(rd) < 3:
            continue
        meter = infer_meter(rb, rd)
        key = "4/4" if meter == 4 else f"non-4/4({meter})" if meter in (3, 5, 6, 7) else f"other({meter})"
        bucket = "4/4" if meter == 4 else "non-4/4"
        feats = s.features[:n].unsqueeze(0).cuda(); obsT = s.frontend_activations[:n].cuda()
        rc = ctrl.filter_deploy(feats, obsT, num_particles=800)
        rm = meterA.filter_deploy(feats, obsT, num_particles=800)
        row = strata.setdefault(bucket, {"ctrl": [], "meterA": [], "dbn": [], "n": 0})
        row["n"] += 1
        fc = dbF(rd, pp(rc["downbeat_activation"], 0.1, 0.30))
        fm = dbF(rd, pp(rm["downbeat_activation"], 0.1, 0.30))
        if fc is not None: row["ctrl"].append(fc)
        if fm is not None: row["meterA"].append(fm)
        if have_dbn:
            try:
                out = down_dbn(obsT.cpu().numpy()); db = out[out[:, 1] == 1, 0]
                fd = dbF(rd, db)
                if fd is not None: row["dbn"].append(fd)
            except Exception:
                pass

print(f"\n{'stratum':10s} | {'n':>4s} | {'control':>8s} {'meterA':>8s} {'DBN':>8s}", flush=True)
for k, r in strata.items():
    print(f"{k:10s} | {r['n']:4d} | {np.nanmean(r['ctrl']):8.3f} {np.nanmean(r['meterA']):8.3f} "
          f"{np.nanmean(r['dbn']) if r['dbn'] else float('nan'):8.3f}", flush=True)
print("DOWNBEAT_BY_METER_DONE", flush=True)
