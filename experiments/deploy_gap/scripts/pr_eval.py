"""Precision/Recall/F1 split for a checkpoint's read-outs (mir_eval +/-70ms matching).
Tells us HOW a collapsed cell fails: fires nothing (R~0) vs fires everywhere (P~0).
Usage: pr_eval.py <ckpt.pt> [label]"""
import sys, math
import numpy as np, torch
import torch.nn.functional as F
import mir_eval
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.elbo import free_run
from faithful.data import FPS, N_MELS, LogMel, iter_val_songs
from faithful.distributions import TWO_PI
from faithful.evaluate import beats_from_barphase, beats_from_activation

dev = "cuda"
ROOT = "/home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data"
DS = ["ballroom", "beatles", "hains", "rwc_popular"]


def prf(ref, est):
    if len(ref) == 0:
        return (np.nan, np.nan, np.nan)
    if len(est) == 0:
        return (np.nan, 0.0, 0.0)
    m = mir_eval.util.match_events(ref, est, 0.07)
    p = len(m) / len(est); r = len(m) / len(ref)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f)


@torch.no_grad()
def tf_posterior(model, h, b):
    B, T, _ = h.shape
    pc = model.encode_prior(h); qc = model.encode_posterior(h, b)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1)
    traj = [phi]; zf = [model.z_features(meter, phi, lt)]
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(
            torch.cat([qc[:, t], model.z_features(meter, phi, lt)], -1)))
        phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1)
        traj.append(phi); zf.append(model.z_features(meter, phi, lt))
    dec = torch.sigmoid(torch.stack([model.decode(zf[t], pc[:, t]) for t in range(T)], 1))
    return torch.stack(traj, 1)[0].cpu().numpy(), dec[0].cpu().numpy()


def main():
    ck = torch.load(sys.argv[1], map_location=dev); a = ck.get("args", {})
    label = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].split("/")[-2]
    model = BarPointerVAE(h_dim=N_MELS, hidden=a.get("hidden", 64), num_meters=a.get("num_meters", 4),
                          latent_only=a.get("latent_only", False)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    logmel = LogMel().to(dev)
    acc = {k: [] for k in ("tf_dec", "tf_lat", "fr_dec", "fr_lat")}
    for key, audio, beats, downs, meta in iter_val_songs(ROOT, DS, max_per_dataset=4):
        T = min(len(beats), 1200)
        ref = np.where(beats.numpy()[:T] > 0.5)[0] / FPS
        df = np.where(downs.numpy()[:T] > 0.5)[0] / FPS
        if len(ref) < 8:
            continue
        m = 4
        if len(df) >= 2:
            bpb = np.median([np.sum((ref >= df[i]) & (ref < df[i+1])) for i in range(len(df)-1)])
            m = max(2, min(int(round(bpb)) if bpb > 0 else 4, 4))
        h = logmel(audio.to(dev).unsqueeze(0))[:, :T]; b = beats[:T].to(dev).unsqueeze(0).float()
        phi, dec = tf_posterior(model, h, b)
        o = free_run(model, h, temperature=0.3)
        frp = o["phase_mu"][0, :T].cpu().numpy(); frd = o["decoder_prob"][0, :T].cpu().numpy()
        acc["tf_dec"].append(prf(ref, beats_from_activation(dec, FPS)))
        acc["tf_lat"].append(prf(ref, beats_from_barphase(phi, m, FPS)))
        acc["fr_dec"].append(prf(ref, beats_from_activation(frd, FPS)))
        acc["fr_lat"].append(prf(ref, beats_from_barphase(frp, m, FPS)))
    print(f"=== {label} ===  (P / R / F1, +/-70ms)")
    for k, v in acc.items():
        arr = np.array(v, float)
        print(f"  {k:8s} P={np.nanmean(arr[:,0]):.3f}  R={np.nanmean(arr[:,1]):.3f}  F1={np.nanmean(arr[:,2]):.3f}")


if __name__ == "__main__":
    main()
