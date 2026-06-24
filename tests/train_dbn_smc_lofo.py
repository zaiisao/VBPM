"""Within-SMC leave-one-fold-out: does a likelihood trained on the SMC distribution
(leak-free) beat the fixed madmom likelihood?

For each of the 8 SMC folds, train a fresh emission on the OTHER 7 folds' SMC tracks
and evaluate the held-out fold; aggregate over all 217. The Beat-This activations are
already held-out (per-fold checkpoints), and our likelihood never sees the test fold,
so it's clean. Compares to peak-pick (0.627) and the fixed-madmom DBN (0.594).

    python tests/train_dbn_smc_lofo.py --steps_per_fold 200
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from scipy.special import expit
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.bar_pointer_dbn import BarPointerDBN

SMC = "/home/sogang/jaehoon/Analyze-SMC"; FPS = 50.0


class Emission(nn.Module):
    def __init__(self, in_dim=2, C=2, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, C))
    def forward(self, h):
        return torch.log_softmax(self.net(h), dim=-1)


def _dilate(x, w):
    k = torch.ones(1, 1, 2 * w + 1, device=x.device)
    return (F.conv1d(x.view(1, 1, -1), k, padding=w)[0, 0] > 0.5).float()


def _peakpick(prob, thresh=0.5, width=7):
    t = torch.from_numpy(np.ascontiguousarray(prob)).float().unsqueeze(0)
    peaks = t.masked_fill(t != F.max_pool1d(t, width, 1, width // 2), -1000.0)
    fr = torch.nonzero(peaks.squeeze(0) > thresh).numpy()[:, 0]
    if len(fr):
        keep = [fr[0]]
        for x in fr[1:]:
            if x - keep[-1] > 1:
                keep.append(x)
        fr = np.array(keep)
    return fr / FPS


def _load_smc():
    ACT = SMC + "/beat_this_activations_cache"; GTd = SMC + "/beat_this_annotations/smc/annotations/beats"
    split = {}
    with open(SMC + "/beat_this_annotations/smc/8-folds.split") as f:
        for line in f:
            t, fo = line.split(); split[t] = int(fo)
    data = {}
    for fpath in sorted(glob.glob(ACT + "/*.npz")):
        tid = os.path.splitext(os.path.basename(fpath))[0]
        if tid not in split:
            continue
        z = np.load(fpath)
        beat, down = expit(z["beat"].astype(np.float64)), expit(z["downbeat"].astype(np.float64))
        gt = None
        for nm in (tid, tid.upper()):
            p = os.path.join(GTd, nm + ".beats")
            if os.path.exists(p):
                d = np.loadtxt(p); gt = d if d.ndim == 1 else d[:, 1]; break
        if gt is None or len(gt) < 2:
            continue
        act = torch.tensor(np.stack([beat, down], -1), dtype=torch.float32)
        T = act.shape[0]
        ind = torch.zeros(T)
        fr = np.round(gt * FPS).astype(int); fr = fr[(fr >= 0) & (fr < T)]; ind[fr] = 1
        data[tid] = {"act": act, "gt": gt, "ind": ind, "fold": split[tid]}
    return data


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps_per_fold", type=int, default=200)
    p.add_argument("--frames", type=int, default=384)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dilate", type=int, default=3)
    cli = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = _load_smc()
    tids = sorted(data); folds = sorted(set(data[t]["fold"] for t in tids))
    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=None, learnable_lambda=False).to(dev)
    print(f"[lofo] {len(tids)} SMC tracks, {len(folds)} folds, DBN {dbn.num_states} states", flush=True)

    pk, fx = [], []
    with torch.no_grad():
        for t in tids:
            d = data[t]
            pk.append(mir_eval.beat.evaluate(d["gt"], _peakpick(d["act"][:, 0].numpy()))["F-measure"])
            bfr, _ = dbn.decode(d["act"].to(dev))
            fx.append(mir_eval.beat.evaluate(d["gt"], bfr.cpu().numpy().astype(float) / FPS)["F-measure"])
    print(f"[lofo] REFERENCES: peak-pick={np.mean(pk):.4f}  fixed-DBN={np.mean(fx):.4f}", flush=True)

    def crop(d):
        act, ind = d["act"], d["ind"]; L = act.shape[0]; T = min(cli.frames, L)
        bi = torch.where(ind > 0.5)[0]
        s = int(min(max(int(bi.float().mean()) - T // 2, 0), max(L - T, 0))) if len(bi) else 0
        return act[s:s + T].to(dev), ind[s:s + T].to(dev)

    allF = []
    for fo in folds:
        tr = [t for t in tids if data[t]["fold"] != fo]
        te = [t for t in tids if data[t]["fold"] == fo]
        head = Emission(2, 2).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=cli.lr)
        for step in range(1, cli.steps_per_fold + 1):
            d = data[tr[step % len(tr)]]
            act, ind = crop(d)
            marg = dbn.forward_backward(dbn.class_logp_to_states(head(act)))
            pb = marg[:, 1].exp().clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(pb, _dilate(ind, cli.dilate))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        with torch.no_grad():
            foldF = []
            for t in te:
                d = data[t]
                bfr, _ = dbn.decode_emission(head(d["act"].to(dev)))
                foldF.append(mir_eval.beat.evaluate(d["gt"], bfr.cpu().numpy().astype(float) / FPS)["F-measure"])
            allF += foldF
            print(f"[lofo] fold {fo}: train {len(tr)} / test {len(te)} -> F={np.mean(foldF):.4f}  "
                  f"(running {len(allF)}/{len(tids)})", flush=True)
    print(f"\n[lofo] LEAVE-ONE-FOLD-OUT SMC ({len(allF)} tracks): TRAINED-on-SMC F={np.mean(allF):.4f}  "
          f"(fixed-DBN {np.mean(fx):.3f}, peak-pick {np.mean(pk):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
