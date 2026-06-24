"""RICH held-out SMC: trainable likelihood on the 512-dim features (leave-one-fold-out).

Uses the held-out rich features (cache/acts/smc_rich_heldout, from per-fold Beat This
checkpoints). For each fold, train the emission (512 -> beat class) on the other 7
folds, decode with the fixed bar-pointer DBN prior, eval the held-out fold. This is
the proper test: rich input (real headroom) + leak-free SMC. Beats to clear:
fixed-DBN 0.594, peak-pick 0.627.

    python tests/dbn_rich_lofo.py --steps_per_fold 250
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.bar_pointer_dbn import BarPointerDBN

SMC = "/home/sogang/jaehoon/Analyze-SMC"
RICH = "/home/sogang/jaehoon/CHART/cache/acts/smc_rich_heldout"
FPS = 50.0


class Emission(nn.Module):
    def __init__(self, in_dim, C=2, hid=128):
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


def _load():
    GTd = SMC + "/beat_this_annotations/smc/annotations/beats"
    data = {}
    for f in sorted(glob.glob(RICH + "/*.pt")):
        r = torch.load(f, map_location="cpu")
        tid = r["tid"]
        gt = None
        for nm in (tid, tid.upper()):
            p = os.path.join(GTd, nm + ".beats")
            if os.path.exists(p):
                d = np.loadtxt(p); gt = d if d.ndim == 1 else d[:, 1]; break
        if gt is None or len(gt) < 2:
            continue
        feat, act2 = r["feat"].float(), r["act2"].float()
        T = feat.shape[0]
        ind = torch.zeros(T)
        fr = np.round(gt * FPS).astype(int); fr = fr[(fr >= 0) & (fr < T)]; ind[fr] = 1
        data[tid] = {"feat": feat, "act2": act2, "gt": gt, "ind": ind, "fold": int(r["fold"])}
    return data


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps_per_fold", type=int, default=250)
    p.add_argument("--frames", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dilate", type=int, default=3)
    cli = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = _load()
    tids = sorted(data); folds = sorted(set(data[t]["fold"] for t in tids))
    in_dim = data[tids[0]]["feat"].shape[-1]
    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=None, learnable_lambda=False).to(dev)
    print(f"[rich-lofo] {len(tids)} SMC tracks, in_dim={in_dim}, {len(folds)} folds, DBN {dbn.num_states} states", flush=True)

    pk, fx = [], []
    with torch.no_grad():
        for t in tids:
            d = data[t]
            pk.append(mir_eval.beat.evaluate(d["gt"], _peakpick(d["act2"][:, 0].numpy()))["F-measure"])
            bfr, _ = dbn.decode(d["act2"].to(dev))
            fx.append(mir_eval.beat.evaluate(d["gt"], bfr.cpu().numpy().astype(float) / FPS)["F-measure"])
    print(f"[rich-lofo] REFERENCES (on derived act2): peak-pick={np.mean(pk):.4f} fixed-DBN={np.mean(fx):.4f}  "
          f"(official cached: 0.627 / 0.594)", flush=True)

    def crop(d):
        feat, ind = d["feat"], d["ind"]; L = feat.shape[0]; T = min(cli.frames, L)
        bi = torch.where(ind > 0.5)[0]
        s = int(min(max(int(bi.float().mean()) - T // 2, 0), max(L - T, 0))) if len(bi) else 0
        return feat[s:s + T].to(dev), ind[s:s + T].to(dev)

    allF_em, allF_act = [], []
    for fo in folds:
        tr = [t for t in tids if data[t]["fold"] != fo]
        te = [t for t in tids if data[t]["fold"] == fo]
        head = Emission(in_dim, 2).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=cli.lr)
        for step in range(1, cli.steps_per_fold + 1):
            d = data[tr[step % len(tr)]]
            feat, ind = crop(d)
            marg = dbn.forward_backward(dbn.class_logp_to_states(head(feat)))
            pb = marg[:, 1].exp().clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(pb, _dilate(ind, cli.dilate))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for t in te:
                d = data[t]; cl = head(d["feat"].to(dev)); a2 = d["act2"].to(dev)
                be, _ = dbn.decode_emission(cl)                          # snap to emission prob
                ba, _ = dbn.decode_emission(cl, snap_act=a2[:, 0])       # snap to sharp activation
                allF_em.append(mir_eval.beat.evaluate(d["gt"], be.cpu().numpy().astype(float) / FPS)["F-measure"])
                allF_act.append(mir_eval.beat.evaluate(d["gt"], ba.cpu().numpy().astype(float) / FPS)["F-measure"])
        print(f"[rich-lofo] fold {fo}: snap-emission F={np.mean(allF_em):.4f} | snap-activation F={np.mean(allF_act):.4f}  "
              f"({len(allF_em)}/{len(tids)})", flush=True)
    print(f"\n[rich-lofo] RICH LOFO SMC ({len(allF_em)} tracks): TRAINED-likelihood "
          f"snap-emission={np.mean(allF_em):.4f}  snap-activation={np.mean(allF_act):.4f}  "
          f"(fixed-DBN {np.mean(fx):.3f}, peak-pick {np.mean(pk):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
