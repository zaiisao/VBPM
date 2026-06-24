"""Floor experiment: does an emission parameterized AT the madmom observation hold/beat it?

Tests the hypothesis that the trained emission lost because it was a log_softmax-NORMALIZED
class posterior (no (lambda-1) balance), which the DBN -- expecting an UN-normalized
likelihood -- cannot reproduce. Three no-training decodes + a LOFO of the residual:

  1. fixed madmom DBN                         -> expect 0.594 (reference floor)
  2. FLOOR-at-init: emission = madmom_logdens(act2) + residual(feat), residual==0
                                               -> MUST equal 0.594 (validates the parameterization)
  3. POSTERIOR form: emission = log([1-act, act]) (normalized, NO (lambda-1))
                                               -> the bug: if ~0.45 it reproduces the trained failure
                                                  with NO training => parameterization, not gradients
  4. LOFO train the residual (BCE) from the floor -> does F hold >=0.594 / improve / collapse?

    python tests/dbn_floor_exp.py
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.bar_pointer_dbn import BarPointerDBN

SMC = "/home/sogang/jaehoon/Analyze-SMC"
RICH = "/home/sogang/jaehoon/CHART/cache/acts/smc_rich_heldout"
FPS, LAM = 50.0, 16


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


def _dilate(x, w):
    k = torch.ones(1, 1, 2 * w + 1, device=x.device)
    return (F.conv1d(x.view(1, 1, -1), k, padding=w)[0, 0] > 0.5).float()


def _madmom_logdens(act2):
    """UN-normalized madmom beats-only class log-densities [T,2]=(no-beat, beat)."""
    b = act2[:, 0].clamp(1e-6, 1 - 1e-6)
    return torch.stack([((1 - b) / (LAM - 1)).clamp_min(1e-6).log(), b.log()], dim=-1)


class FloorEmission(nn.Module):
    """emission = madmom log-density + residual(feat); residual zero-init -> starts AT madmom."""
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.res = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)
    def forward(self, act2, feat):
        return _madmom_logdens(act2) + self.res(feat)


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
        ind = torch.zeros(T); fr = np.round(gt * FPS).astype(int); fr = fr[(fr >= 0) & (fr < T)]; ind[fr] = 1
        data[tid] = {"feat": feat, "act2": act2, "gt": gt, "ind": ind, "fold": int(r["fold"])}
    return data


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load(); tids = sorted(data)
    in_dim = data[tids[0]]["feat"].shape[-1]
    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=None, learnable_lambda=False).to(dev)
    Fm = lambda gt, fr: mir_eval.beat.evaluate(gt, fr.cpu().numpy().astype(float) / FPS)["F-measure"]
    print(f"[floor] {len(tids)} SMC tracks, in_dim={in_dim}, DBN {dbn.num_states} states", flush=True)

    pk, fx, floor, post = [], [], [], []
    head0 = FloorEmission(in_dim).to(dev)   # residual==0 (zero-init)
    with torch.no_grad():
        for t in tids:
            d = data[t]; a2 = d["act2"].to(dev); feat = d["feat"].to(dev)
            pk.append(mir_eval.beat.evaluate(d["gt"], _peakpick(d["act2"][:, 0].numpy()))["F-measure"])
            fx.append(Fm(d["gt"], dbn.decode(a2)[0]))
            floor.append(Fm(d["gt"], dbn.decode_emission(head0(a2, feat), snap_act=a2[:, 0])[0]))
            postem = torch.stack([(1 - a2[:, 0]).clamp_min(1e-6).log(), a2[:, 0].clamp_min(1e-6).log()], -1)
            post.append(Fm(d["gt"], dbn.decode_emission(postem, snap_act=a2[:, 0])[0]))
    print(f"[floor] NO-TRAINING DECODES:")
    print(f"  peak-pick                         = {np.mean(pk):.4f}")
    print(f"  fixed madmom DBN                  = {np.mean(fx):.4f}   (reference floor)")
    print(f"  FLOOR-at-init (madmom+residual=0) = {np.mean(floor):.4f}   (should == fixed)")
    print(f"  POSTERIOR form  log([1-act,act])  = {np.mean(post):.4f}   (normalized, no (lambda-1) -> the bug?)", flush=True)

    # --- LOFO: train the residual from the floor ---
    folds = sorted(set(data[t]["fold"] for t in tids))
    def crop(d):
        L = d["feat"].shape[0]; T = min(384, L)
        bi = torch.where(d["ind"] > 0.5)[0]
        s = int(min(max(int(bi.float().mean()) - T // 2, 0), max(L - T, 0))) if len(bi) else 0
        return d["act2"][s:s+T].to(dev), d["feat"][s:s+T].to(dev), d["ind"][s:s+T].to(dev)
    allF = []
    for fo in folds:
        tr = [t for t in tids if data[t]["fold"] != fo]; te = [t for t in tids if data[t]["fold"] == fo]
        head = FloorEmission(in_dim).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        for step in range(1, 151):
            d = data[tr[step % len(tr)]]; a2, feat, ind = crop(d)
            marg = dbn.forward_backward(dbn.class_logp_to_states(head(a2, feat)))
            pb = marg[:, 1].exp().clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(pb, _dilate(ind, 3))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for t in te:
                d = data[t]; a2 = d["act2"].to(dev); feat = d["feat"].to(dev)
                allF.append(Fm(d["gt"], dbn.decode_emission(head(a2, feat), snap_act=a2[:, 0])[0]))
        print(f"[floor] LOFO fold {fo}: trained-from-floor running F={np.mean(allF):.4f} ({len(allF)}/{len(tids)})", flush=True)
    print(f"\n[floor] TRAINED-FROM-FLOOR LOFO = {np.mean(allF):.4f}  "
          f"(floor {np.mean(floor):.3f}, fixed {np.mean(fx):.3f}, peak-pick {np.mean(pk):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
