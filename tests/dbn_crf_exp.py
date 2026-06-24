"""CRF objective: init the emission at the madmom floor, train with the STRUCTURED
likelihood (forward-algorithm), not the marginal-BCE surrogate.

CRF loss = forward_logpartition(emission) - forward_logpartition(emission masked to
GT-consistent states) = -log P(GT beat/no-beat class sequence | model). This optimizes
the JOINT path the Viterbi decodes (no marginal-vs-joint gap), which the BCE surrogate
did not -- BCE destroyed the floor (0.594 -> 0.27). Expectation: CRF holds >=0.594 and
can climb past it.

    python tests/dbn_crf_exp.py
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
    b = act2[:, 0].clamp(1e-6, 1 - 1e-6)
    return torch.stack([((1 - b) / (LAM - 1)).clamp_min(1e-6).log(), b.log()], dim=-1)


class FloorEmission(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.res = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)
    def forward(self, act2, feat):
        return _madmom_logdens(act2) + self.res(feat)


def crf_loss(dbn, emission, ind, w, C=6.0):
    """Soft-CRF: logZ(all paths) - logZ(paths SOFT-constrained to GT class sequence). A state
    whose class mismatches the GT label at a frame is penalized by C (finite, so the
    constrained lattice stays feasible even under rubato/tempo-grid mismatch)."""
    obs = dbn.class_logp_to_states(emission)                      # [T, S]
    logZ = dbn.forward_logpartition(obs)
    win = _dilate(ind, w) > 0.5                                   # [T] beat window
    beat_states = (dbn.obs_ptr >= 1)                              # [S]
    T = obs.shape[0]
    allowed = torch.where(win[:, None], beat_states[None, :].expand(T, -1),
                          (~beat_states)[None, :].expand(T, -1))  # [T, S]
    obs_gt = obs + torch.where(allowed, obs.new_zeros(()), obs.new_full((), -C))
    logZ_gt = dbn.forward_logpartition(obs_gt)
    return (logZ - logZ_gt) / T


def _load():
    GTd = SMC + "/beat_this_annotations/smc/annotations/beats"
    data = {}
    for f in sorted(glob.glob(RICH + "/*.pt")):
        r = torch.load(f, map_location="cpu"); tid = r["tid"]
        gt = None
        for nm in (tid, tid.upper()):
            p = os.path.join(GTd, nm + ".beats")
            if os.path.exists(p):
                d = np.loadtxt(p); gt = d if d.ndim == 1 else d[:, 1]; break
        if gt is None or len(gt) < 2:
            continue
        feat, act2 = r["feat"].float(), r["act2"].float(); T = feat.shape[0]
        ind = torch.zeros(T); fr = np.round(gt * FPS).astype(int); fr = fr[(fr >= 0) & (fr < T)]; ind[fr] = 1
        data[tid] = {"feat": feat, "act2": act2, "gt": gt, "ind": ind, "fold": int(r["fold"])}
    return data


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--leash", type=float, default=0.0, help="L2 pull of the residual toward 0 (madmom floor)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--W", type=int, default=3)
    cli = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = cli.W
    data = _load(); tids = sorted(data); in_dim = data[tids[0]]["feat"].shape[-1]
    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=None, learnable_lambda=False).to(dev)
    Fm = lambda gt, fr: mir_eval.beat.evaluate(gt, fr.cpu().numpy().astype(float) / FPS)["F-measure"]
    print(f"[crf] {len(tids)} SMC tracks, in_dim={in_dim}, DBN {dbn.num_states} states, window=+/-{W}", flush=True)

    pk, fx, floor = [], [], []
    head0 = FloorEmission(in_dim).to(dev)
    with torch.no_grad():
        for t in tids:
            d = data[t]; a2 = d["act2"].to(dev); feat = d["feat"].to(dev)
            pk.append(mir_eval.beat.evaluate(d["gt"], _peakpick(d["act2"][:, 0].numpy()))["F-measure"])
            fx.append(Fm(d["gt"], dbn.decode(a2)[0]))
            floor.append(Fm(d["gt"], dbn.decode_emission(head0(a2, feat), snap_act=a2[:, 0])[0]))
    print(f"[crf] REFERENCES: peak-pick={np.mean(pk):.4f}  fixed-DBN={np.mean(fx):.4f}  floor-at-init={np.mean(floor):.4f}", flush=True)

    def crop(d):
        L = d["feat"].shape[0]; T = min(384, L)
        bi = torch.where(d["ind"] > 0.5)[0]
        s = int(min(max(int(bi.float().mean()) - T // 2, 0), max(L - T, 0))) if len(bi) else 0
        return d["act2"][s:s+T].to(dev), d["feat"][s:s+T].to(dev), d["ind"][s:s+T].to(dev)

    folds = sorted(set(data[t]["fold"] for t in tids))
    allF = []
    for fo in folds:
        tr = [t for t in tids if data[t]["fold"] != fo]; te = [t for t in tids if data[t]["fold"] == fo]
        head = FloorEmission(in_dim).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=cli.lr)
        for step in range(1, cli.steps + 1):
            d = data[tr[step % len(tr)]]; a2, feat, ind = crop(d)
            em = head(a2, feat)
            loss = crf_loss(dbn, em, ind, W) + cli.leash * (em - _madmom_logdens(a2)).pow(2).mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for t in te:
                d = data[t]; a2 = d["act2"].to(dev); feat = d["feat"].to(dev)
                allF.append(Fm(d["gt"], dbn.decode_emission(head(a2, feat), snap_act=a2[:, 0])[0]))
        print(f"[crf] LOFO fold {fo}: CRF-trained-from-floor running F={np.mean(allF):.4f} ({len(allF)}/{len(tids)})", flush=True)
    print(f"\n[crf] CRF-TRAINED-FROM-FLOOR LOFO = {np.mean(allF):.4f}  "
          f"(floor {np.mean(floor):.3f}, fixed {np.mean(fx):.3f}, peak-pick {np.mean(pk):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
