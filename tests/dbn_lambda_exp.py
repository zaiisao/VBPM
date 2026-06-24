"""Per-song trainable lambdas: a NN reads each song and emits its transition-lambda and
observation-bias, trained end-to-end through the beats -- the thing madmom's fixed,
hand-built DBN structurally cannot do.

Emission is PINNED to the madmom observation (+ a per-song beat bias); the ONLY trainable
thing is the lambda-head. It is zero-initialized so it starts EXACTLY at the fixed-lambda
floor (0.593); any change is the adaptive lambda earning its keep. Trained with the CRF
loss, evaluated leave-one-fold-out on held-out SMC.

    python tests/dbn_lambda_exp.py
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
FPS, LAM, INIT_LAM = 50.0, 16, 100.0


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


class LambdaHead(nn.Module):
    """pooled features -> [log lambda_trans (per beat), beat_bias]. Zero-init final layer so it
    starts at lambda=INIT_LAM, bias=0 (== the fixed-lambda madmom floor)."""
    def __init__(self, in_dim, num_beats=1, hid=128):
        super().__init__()
        self.num_beats = num_beats
        self.net = nn.Sequential(nn.Linear(2 * in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, num_beats + 1))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias[:num_beats] = float(np.log(INIT_LAM))
            self.net[-1].bias[num_beats] = 0.0
    def forward(self, feat):                                   # feat [T, in_dim]
        pooled = torch.cat([feat.mean(0), feat.std(0)])
        out = self.net(pooled)
        return out[:self.num_beats], out[self.num_beats]       # log_lam [num_beats], bias scalar


def _emission(a2, bias):
    em = _madmom_logdens(a2)                                   # [T,2]
    return em + torch.stack([torch.zeros_like(a2[:, 0]), bias.expand(a2.shape[0])], dim=-1)


def crf_loss(dbn, em, ind, w, elp, C=6.0):
    obs = dbn.class_logp_to_states(em)
    logZ = dbn.forward_logpartition(obs, elp=elp)
    win = _dilate(ind, w) > 0.5; beat_states = (dbn.obs_ptr >= 1); T = obs.shape[0]
    allowed = torch.where(win[:, None], beat_states[None, :].expand(T, -1),
                          (~beat_states)[None, :].expand(T, -1))
    obs_gt = obs + torch.where(allowed, obs.new_zeros(()), obs.new_full((), -C))
    logZ_gt = dbn.forward_logpartition(obs_gt, elp=elp)
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
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--W", type=int, default=3)
    cli = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load(); tids = sorted(data); in_dim = data[tids[0]]["feat"].shape[-1]
    dbn = BarPointerDBN(fps=FPS, beats_only=True, num_intervals=None, learnable_lambda=False).to(dev)
    Fm = lambda gt, fr: mir_eval.beat.evaluate(gt, fr.cpu().numpy().astype(float) / FPS)["F-measure"]
    print(f"[lam] {len(tids)} SMC tracks, in_dim={in_dim}, DBN {dbn.num_states} states", flush=True)

    pk, fx = [], []
    with torch.no_grad():
        for t in tids:
            d = data[t]; a2 = d["act2"].to(dev)
            pk.append(mir_eval.beat.evaluate(d["gt"], _peakpick(d["act2"][:, 0].numpy()))["F-measure"])
            fx.append(Fm(d["gt"], dbn.decode(a2)[0]))
    print(f"[lam] REFERENCES: peak-pick={np.mean(pk):.4f}  fixed-lambda-DBN={np.mean(fx):.4f}", flush=True)

    def crop_idx(d):
        L = d["act2"].shape[0]; T = min(384, L)
        bi = torch.where(d["ind"] > 0.5)[0]
        s = int(min(max(int(bi.float().mean()) - T // 2, 0), max(L - T, 0))) if len(bi) else 0
        return s, s + T

    folds = sorted(set(data[t]["fold"] for t in tids))
    allF, lams = [], []
    for fo in folds:
        tr = [t for t in tids if data[t]["fold"] != fo]; te = [t for t in tids if data[t]["fold"] == fo]
        head = LambdaHead(in_dim).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=cli.lr)
        for step in range(1, cli.steps + 1):
            d = data[tr[step % len(tr)]]
            log_lam, bias = head(d["feat"].to(dev))                 # per-song lambda from FULL features
            elp = dbn._edge_logp(log_lambda=log_lam)
            s, e = crop_idx(d)
            a2 = d["act2"][s:e].to(dev); ind = d["ind"][s:e].to(dev)
            loss = crf_loss(dbn, _emission(a2, bias), ind, cli.W, elp)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for t in te:
                d = data[t]; a2 = d["act2"].to(dev)
                log_lam, bias = head(d["feat"].to(dev))
                elp = dbn._edge_logp(log_lambda=log_lam)
                bfr, _ = dbn.decode_emission(_emission(a2, bias), snap_act=a2[:, 0], elp=elp)
                allF.append(Fm(d["gt"], bfr[0] if isinstance(bfr, tuple) else bfr))
                lams.append(float(log_lam.exp().mean()))
        print(f"[lam] fold {fo}: per-song-lambda running F={np.mean(allF):.4f} "
              f"(lambda range {np.min(lams):.0f}-{np.max(lams):.0f}, {len(allF)}/{len(tids)})", flush=True)
    print(f"\n[lam] PER-SONG-LAMBDA LOFO = {np.mean(allF):.4f}  "
          f"(fixed-lambda {np.mean(fx):.3f}, peak-pick {np.mean(pk):.3f})  "
          f"| learned lambda: mean={np.mean(lams):.1f} range {np.min(lams):.0f}-{np.max(lams):.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
