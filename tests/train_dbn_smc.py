"""Trainable likelihood + fixed bar-pointer DBN prior, evaluated LEAK-FREE on SMC.

Train the emission (likelihood) on our 4 datasets' [T,2] Beat-This activations
(SMC is never in training), decode with the fixed madmom DBN prior, and score on
the HELD-OUT SMC activations (Analyze-SMC's per-fold cache, 50 fps) -- the exact
set that gives peak-pick 0.627. Tests: can a TRAINED likelihood beat the FIXED
madmom likelihood (0.594) and peak-pick (0.627) on SMC?

    python tests/train_dbn_smc.py --steps 400
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

SMC = "/home/sogang/jaehoon/Analyze-SMC"
SMC_FPS = 50.0
TRAIN_FPS = 22050 / 256


class Emission(nn.Module):
    """[T,2] activation -> per-class log-potential [T,2] (no-beat, beat). Per-frame; the
    DBN supplies the dynamics. This is the trainable likelihood; the prior is fixed."""
    def __init__(self, in_dim=2, C=2, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, C))
    def forward(self, h):
        return torch.log_softmax(self.net(h), dim=-1)


def _dilate(x, w):
    k = torch.ones(1, 1, 2 * w + 1, device=x.device)
    return (F.conv1d(x.view(1, 1, -1), k, padding=w)[0, 0] > 0.5).float()


def _peakpick(prob, fps=SMC_FPS, thresh=0.5, width=7):
    t = torch.from_numpy(np.ascontiguousarray(prob)).float().unsqueeze(0)
    peaks = t.masked_fill(t != F.max_pool1d(t, width, 1, width // 2), -1000.0)
    fr = torch.nonzero(peaks.squeeze(0) > thresh).numpy()[:, 0]
    if len(fr):
        keep = [fr[0]]
        for x in fr[1:]:
            if x - keep[-1] > 1:
                keep.append(x)
        fr = np.array(keep)
    return fr / fps


def _load_smc():
    ACT, GT = SMC + "/beat_this_activations_cache", SMC + "/beat_this_annotations/smc/annotations/beats"
    out = []
    for f in sorted(glob.glob(ACT + "/*.npz")):
        tid = os.path.splitext(os.path.basename(f))[0]
        z = np.load(f)
        beat, down = expit(z["beat"].astype(np.float64)), expit(z["downbeat"].astype(np.float64))
        gt = None
        for nm in (tid, tid.upper()):
            p = os.path.join(GT, nm + ".beats")
            if os.path.exists(p):
                d = np.loadtxt(p); gt = d if d.ndim == 1 else d[:, 1]; break
        if gt is None or len(gt) < 2:
            continue
        out.append({"act": torch.tensor(np.stack([beat, down], -1), dtype=torch.float32), "gt": gt})
    return out


def _load_train(cache, n):
    recs = []
    for f in sorted(glob.glob(cache + "/*.pt"))[:n]:
        r = torch.load(f, map_location="cpu")
        if "act2" not in r:
            continue
        recs.append({"act": r["act2"].float(), "bt": r["beat_targets"].float()})
    return recs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", default="cache/acts/bt_train_rich")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--frames", type=int, default=384)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max_train", type=int, default=800)
    p.add_argument("--dilate", type=int, default=3)
    p.add_argument("--train_intervals", type=int, default=50)
    p.add_argument("--anchor_weight", type=float, default=0.0,
                   help="MSE pull of the emission toward madmom's fixed log-activation observation")
    cli = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = _load_train(cli.train_cache, cli.max_train)
    smc = _load_smc()
    dbn_tr = BarPointerDBN(fps=TRAIN_FPS, beats_only=True, num_intervals=cli.train_intervals,
                           learnable_lambda=False).to(dev)
    dbn_ev = BarPointerDBN(fps=SMC_FPS, beats_only=True, num_intervals=None,
                           learnable_lambda=False).to(dev)   # madmom-exact eval prior
    head = Emission(2, 2).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=cli.lr)
    print(f"[dbn_smc] train={len(train)} (4-ds [T,2]) | SMC heldout={len(smc)} | "
          f"train-DBN {dbn_tr.num_states} st @ {TRAIN_FPS:.1f}fps | eval-DBN {dbn_ev.num_states} st @ 50fps")

    # --- references on held-out SMC ---
    pk, fx = [], []
    with torch.no_grad():
        for s in smc:
            a = s["act"].to(dev)
            pk.append(mir_eval.beat.evaluate(s["gt"], _peakpick(s["act"][:, 0].numpy()))["F-measure"])
            bfr, _ = dbn_ev.decode(a)
            fx.append(mir_eval.beat.evaluate(s["gt"], bfr.cpu().numpy().astype(float) / SMC_FPS)["F-measure"])
    print(f"[dbn_smc] REFERENCES: peak-pick={np.mean(pk):.4f}  fixed-madmom-DBN={np.mean(fx):.4f}\n")

    def crop(r):
        act, bt = r["act"], r["bt"]
        L = act.shape[0]; T = min(cli.frames, L)
        bi = torch.where(bt > 0.5)[0]
        s = int(min(max(int(bi.float().mean()) - T // 2, 0), max(L - T, 0))) if len(bi) else 0
        return act[s:s + T].to(dev), bt[s:s + T].to(dev)

    @torch.no_grad()
    def eval_smc():
        Fs = []
        for s in smc:
            cl = head(s["act"].to(dev))
            bfr, _ = dbn_ev.decode_emission(cl)
            Fs.append(mir_eval.beat.evaluate(s["gt"], bfr.cpu().numpy().astype(float) / SMC_FPS)["F-measure"])
        return float(np.mean(Fs))

    for step in range(1, cli.steps + 1):
        r = train[step % len(train)]
        act, bt = crop(r)
        cl = head(act)                                                          # [T,2] log_softmax
        marg = dbn_tr.forward_backward(dbn_tr.class_logp_to_states(cl))         # [T,2] log
        p_beat = marg[:, 1].exp().clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(p_beat, _dilate(bt, cli.dilate))
        if cli.anchor_weight > 0:                                               # anchor to madmom obs
            a = act[:, 0].clamp(1e-6, 1 - 1e-6)
            tgt = torch.log_softmax(torch.stack([((1 - a) / 15.0).log(), a.log()], -1), -1)
            loss = loss + cli.anchor_weight * F.mse_loss(cl, tgt.detach())
        if not torch.isfinite(loss):
            print(f"[dbn_smc] non-finite step {step}, skip"); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if step % cli.eval_every == 0 or step == cli.steps:
            print(f"[dbn_smc] step {step:04d} | loss={loss.item():.3f} | "
                  f"TRAINED-likelihood DBN on SMC = {eval_smc():.4f}  "
                  f"(fixed-DBN {np.mean(fx):.3f}, peak-pick {np.mean(pk):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
