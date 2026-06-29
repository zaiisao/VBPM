"""Per-song dynamic-lambda (the FULL version of the user's idea): a small head reads each song's
pooled features and emits that song's transition-lambda; the differentiable DBN is trained/decoded
with the per-song edge log-probs (elp). Compares to fixed-madmom (0.948) and global-learnable (0.961).
Tests whether amortizing the DBN's tempo-stiffness hyperparameter PER SONG helps.
"""
import sys, glob, random, argparse, importlib.util, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
from models.bar_pointer_dbn import BarPointerDBN
from evaluation.score import evaluate_beats, frames_to_beat_times
sp = importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py")
da = importlib.util.module_from_spec(sp); sp.loader.exec_module(da); fmeas = da.fmeas
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG100 = math.log(100.0)


class Emission(nn.Module):
    def __init__(self, i, C, h=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, C))
    def forward(self, x): return torch.log_softmax(self.net(x), -1)


class LambdaHead(nn.Module):
    """pooled song features -> per-beat log_lambda in [log100-2.5, log100+2.5] (lambda ~ 8..1200)."""
    def __init__(self, i, num_beats, h=64):
        super().__init__(); self.num_beats = num_beats
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, num_beats))
    def forward(self, pooled):
        return LOG100 + 2.5 * torch.tanh(self.net(pooled))            # [num_beats]


def _load(c, n): return [torch.load(f, map_location="cpu") for f in sorted(glob.glob(c + "/*.pt"))[:n]]
def _dilate(x, w):
    k = torch.ones(1, 1, 2 * w + 1, device=x.device); return (F.conv1d(x.view(1, 1, -1), k, padding=w)[0, 0] > 0.5).float()
def _crop(r, T, dev):
    act = r["activations"].float().to(dev); bt = r["beat_targets"].float().to(dev); db = r["downbeat_targets"].float().to(dev)
    L = act.shape[0]; bi = torch.where(bt > 0.5)[0]
    s = min(max(int(bi.float().mean()) - T // 2, 0), max(L - T, 0)) if len(bi) else max(0, (L - T) // 2)
    return act[s:s + T], bt[s:s + T], db[s:s + T]


@torch.no_grad()
def score(dbn, head, lam_head, val, fps, maxT):
    lb, ld, lams = [], [], []
    for r in val:
        act = r["activations"].float().to(DEV)[:maxT]
        bt = r["beat_targets"].numpy()[:maxT]; db = r["downbeat_targets"].numpy()[:maxT]
        ref = np.where(bt > 0.5)[0] / fps; refd = np.where(db > 0.5)[0] / fps
        if len(ref) < 2: continue
        llam = lam_head(act.mean(0)); elp = dbn._edge_logp(llam); lams.append(float(llam.exp().mean()))
        bfr, dfr = dbn.decode_emission(head(act), elp=elp)
        lb.append(fmeas(ref, bfr.cpu().numpy().astype(float) / fps))
        if len(refd) >= 2 and not dbn.beats_only:
            ld.append(fmeas(refd, dfr.cpu().numpy().astype(float) / fps))
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(lb), m(ld), m(lams), (float(np.std(lams)) if lams else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=700); p.add_argument("--num_beats", type=int, default=4)
    p.add_argument("--num_intervals", type=int, default=40); p.add_argument("--frames", type=int, default=384)
    p.add_argument("--max_train", type=int, default=400); p.add_argument("--max_val", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--dilate", type=int, default=3)
    cli = p.parse_args(); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = _load("cache/acts/bt_train_rich", cli.max_train); val = _load("cache/acts/bt_val_rich", cli.max_val)
    fps = float(train[0]["fps"]); in_dim = train[0]["activations"].shape[-1]
    dbn = BarPointerDBN(fps=fps, num_beats=cli.num_beats, num_intervals=cli.num_intervals).to(DEV)
    head = Emission(in_dim, dbn.num_classes).to(DEV); lam_head = LambdaHead(in_dim, cli.num_beats).to(DEV)
    opt = torch.optim.AdamW(list(head.parameters()) + list(lam_head.parameters()), lr=cli.lr)
    print(f"PER-SONG-LAMBDA | {dbn.num_states} states C={dbn.num_classes} in_dim={in_dim} "
          f"train={len(train)} val={len(val)} fps={fps:.2f}", flush=True)
    for step in range(1, cli.steps + 1):
        r = train[step % len(train)]; act, bt, db = _crop(r, cli.frames, DEV)
        llam = lam_head(act.mean(0)); elp = dbn._edge_logp(llam)
        marg = dbn.forward_backward(dbn.class_logp_to_states(head(act)), elp=elp)
        if dbn.num_classes == 3:
            pb = torch.logsumexp(marg[:, 1:], 1).exp().clamp(1e-6, 1 - 1e-6); pd = marg[:, 2].exp().clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(pb, _dilate(bt, cli.dilate)) + F.binary_cross_entropy(pd, _dilate(db, cli.dilate))
        else:
            pb = marg[:, 1].exp().clamp(1e-6, 1 - 1e-6); loss = F.binary_cross_entropy(pb, _dilate(bt, cli.dilate))
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(lam_head.parameters()), 1.0); opt.step()
        if step % 100 == 0 or step == cli.steps:
            lb, ld, lam_mean, lam_std = score(dbn, head, lam_head, val, fps, 1500)
            print(f"step {step:04d} | loss {loss.item():.3f} | PERSONG-LAMBDA-DBN mir: beat {lb:.3f} db {ld:.3f}"
                  f" | per-song lambda mean {lam_mean:.1f} std {lam_std:.1f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
