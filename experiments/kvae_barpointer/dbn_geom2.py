"""Geometric bar-pointer via the differentiable DBN -- consolidated:
  * learned emission head trained THROUGH the DBN forward-backward (tempo-phase HMM),
  * eval on BOTH mir_eval (da.fmeas, == the M1/M2 metric, comparable to M1's 0.878) AND the
    evaluation/score metric (== tests/train_dbn.py, comparable to its 0.971/peak 0.984),
  * --learn_lambda: make the tempo-transition lambda an nn.Parameter (the user's dynamic-lambda Q,
    global version) and report how far it moves from madmom's init=100,
  * saves the model for OOD eval.
Deploy = Viterbi over the learned emission (the faithful geometric inference).
"""
import sys, glob, math, random, argparse, importlib.util
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = "/home/sogang/jaehoon/CHART"; sys.path.insert(0, ROOT)
from models.bar_pointer_dbn import BarPointerDBN
from evaluation.score import evaluate_beats, frames_to_beat_times
da = importlib.util.module_from_spec(importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py"))
importlib.util.spec_from_file_location("da", f"{ROOT}/experiments/diagram_arch/run.py").loader.exec_module(da)
fmeas = da.fmeas
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Emission(nn.Module):
    def __init__(self, in_dim, C, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, C))
    def forward(self, h): return torch.log_softmax(self.net(h), -1)


def _load(cache, n): return [torch.load(f, map_location="cpu") for f in sorted(glob.glob(cache + "/*.pt"))[:n]]
def _dilate(x, w):
    k = torch.ones(1, 1, 2 * w + 1, device=x.device)
    return (F.conv1d(x.view(1, 1, -1), k, padding=w)[0, 0] > 0.5).float()
def _crop(rec, T, dev):
    act = rec["activations"].float().to(dev); bt = rec["beat_targets"].float().to(dev); db = rec["downbeat_targets"].float().to(dev)
    L = act.shape[0]; bi = torch.where(bt > 0.5)[0]
    s = min(max(int(bi.float().mean()) - T // 2, 0), max(L - T, 0)) if len(bi) else max(0, (L - T) // 2)
    e = min(s + T, L); return act[s:e], bt[s:e], db[s:e]


@torch.no_grad()
def score(dbn, head, val, fps, maxT):
    """Both metrics + peak-pick ceiling, on the SAME songs."""
    lb_mir, ld_mir, lb_sc, peak_mir = [], [], [], []
    for r in val:
        act = r["activations"].float().to(DEV)[:maxT]; a2 = r["act2"].float().to(DEV)[:maxT]
        bt = r["beat_targets"].numpy()[:maxT]; db = r["downbeat_targets"].numpy()[:maxT]
        ref = np.where(bt > 0.5)[0] / fps; refd = np.where(db > 0.5)[0] / fps
        if len(ref) < 2: continue
        bfr, dfr = dbn.decode_emission(head(act))
        est = bfr.cpu().numpy().astype(float) / fps
        lb_mir.append(fmeas(ref, est))
        lb_sc.append(evaluate_beats(frames_to_beat_times(bt, fps), est)["F-measure"])
        if len(refd) >= 2 and not dbn.beats_only:
            ld_mir.append(fmeas(refd, dfr.cpu().numpy().astype(float) / fps))
        peak_mir.append(fmeas(ref, da.peaks(a2[:, 0].cpu().numpy())))
    m = lambda x: float(np.nanmean(x)) if x else float("nan")
    return m(lb_mir), m(ld_mir), m(lb_sc), m(peak_mir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=700); p.add_argument("--num_beats", type=int, default=4)
    p.add_argument("--num_intervals", type=int, default=40); p.add_argument("--frames", type=int, default=384)
    p.add_argument("--max_train", type=int, default=400); p.add_argument("--max_val", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--dilate", type=int, default=3)
    p.add_argument("--learn_lambda", action="store_true"); p.add_argument("--save", default="")
    cli = p.parse_args()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    train = _load("cache/acts/bt_train_rich", cli.max_train); val = _load("cache/acts/bt_val_rich", cli.max_val)
    fps = float(train[0]["fps"]); in_dim = train[0]["activations"].shape[-1]
    dbn = BarPointerDBN(fps=fps, num_beats=cli.num_beats, num_intervals=cli.num_intervals,
                        learnable_lambda=cli.learn_lambda).to(DEV)
    head = Emission(in_dim, dbn.num_classes).to(DEV)
    params = list(head.parameters()) + (list(dbn.parameters()) if cli.learn_lambda else [])
    opt = torch.optim.AdamW(params, lr=cli.lr)
    print(f"GEOM-DBN2 | learn_lambda={cli.learn_lambda} | {dbn.num_states} states C={dbn.num_classes} "
          f"in_dim={in_dim} train={len(train)} val={len(val)} fps={fps:.2f}", flush=True)

    for step in range(1, cli.steps + 1):
        r = train[step % len(train)]; act, bt, db = _crop(r, cli.frames, DEV)
        marg = dbn.forward_backward(dbn.class_logp_to_states(head(act)))
        if dbn.num_classes == 3:
            pb = torch.logsumexp(marg[:, 1:], 1).exp().clamp(1e-6, 1 - 1e-6); pd = marg[:, 2].exp().clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(pb, _dilate(bt, cli.dilate)) + F.binary_cross_entropy(pd, _dilate(db, cli.dilate))
        else:
            pb = marg[:, 1].exp().clamp(1e-6, 1 - 1e-6); loss = F.binary_cross_entropy(pb, _dilate(bt, cli.dilate))
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 100 == 0 or step == cli.steps:
            lbm, ldm, lbs, pk = score(dbn, head, val, fps, 1500)
            lam = ""
            if cli.learn_lambda:
                with torch.no_grad(): lam = f" | lambda={dbn._lam.mean().item():.1f}(init100)"
            print(f"step {step:04d} | loss {loss.item():.3f} | LEARNED-DBN  mir: beat {lbm:.3f} db {ldm:.3f}"
                  f" | score-metric beat {lbs:.3f} | peak(mir) {pk:.3f}{lam}", flush=True)
    if cli.save:
        torch.save({"head": head.state_dict(), "in_dim": in_dim, "num_beats": cli.num_beats,
                    "num_intervals": cli.num_intervals, "learn_lambda": cli.learn_lambda,
                    "dbn_lambda": (dbn.log_lambda.detach().cpu() if cli.learn_lambda else None)}, cli.save)
        print(f"[saved] {cli.save}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
