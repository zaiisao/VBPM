"""Install the truth, then let the objective decide whether to keep it.

Every "the model cannot find X" claim has a second reading: the model finds X and the
objective walks away from it. This distinguishes them. The potentials are first fitted
to the ORACLE phase path -- a spike at the true bin every frame -- which puts the
posterior on the annotated trajectory without the ELBO having been consulted. Then
training proceeds on the ELBO alone and the trajectory is watched.

  stays   -> truth is (locally) the objective's own optimum; failures are search
  leaves  -> the objective prefers something else, and no amount of search will help

The oracle path is built from annotations, so this is a diagnostic, never a read-out.
"""
from __future__ import annotations

import argparse
import importlib
import math

import numpy as np
import torch

from ..config import load_config
from ..data.dataset import load_catalog
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import continuity_scores, f_measure


def oracle_bins(crop, bins):
    """True phase, quantised to the chain's bin grid: [T] long."""
    n = len(crop["y"])
    times = crop["t0"] + np.arange(n) / crop["fps"]
    anc = np.asarray(crop["anchors"], dtype=np.float64)
    turns = 2 * np.pi * np.arange(len(anc))
    left = turns[0] + (times[0] - anc[0]) * 2 * np.pi / (anc[1] - anc[0])
    right = turns[-1] + (times[-1] - anc[-1]) * 2 * np.pi / (anc[-1] - anc[-2])
    phi = np.interp(times, anc, turns, left=left, right=right)
    return torch.tensor(np.round(np.mod(phi, 2 * np.pi) / (2 * np.pi) * bins).astype(int)
                        % bins)


def main():
    """Install the truth, train on the ELBO, and watch whether it survives."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="vbpm/configs/tchain.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--dataset", default="asap")
    p.add_argument("--song", type=int, default=355)
    p.add_argument("--fit-steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    cfg, hooks = load_config(args.config, args.set)
    cfg.epochs = args.epochs
    dev = torch.device(f"cuda:{args.gpu}")
    fe = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=f"cuda:{args.gpu}", output="features")
    song = sorted(sum(load_catalog([args.dataset]).values(), []),
                  key=lambda s: s.song_id)[args.song]
    data = ExcerptDataset([song], fe, cfg.excerpt_seconds, deterministic=True,
                          target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    raw = collate_excerpts([data[0]])
    with torch.no_grad():
        h = fe.forward_features(raw["input"])
    mask, y = raw["mask"].to(dev), raw["y"].to(dev)
    crop = {"y": raw["y"][0].numpy(), "fps": float(raw["fps"][0]),
            "t0": float(raw["t0"][0]), "anchors": np.asarray(raw["anchors"][0])}
    truth = np.asarray(raw["downbeat_times"][0])
    period = float(np.median(np.diff(crop["anchors"])))

    torch.manual_seed(0)
    model = hooks.build_model(cfg, fe.num_channels).to(dev)
    target = oracle_bins(crop, model.bins).to(dev)

    def report(tag):
        model.eval()
        with torch.no_grad():
            out = model(h, mask, y)
            path = model.infer_phase(h, mask)[0]
            wraps = torch.nonzero(torch.diff(torch.floor(path / (2 * math.pi))) > 0)[:, 0]
            times = wraps.cpu().numpy() / crop["fps"] + crop["t0"]
            err = [min(abs(d - times)) * 1000 for d in truth] if len(times) else [9e9]
            F = f_measure(times, truth)[0]
            cmlt, _a = continuity_scores(truth, times)
        model.train()
        print(f"{tag:>12} elbo {float(out['elbo'].reshape(-1)[0]):9.2f}  "
              f"recon {float(out['recon'].reshape(-1)[0]):9.2f}  "
              f"med|err| {np.median(err):6.0f}ms  in-tol {np.mean(np.asarray(err) <= 70):5.0%}  "
              f"F {F:.3f}  CMLt {cmlt:.3f}", flush=True)

    print(f"{song.song_id}  bar {period:.3f}s  {len(truth)} downbeats  "
          f"bins {model.bins}", flush=True)
    report("random init")

    opt_fit = torch.optim.Adam(list(model.encoder.parameters())
                               + list(model.psi_head.parameters()), lr=1e-3)
    for _ in range(args.fit_steps):
        opt_fit.zero_grad()
        feats = model.encoder(h, mask)
        loss = torch.nn.functional.cross_entropy(
            model.psi_head(feats)[0], target)
        loss.backward()
        opt_fit.step()
    print(f"  (potentials fitted to the oracle path, ce {float(loss):.4f})", flush=True)
    report("ORACLE start")

    opt, clip = hooks.optimizer(model, cfg)
    frames = mask.sum(1).clamp(min=1.0)
    for epoch in range(args.epochs):
        hooks.on_epoch(model, cfg, epoch)
        out = model(h, mask, y, pos_weight=cfg.pos_weight)
        loss = -(hooks.objective(out, 1.0, cfg) / frames).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip, cfg.clip)
        opt.step()
        if (epoch + 1) % args.every == 0:
            report(f"ELBO ep {epoch + 1}")


if __name__ == "__main__":
    main()
