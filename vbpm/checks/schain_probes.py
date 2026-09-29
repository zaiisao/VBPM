"""E1 and E2: is the chain load-bearing, or a smoother on a per-frame classifier?

E1 asks whether the learned potential is a rank-one encoding of a per-frame downbeat
probability. For this generative model the ELBO-optimal potential is

    log psi*_t(k) = ll_neg(k) + p_t (ll_pos(k) - ll_neg(k)) + const,   p_t = E[y_t | x]

one scalar per frame against a fixed pair of shape vectors. If the trained potential
lies on that line, the chain cannot add information the per-frame term lacks.

E2 is the ablation the earlier results table was mislabelled as. Both read-outs in that
table ran forward-backward; the only thing that differed was the estimator. Here the
rate and the peak-picker period are PINNED across every cell, so the two axes separate:
posterior (chain marginals vs the potentials alone) crossed with estimator (the mean
phase path vs the emission read through the marginals).
"""
from __future__ import annotations

import argparse
import importlib
import math

import numpy as np
import torch

from ..config import load_config
from ..variants.base import load_model_state
from ..data.dataset import split_songs
from ..data.excerpts import ExcerptDataset, collate_excerpts
from ..scoring.evaluation import (continuity_scores, f_measure, peak_times,
                                  scoring_records)


def load(checkpoint: str, config: str, gpu: int):
    """Config, model, frontend and device for a saved checkpoint."""
    cfg, hooks = load_config(config, [])
    device = torch.device(f"cuda:{gpu}")
    frontend = importlib.import_module(f"vbpm.data.frontends.{cfg.frontend}").FRONTEND(
        checkpoint=cfg.frontend_checkpoint, device=f"cuda:{gpu}", output="features")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = hooks.build_model(cfg, frontend.num_channels).to(device)
    load_model_state(model, state["model"])
    model.eval()
    if "frontend" in state:
        frontend._audio2frames.model.load_state_dict(state["frontend"])
    return cfg, model, frontend, device


def crops(cfg, frontend, songs, batch_size: int = 8):
    """Deterministic scoring records for these songs, one per song."""
    dataset = ExcerptDataset(songs, frontend, cfg.excerpt_seconds, deterministic=True,
                             target_tol_frames=getattr(cfg, "target_tol_frames", 0))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=collate_excerpts)
    for raw in loader:
        yield raw, scoring_records(raw)


def e1_rank_one(cfg, model, frontend, device, songs):
    """Regress log psi onto the ELBO-optimal one-parameter family. R^2 near 1 means the

    potential carries a single scalar per frame and the chain is a post-process.
    """
    e = model.emission_logits_at_bins().detach()
    ll_pos = -torch.nn.functional.softplus(-e)
    ll_neg = -torch.nn.functional.softplus(e)
    basis = torch.stack([ll_neg, ll_pos - ll_neg, torch.ones_like(e)], dim=1)
    control = torch.stack([torch.cos(model.theta), torch.sin(model.theta),
                           torch.ones_like(e)], dim=1)

    scores, controls, slopes = [], [], []
    with torch.no_grad():
        for raw, records in crops(cfg, model, songs) if False else crops(cfg, frontend, songs):
            h = frontend.forward_features(raw["input"])
            mask = raw["mask"].to(device)
            feats = model.encoder(h, mask)
            log_psi = torch.log_softmax(model.psi_head(feats), dim=-1)
            for i, rec in enumerate(records):
                if rec is None:
                    continue
                t = len(rec["y"])
                target = log_psi[i, :t]
                for design, out in ((basis, scores), (control, controls)):
                    sol = torch.linalg.lstsq(design.double(), target.T.double()).solution
                    fit = (design.double() @ sol).T
                    resid = ((target.double() - fit) ** 2).sum()
                    total = ((target.double() - target.double().mean(1, keepdim=True)) ** 2).sum()
                    out.append(float(1.0 - resid / total.clamp(min=1e-12)))
                    if out is scores:
                        slopes.append(sol[1].cpu().numpy())
    return np.array(scores), np.array(controls), slopes


def viterbi(model, log_psi, log_T, mask, rate_idx):
    """MAP path over bins for the selected rate, a diagnostic ceiling.

    The tutorial's design is amortized rather than Viterbi (section 8.1.6), so this is a ceiling, not a
    deployment read-out.
    """
    b, t, k = log_psi.shape
    lt = log_T[rate_idx]
    delta = torch.full((b, k), -math.log(k), device=log_psi.device) + log_psi[:, 0]
    back = []
    for i in range(1, t):
        scores = delta[:, :, None] + lt
        best, arg = scores.max(dim=1)
        delta = best + mask[:, i][:, None] * log_psi[:, i]
        back.append(arg)
    path = [delta.argmax(1)]
    for arg in reversed(back):
        path.append(arg.gather(1, path[-1][:, None])[:, 0])
    return torch.stack(list(reversed(path)), dim=1)


def unwrap_bins(model, idx):
    """Bin indices to an unwrapped phase path."""
    phase = model.theta[idx]
    step = torch.diff(phase, dim=-1)
    step = torch.atan2(torch.sin(step), torch.cos(step))
    return torch.cat([phase[:, :1], phase[:, :1] + torch.cumsum(step, -1)], -1)


def e2_matched_ablation(cfg, model, frontend, device, songs):
    """Four cells with rate and period PINNED, plus Viterbi as a ceiling."""
    rows = []
    emission_p = torch.sigmoid(model.emission_logits_at_bins()).detach()
    with torch.no_grad():
        for raw, records in crops(cfg, frontend, songs):
            h = frontend.forward_features(raw["input"])
            mask = raw["mask"].to(device)
            log_psi, log_T, log_g, logZ = model.posterior_marginals(h, mask)
            best = torch.log_softmax(logZ + model.rate_log_prior[None], dim=1).argmax(1)
            gamma_chain = log_g.exp().gather(
                1, best[:, None, None, None].expand(-1, 1, log_g.shape[2],
                                                    log_g.shape[3])).squeeze(1)
            gamma_psi = log_psi.exp()
            path_v = unwrap_bins(model, viterbi(model, log_psi, log_T, mask, best))
            for i, rec in enumerate(records):
                if rec is None:
                    continue
                t = len(rec["y"])
                truth = np.asarray(rec["downbeat_times"])
                rate = float(model.rates[best[i]])
                period = 2 * math.pi / (rate * rec["fps"])          # PINNED everywhere
                cells = {}
                for name, gamma in (("chain", gamma_chain[i:i + 1, :t]),
                                    ("psi", gamma_psi[i:i + 1, :t])):
                    path = model.mean_path(gamma)[0][0]
                    wraps = torch.nonzero(torch.diff(
                        torch.floor(path / (2 * math.pi))) > 0)[:, 0]
                    times = wraps.cpu().numpy() / rec["fps"] + rec["t0"]
                    cells[f"{name}/rule-g"] = times
                    probs = (gamma[0] * emission_p).sum(-1).cpu().numpy()
                    cells[f"{name}/emission"] = (peak_times(probs, rec["fps"], period)
                                                 + rec["t0"])
                wraps = torch.nonzero(torch.diff(
                    torch.floor(path_v[i, :t] / (2 * math.pi))) > 0)[:, 0]
                cells["viterbi/rule-g"] = wraps.cpu().numpy() / rec["fps"] + rec["t0"]
                row = {}
                for name, pred in cells.items():
                    row[name] = (f_measure(pred, truth)[0],
                                 continuity_scores(truth, pred)[0])
                row["period_ratio"] = period / float(
                    np.median(np.diff(np.asarray(rec["anchors"], float))))
                rows.append(row)
    return rows


def main():
    """E1 and E2: is the chain load-bearing or a smoother?"""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="vbpm/configs/schain.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", default="val", choices=("val", "test"))
    p.add_argument("--limit-per-fold", type=int, default=30)
    args = p.parse_args()

    cfg, model, frontend, device = load(args.checkpoint, args.config, args.gpu)
    _train, val, test = split_songs(0, args.limit_per_fold)
    songs = val if args.split == "val" else test

    r2, r2_control, slopes = e1_rank_one(cfg, model, frontend, device, songs)
    slope = np.concatenate(slopes)
    print("\n=== E1: is log psi a rank-one encoding of a per-frame scalar? ===")
    print(f"   R^2 on the ELBO-optimal line   mean {r2.mean():.4f}  "
          f"median {np.median(r2):.4f}  min {r2.min():.4f}  (n={len(r2)} crops)")
    print(f"   R^2 on a cos/sin control basis mean {r2_control.mean():.4f}")
    print(f"   fitted per-frame scalar: mean {slope.mean():.4f}  sd {slope.std():.4f}  "
          f"in [0,1] {np.mean((slope >= 0) & (slope <= 1)):.1%}")

    rows = e2_matched_ablation(cfg, model, frontend, device, songs)
    print("\n=== E2: matched ablation, rate and period PINNED ===")
    keys = [k for k in rows[0] if k != "period_ratio"]
    print(f"   {'cell':22} {'F':>7} {'CMLt':>7}")
    for k in keys:
        f = np.mean([r[k][0] for r in rows])
        c = np.mean([r[k][1] for r in rows])
        print(f"   {k:22} {f:7.3f} {c:7.3f}")
    print(f"   period ratio {np.mean([r['period_ratio'] for r in rows]):.3f}  "
          f"(n={len(rows)})")


if __name__ == "__main__":
    main()
