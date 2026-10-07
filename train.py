"""Train and evaluate VBPM with phase, tempo and meter, including pure GSNN."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from vbpm.config import load_config
from vbpm.scoring.evaluation import evaluate, print_table
from vbpm.data.excerpts import ExcerptDataset, collate_excerpts
from vbpm.frontends import build_frontend


# JA: For naive training, we always choose fold 7 to serve as the validation fold.
VAL_FOLD = 7


def beta_at(epoch: int, cfg) -> float:
    """Linear KL annealing from ``beta_start`` to ``beta_end`` over ``beta_warmup``."""
    if cfg.beta_warmup <= 0:
        return cfg.beta_end
    fraction = min(1.0, epoch / cfg.beta_warmup)
    return cfg.beta_start + fraction * (cfg.beta_end - cfg.beta_start)


def _seed_worker(_worker_id: int) -> None:
    """Reseed np.random per DataLoader worker."""
    np.random.seed(torch.initial_seed() % 2**32)


def train(
    dataset,
    frontend,
    device,
    cfg,
    hooks,
    seed: int,
    workers: int,
    val_set=None,
    select: str = "none",
    init_from: str = None,
    save_dir=None,
):
    """One seed: run the controls, then fit the objective the hooks define.

    ``select`` names a CHECKPOINT RULE, declared before the run rather than chosen
    afterwards: the returned model is the epoch that scored best on the VALIDATION
    split by that metric, never the last epoch and never anything read off the test
    set. "none" keeps the final epoch, which is only defensible when the trajectory
    is known to be monotone.
    """
    torch.manual_seed(seed)
    model = hooks.build_model(cfg, frontend).to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_excerpts,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )

    if init_from:
        checkpoint = torch.load(init_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model weights from {init_from}", flush=True)

    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)

    best = {"score": -float("inf"), "epoch": -1, "state": None}
    gsnn_only = cfg.gsnn_alpha == 0.0

    for epoch in range(cfg.epochs):
        model.train()
        beta = beta_at(epoch, cfg)

        totals, steps = np.zeros(3), 0
        gnorm = 0.0
        for raw in loader:
            with torch.no_grad():
                h = frontend.forward_features(raw["input"])

            mask = raw["mask"].to(device, non_blocking=True)
            cls = raw["cls"].to(device, non_blocking=True)
            out = model(h, mask, cls=cls, gsnn_only=gsnn_only)

            # per-frame normalisation and beta-annealed loss; reported elbo is beta=1.
            # clamp: a backstop item (fully-masked window) must cost 0, not produce nan.
            frames = mask.sum(1).clamp(min=1.0)
            cvae = out["recon"] - beta * out["kl"]
            loss = -(
                (cfg.gsnn_alpha * cvae + (1.0 - cfg.gsnn_alpha) * out["prior_recon"]) / frames
            ).mean()

            opt.zero_grad()
            loss.backward()

            gnorm += float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))

            opt.step()

            totals += [
                float(out["elbo"].mean()),
                float(out["recon"].mean()),
                float(out["kl"].mean()),
            ]
            steps += 1

        if select != "none" and val_set is not None and len(val_set):
            scored = evaluate(model, val_set, frontend, device, cfg.batch_size, seed=seed)
            per = next(iter(scored.values()))
            score = per.get(select, (float("nan"), 0))[0]
            if score > best["score"]:
                best = {
                    "score": score,
                    "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                }
            print(
                f"            select[{select}] {score:.4f}  "
                f"best {best['score']:.4f} @ epoch {best['epoch']}",
                flush=True,
            )

        print(
            f"  epoch {epoch:2d}  beta {beta:5.3f}  elbo {totals[0] / steps:9.2f}  "
            f"recon {totals[1] / steps:8.2f}  kl {totals[2] / steps:9.2f}  "
            f"|g| {gnorm / steps:8.2f}",
            flush=True,
        )

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(cfg),
                    "seed": seed,
                    "epoch": epoch,
                },
                save_dir / f"seed{seed}_epoch{epoch:02d}.pt",
            )

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(
            f"  checkpoint rule [{select}] selected epoch {best['epoch']} "
            f"(score {best['score']:.4f})",
            flush=True,
        )
    return model


def parse_args():
    """Run mechanics ONLY -- the recipe is the config's business."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="vbpm/configs/baseline.yaml",
        help="YAML recipe; its variant: key names the hooks module",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key for this run (repeatable)",
    )
    p.add_argument("--gpu", type=int, default=1, choices=(0, 1, 2, 3))
    p.add_argument(
        "--seed", type=int, default=0, help="one run = one seed; sweep seeds with an outer script"
    )
    p.add_argument(
        "--select",
        default="none",
        help="checkpoint rule: a validation metric name from the scoring "
        "table (e.g. downbeat F, beat F). Declared before the run; "
        "none keeps the last epoch.",
    )
    p.add_argument(
        "--workers", type=int, default=4, help="DataLoader workers (window draws + mmap reads)"
    )
    p.add_argument(
        "--init-from",
        default=None,
        help="import staged generator weights from a checkpoint",
    )
    p.add_argument("--save-dir", default=None, help="save the model to <save-dir>/seed<k>.pt")
    return p.parse_args()


def main() -> None:
    """Load Beat This splits, train, evaluate, print the per-dataset table."""
    args = parse_args()
    cfg, hooks = load_config(args.config, args.set)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"config {args.config}  seed {args.seed}  ->  {vars(cfg)}", flush=True)

    checkpoint = {"checkpoint": cfg.frontend_checkpoint} if cfg.frontend_checkpoint else {}
    frontend = build_frontend(cfg.frontend, device=f"cuda:{args.gpu}", **checkpoint)
    from vbpm.data.dataset import load_beat_this

    data = load_beat_this(VAL_FOLD)
    data.setup("test")

    train_source, val_source, test_source = (
        data.train_dataset,
        data.val_dataset,
        data.test_dataset,
    )
    for source in (train_source, val_source, test_source):
        original_count = len(source.items)
        kept = []
        for song in source.items:
            values = np.asarray(song["beat_value"])
            downbeats = np.flatnonzero(values == 1)
            if len(downbeats) >= 2 and np.all(np.diff(downbeats) == 4) and np.max(values) == 4:
                kept.append(song)
        source.items = kept
        print(f"Fixed 4/4 cohort: {len(kept)}/{original_count} recordings", flush=True)
    if not len(train_source.items):
        raise ValueError("No fixed 4/4 training recordings remain")

    train_set = ExcerptDataset(train_source, frontend, cfg.excerpt_seconds, full_length=True)
    val_set = ExcerptDataset(val_source, frontend, cfg.excerpt_seconds, centered=True)
    test_set = ExcerptDataset(test_source, frontend, cfg.excerpt_seconds, centered=True)

    print(
        f"songs: train {len(train_source)} / val {len(val_source)} / gtzan-test {len(test_source)}"
    )
    print(
        f"train: {len(train_set)} songs, fresh {cfg.excerpt_seconds:.0f}s window "
        f"per epoch, rejects {len(train_set.rejects)}"
    )

    model = train(
        train_set,
        frontend,
        device,
        cfg,
        hooks,
        args.seed,
        args.workers,
        val_set=val_set,
        select=args.select,
        init_from=args.init_from,
        save_dir=pathlib.Path(args.save_dir) if args.save_dir else None,
    )

    if args.save_dir:
        save_dir = pathlib.Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": vars(cfg),
                "seed": args.seed,
                "config_path": args.config,
                "overrides": list(args.set),
            },
            save_dir / f"seed{args.seed}.pt",
        )

    results = {
        name: evaluate(model, split, frontend, device, cfg.batch_size, seed=args.seed)
        for split, name in ((val_set, "val"), (test_set, "gtzan"))
        if len(split)
    }

    print_table(results)
    print(
        f"\nfps={frontend.output_fps}  excerpt={cfg.excerpt_seconds}s (fresh window per epoch)  "
        f"frontend={cfg.frontend}/{cfg.frontend_checkpoint}  "
        f"generator=audio phase/tempo, meters={cfg.meters}"
    )


if __name__ == "__main__":
    main()
