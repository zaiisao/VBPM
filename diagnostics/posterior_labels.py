"""Controlled posterior label-path ablation with coherent bar-label interventions.

Prior and decoder frozen only for diagnosis. No physical targets train recognition.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from diagnostics.likelihood import restore
from diagnostics.phase_hybrid import PhaseRecognition, noise
from diagnostics.readouts import phase_events
from vbpm.scoring.evaluation import f_measure

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "runs/sohn_train12_generator30_phase_hybrid_alpha07"


class CircularObservationRecognition(PhaseRecognition):
    """Infer circular bar origin from observation harmonics."""

    def __init__(self, p):
        nn.Module.__init__(self)
        self.net = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, 3))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.copy_(torch.tensor([1.0, 0.0, math.log(0.5)]))

    def parameters_for(self, p, h, y):
        """Return conditional phase, velocity, and concentration parameters."""
        guess = p.prediction_for(h)
        phase = guess["phase"]
        o = F.one_hot(y, 3).float()
        n = o.sum(1).clamp_min(1)
        obs = torch.cat(
            (
                o.mean(1),
                (o * phase.cos()[..., None]).sum(1) / n,
                (o * phase.sin()[..., None]).sum(1) / n,
            ),
            -1,
        )
        raw = self.net(obs)
        vel = p.velocity_for(h, guess)
        rel = torch.cat((torch.zeros_like(vel[:, :1]), vel.cumsum(-1)), 1)
        origin = phase.mean(1) + torch.atan2(raw[:, 1], raw[:, 0]) - rel.mean(1)
        return origin, vel, p.transform_concentration(raw[:, 2])


def variants(labels):
    """Rotate downbeat identities among observed beat events."""
    out = []
    for k in range(4):
        y = labels.clone()
        for i in range(len(y)):
            idx = torch.where(labels[i] > 0)[0]
            y[i, idx] = labels[i, idx].roll(k)
        out.append(y)
    return out


@torch.no_grad()
def assess(p, q, b):
    """Score recognition and its response to bar-label interventions."""
    h = b["h"]
    y = b["labels"]
    guess = p.prediction_for(h)
    pp = p.parameters_for(h)
    qp = q.parameters_for(p, h, y)
    phase, vel = p.trajectory(h, parameters=qp)
    # The existing posterior already receives these sine/cosine statistics of D labels.
    w = (y == 2).float()
    angle = torch.atan2((w * guess["phase"].sin()).sum(1), (w * guess["phase"].cos()).sum(1))
    rel = torch.cat((torch.zeros_like(vel[:, :1]), vel.cumsum(-1)), 1)
    target = guess["phase"].mean(1) - angle - rel.mean(1)
    analytic, _ = p.trajectory(h, parameters=(target, pp[1], pp[2]))
    scores = {}
    for name, ph in [("learned_q", phase), ("D_harmonic_diagnostic", analytic)]:
        scores[name] = sum(
            f_measure(phase_events(ph[i].numpy())[1], b["downbeat_times"][i])[0]
            for i in range(len(h))
        ) / len(h)
    vy = variants(y)
    origins = torch.stack([q.parameters_for(p, h, v)[0] for v in vy])
    diff = torch.atan2((origins - origins[:1]).sin(), (origins - origins[:1]).cos())
    scores["counterfactual_origin_changes_deg"] = (diff * 180 / math.pi).tolist()
    return scores


def main():
    """Run the command-line tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument(
        "--train-cache", type=Path, default=ROOT / "runs/sohn_train12_generator_30seconds/batch.pt"
    )
    p.add_argument(
        "--eval-cache", type=Path, default=ROOT / "runs/sohn_unseen_generator_30seconds/batch.pt"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--variant", choices=["copied", "residual", "broad", "circular"], default="circular"
    )
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--beta", type=float, nargs="+", default=[0.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = p.parse_args()
    if args.output.exists() or args.steps < 1 or any(b < 0 for b in args.beta):
        p.error("Fresh output, positive steps and nonnegative beta required")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    train = torch.load(args.train_cache, weights_only=True)
    dev = torch.load(args.eval_cache, weights_only=True)
    report = dict(
        variant=args.variant,
        steps=args.steps,
        train_cache=str(args.train_cache),
        eval_cache=str(args.eval_cache),
        model_dir=str(args.model_dir),
        scope=(
            "Recognition-only diagnostic, frozen trained prior and decoder; co"
            "unterfactual labels rotate B/D classes among existing beat events"
            ". All four variants have identical audio. beta=0 isolates label l"
            "earnability; beta=1 retains exact KL. Not production training."
        ),
        results=[],
    )
    for seed in args.seeds:
        for beta in args.beta:
            p = restore(args.model_dir, seed)
            for v in p.parameters():
                v.requires_grad_(False)
            torch.manual_seed(81200 + seed)
            q = (
                CircularObservationRecognition(p)
                if args.variant == "circular"
                else PhaseRecognition(p, observation_residual=args.variant in {"residual", "broad"})
            )
            if args.variant != "circular":
                missing = q.load_state_dict(
                    torch.load(args.model_dir / f"posterior_seed{seed}.pt", weights_only=True)[
                        "state"
                    ],
                    strict=False,
                )
                assert not missing.unexpected_keys and all(
                    k.startswith("observation_head.") for k in missing.missing_keys
                )
            if args.variant == "broad":
                with torch.no_grad():
                    q.head[-1].weight[1].zero_()
                    q.head[-1].bias[1].fill_(math.log(0.5))
            before = assess(p, q, dev)
            ys = variants(train["labels"])
            h = train["h"]
            pred = p.prediction_for(h)
            opt = torch.optim.Adam(q.parameters(), lr=0.0003)
            g = torch.Generator().manual_seed(91234 + seed)
            for step in range(args.steps + 1):
                if step == args.steps:
                    break
                idx = torch.randperm(len(h), generator=g)[:4]
                k = step % 4
                y = ys[k][idx]
                p.cached_prediction = {key: v[idx] for key, v in pred.items()}
                opt.zero_grad(set_to_none=True)
                rq, kl = q.loss(p, h[idx], y, noise(g, 4, h.shape[1]))
                loss = rq + beta * kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10)
                opt.step()
                if step % 200 == 0:
                    print(
                        json.dumps(
                            dict(
                                seed=seed,
                                beta=beta,
                                step=step,
                                Rq=float(rq.detach()),
                                KL=float(kl.detach()),
                            )
                        ),
                        flush=True,
                    )
            p.cached_prediction = None
            row = dict(
                seed=seed,
                beta=beta,
                variant=args.variant,
                steps=args.steps,
                before_dev=before,
                after_dev=assess(p, q, dev),
                after_source=assess(p, q, train),
            )
            report["results"].append(row)
            torch.save(
                dict(state=q.state_dict(), variant=args.variant, seed=seed, beta=beta),
                args.output / f"posterior_seed{seed}_beta{beta}.pt",
            )
            print(json.dumps(row), flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
