"""Posterior origin, KL-penalty and saturation audits."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnostics.likelihood import restore
from diagnostics.phase_hybrid import PhaseRecognition, initialize
from diagnostics.posterior_labels import MODEL, ROOT
from diagnostics.readouts import phase_events
from vbpm.scoring.evaluation import f_measure
from vbpm.util.vonmises import kl_vonmises


def metrics(phase, velocity, model, batch):
    """Score trajectory likelihood and physical event timing."""
    ce = (
        F.cross_entropy(
            model.decoder(phase, velocity).reshape(-1, 3),
            batch["labels"].reshape(-1),
            reduction="none",
        )
        .reshape_as(phase)
        .sum(-1)
    )
    rows = []
    for i in range(len(phase)):
        events = phase_events(phase[i].numpy())
        rows.append(
            dict(
                song=batch["songs"][i],
                CE=float(ce[i]),
                beat_F1=f_measure(events[0], batch["beat_times"][i])[0],
                downbeat_F1=f_measure(events[1], batch["downbeat_times"][i])[0],
            )
        )
    return rows


@torch.no_grad()
def origin():
    """Compare learned recognition with likelihood-selected bar origins."""
    torch.set_num_threads(1)
    results = []
    for cohort, cache in [
        ("source", "sohn_train12_generator_30seconds"),
        ("old_dev", "sohn_unseen_generator_30seconds"),
        ("fresh16", "sohn_fresh16_validation_generator30"),
    ]:
        batch = torch.load(ROOT / "runs" / cache / "batch.pt", weights_only=True)
        h = batch["h"]
        labels = batch["labels"]
        for seed in (0, 1):
            model = restore(MODEL, seed)
            q = PhaseRecognition(model)
            q.load_state_dict(
                torch.load(MODEL / f"posterior_seed{seed}.pt", weights_only=True)["state"]
            )
            q.eval()
            model.cached_prediction = model.prediction_for(h)
            pp = model.parameters_for(h)
            qp = q.parameters_for(model, h, labels)
            phase, velocity = model.trajectory(h, parameters=pp)
            qphase, _ = model.trajectory(h, parameters=qp)
            # Preserve class counts while destroying event timing, without introducing physical
            # targets.
            shuffled = labels.roll(37, 1)
            sq = q.parameters_for(model, h, shuffled)
            sensitivity = (
                torch.atan2((sq[0] - qp[0]).sin(), (sq[0] - qp[0]).cos()).abs() * 180 / math.pi
            )
            # Dense full-circle search of emission likelihood, with velocity and decoder held fixed.
            best = torch.full((len(h),), float("inf"))
            offset = torch.zeros(len(h))
            for k in range(720):
                delta = k * 2 * math.pi / 720
                ce = (
                    F.cross_entropy(
                        model.decoder(phase + delta, velocity).reshape(-1, 3),
                        labels.reshape(-1),
                        reduction="none",
                    )
                    .reshape_as(phase)
                    .sum(-1)
                )
                improve = ce < best
                best = torch.where(improve, ce, best)
                offset = torch.where(improve, torch.full_like(offset, delta), offset)
            rows = dict(
                cohort=cohort,
                seed=seed,
                prior=metrics(phase, velocity, model, batch),
                posterior=metrics(qphase, velocity, model, batch),
                label_likelihood_selected_origin=metrics(
                    phase + offset[:, None], velocity, model, batch
                ),
                posterior_label_shift_response_deg=sensitivity.tolist(),
                selected_offset_deg=(offset * 180 / math.pi).tolist(),
                prior_kappa=pp[2].tolist(),
                posterior_kappa=qp[2].tolist(),
            )
            results.append(rows)
            print(
                json.dumps(
                    dict(
                        cohort=cohort,
                        seed=seed,
                        **{
                            name: sum(x["downbeat_F1"] for x in rows[name]) / len(h)
                            for name in ["prior", "posterior", "label_likelihood_selected_origin"]
                        },
                        label_shift_response_mean_deg=float(sensitivity.mean()),
                    )
                ),
                flush=True,
            )
    output = OUT / "posterior_origin_probe.json"
    output.write_text(
        json.dumps(
            dict(
                scope=(
                    "Diagnostic only: learned posterior versus prior; likelihood-selec"
                    "ted origin uses observed N/B/D labels, never deployment or physic"
                    "al-reference selection. Fixed velocity/decoder; no KL penalty in "
                    "scan."
                ),
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


def penalty():
    """Compare reconstruction gain with the KL cost of changing bar origin."""
    torch.set_num_threads(1)
    probe = json.loads((OUT / "posterior_origin_probe.json").read_text())
    out = []
    for row in probe["results"]:
        if row["cohort"] != "old_dev":
            continue
        b = torch.load(ROOT / "runs/sohn_unseen_generator_30seconds/batch.pt", weights_only=True)
        p = restore(MODEL, row["seed"])
        q = PhaseRecognition(p)
        q.load_state_dict(
            torch.load(MODEL / f"posterior_seed{row['seed']}.pt", weights_only=True)["state"]
        )
        with torch.no_grad():
            pp = p.parameters_for(b["h"])
            qp = q.parameters_for(p, b["h"], b["labels"])
            delta = torch.tensor(row["selected_offset_deg"]) * math.pi / 180
            current = kl_vonmises(qp[0], qp[2], pp[0], pp[2])
            moved = kl_vonmises(pp[0] + delta, qp[2], pp[0], pp[2])
            for i in range(len(delta)):
                gain = row["posterior"][i]["CE"] - row["label_likelihood_selected_origin"][i]["CE"]
                out.append(
                    dict(
                        seed=row["seed"],
                        song=b["songs"][i],
                        prior_kappa=float(pp[2][i]),
                        posterior_kappa=float(qp[2][i]),
                        origin_shift_deg=float(delta[i] * 180 / math.pi),
                        mean_trajectory_CE_gain=gain,
                        current_KL=float(current[i]),
                        selected_origin_KL_same_posterior_kappa=float(moved[i]),
                        KL_increase=float(moved[i] - current[i]),
                    )
                )
    (OUT / "posterior_origin_penalty.json").write_text(
        json.dumps(
            dict(
                scope=(
                    "Diagnostic comparison: deterministic-mean CE gain versus phase KL"
                    " at unchanged recognition concentration. Not full expected ELBO o"
                    "r proof of training causality."
                ),
                results=out,
            ),
            indent=2,
        )
        + "\n"
    )
    for r in out:
        print(json.dumps(r), flush=True)


def saturation():
    """Measure audio/label activation scales and recognition saturation."""
    torch.set_num_threads(1)
    b = torch.load(ROOT / "runs/sohn_train12_generator_30seconds/batch.pt", weights_only=True)
    rows = []
    for seed in (0, 1):
        for stage in ("initial", "trained"):
            p = (
                initialize(ROOT / "runs/sohn_audio_train8_bernoulli_clock_phase_width02", seed)
                if stage == "initial"
                else restore(MODEL, seed)
            )
            q = PhaseRecognition(p)
            if stage == "trained":
                q.load_state_dict(
                    torch.load(MODEL / f"posterior_seed{seed}.pt", weights_only=True)["state"]
                )
            with torch.no_grad():
                h = b["h"]
                phase = p.prediction_for(h)["phase"]
                x = torch.cat(
                    (
                        h.mean(1),
                        (h * phase.cos()[..., None]).mean(1),
                        (h * phase.sin()[..., None]).mean(1),
                    ),
                    -1,
                )
                x = F.layer_norm(x.reshape(len(h), 3, 512), (512,)).reshape(len(h), 1536)
                y = F.one_hot(b["labels"], 3).float()
                n = y.sum(1).clamp_min(1)
                obs = torch.cat(
                    (
                        y.mean(1),
                        (y * phase.cos()[..., None]).sum(1) / n,
                        (y * phase.sin()[..., None]).sum(1) / n,
                    ),
                    -1,
                )
                a = F.linear(x, q.head[1].weight[:, :1536], q.head[1].bias)
                o = F.linear(obs, q.head[1].weight[:, 1536:])
                z = a + o
                rows.append(
                    dict(
                        seed=seed,
                        stage=stage,
                        audio_preactivation_rms=float(a.square().mean().sqrt()),
                        label_preactivation_rms=float(o.square().mean().sqrt()),
                        mean_tanh_derivative=float((1 - z.tanh().square()).mean()),
                        fraction_abs_preactivation_gt3=float((z.abs() > 3).float().mean()),
                    )
                )
    (OUT / "recognition_saturation.json").write_text(
        json.dumps(
            dict(scope="Source12 recognition hidden layer scale and saturation", results=rows),
            indent=2,
        )
        + "\n"
    )
    for r in rows:
        print(json.dumps(r))


def main():
    """Run the command-line tool."""
    global MODEL, OUT
    p = argparse.ArgumentParser(
        description="Posterior origin, KL penalty, and activation diagnostics"
    )
    p.add_argument("kind", choices=["origin", "penalty", "saturation"])
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    MODEL = a.model_dir
    OUT = a.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    target = (
        OUT
        / {
            "origin": "posterior_origin_probe.json",
            "penalty": "posterior_origin_penalty.json",
            "saturation": "recognition_saturation.json",
        }[a.kind]
    )
    if target.exists():
        p.error("Output file already exists")
    {"origin": origin, "penalty": penalty, "saturation": saturation}[a.kind]()


if __name__ == "__main__":
    main()
