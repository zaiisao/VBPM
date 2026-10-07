"""Compare conditional likelihood using a common recognition proposal.

Evaluation samples a 50/50 mixture of learned q phase and target-model p phase.
The mixture is an importance proposal only; it does not change either prior.
Both use the target's velocity law, whose density cancels in the weights.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.audio_generator import AudioPhaseFirstGSNN, _VonMisesInvCDF
from diagnostics.phase_hybrid import PhaseRecognition
from diagnostics.readouts import phase_events
from vbpm.scoring.evaluation import f_measure
from vbpm.util.vonmises import kl_vonmises


def vm_logprob(value, mean, kappa):
    """Evaluate a stable von Mises log density."""
    value, mean, kappa = (x.double() for x in (value, mean, kappa))
    return (
        -2 * kappa * torch.sin((value - mean) / 2).square()
        - torch.special.i0e(kappa).log()
        - math.log(2 * math.pi)
    )


def importance_weights(log_observation, logp, logq):
    """Compute likelihood weights for the prior/recognition proposal mixture."""
    return log_observation + logp - torch.logaddexp(logp - math.log(2), logq - math.log(2))


def restore(path, seed):
    """Load a phase-only Bernoulli-clock prior checkpoint."""
    c = torch.load(path / f"seed{seed}.pt", weights_only=True)
    assert not c["learn_tempo"] and c.get("bernoulli_clock_emission")
    model = AudioPhaseFirstGSNN(
        aligned_context=True,
        smooth_concentration=True,
        bernoulli_clock_emission=True,
        proposal_min_probability=0.5,
    )
    model.load_state_dict(c["state"])
    model.eval()
    return model


@torch.no_grad()
def estimate(model, q, batch, draws):
    """Estimate conditional likelihood and report sampling diagnostics."""
    h = batch["h"]
    labels = batch["labels"]
    count, frames = h.shape[:2]
    proposal = model.prediction_for(h)
    model.cached_prediction = proposal
    pp = model.parameters_for(h)
    qp = q.parameters_for(model, h, labels)
    g = torch.Generator().manual_seed(416971)
    mixture_weights = []
    prior_logobs = []
    posterior_logobs = []
    for left in range(0, draws, 8):
        n = min(8, draws - left)
        assert n % 2 == 0
        u = torch.rand(n, count, generator=g).clamp(1e-6, 1 - 1e-6)
        origin = torch.cat((qp[0].expand(n // 2, -1), pp[0].expand(n // 2, -1)), 0)
        kappa = torch.cat((qp[2].expand(n // 2, -1), pp[2].expand(n // 2, -1)), 0)
        eta = origin + _VonMisesInvCDF.apply(kappa, u)
        sample_noise = dict(
            phase=eta - pp[0],
            initial=torch.randn(n, count, generator=g),
            increments=torch.randn(n, count, frames - 2, generator=g),
        )
        phase, velocity = model.trajectory(h, sample_noise, parameters=pp)
        logits = model.decoder(phase, velocity)
        logobs = (
            -F.cross_entropy(
                logits.reshape(-1, 3),
                labels.unsqueeze(0).expand_as(phase).reshape(-1),
                reduction="none",
            )
            .reshape_as(phase)
            .sum(-1)
            .double()
        )
        lp = vm_logprob(eta, pp[0], pp[2])
        lq = vm_logprob(eta, qp[0], qp[2])
        mixture_weights.append(importance_weights(logobs, lp, lq))
        posterior_logobs.append(logobs[: n // 2])
        prior_logobs.append(logobs[n // 2 :])
    w = torch.cat(mixture_weights)
    po = torch.cat(prior_logobs)
    qo = torch.cat(posterior_logobs)
    is_nll = -(torch.logsumexp(w, 0) - math.log(len(w)))
    mc_nll = -(torch.logsumexp(po, 0) - math.log(len(po)))
    ess = (2 * torch.logsumexp(w, 0) - torch.logsumexp(2 * w, 0)).exp()
    kl = kl_vonmises(qp[0], qp[2], pp[0], pp[2])
    qphase, _ = model.trajectory(h, parameters=qp)
    qf = []
    for i in range(count):
        events = phase_events(qphase[i].numpy())
        qf.append(
            [
                f_measure(e, r)[0]
                for e, r in zip(events, (batch["beat_times"][i], batch["downbeat_times"][i]))
            ]
        )
    model.cached_prediction = None
    return dict(
        mixture_importance_draws=draws,
        prior_mc_draws=len(po),
        mixture_IS_negative_CLL_mean=float(is_nll.mean()),
        mixture_IS_negative_CLL_per_frame=float(is_nll.mean() / frames),
        prior_MC_negative_CLL_mean=float(mc_nll.mean()),
        expected_prior_negative_loglik=float(-po.mean()),
        common_q_negative_ELBO_mean=float((-qo.mean(0) + kl).mean()),
        KL_phase_mean=float(kl.mean()),
        importance_ESS_per_window=ess.tolist(),
        mixture_IS_negative_CLL_per_window=is_nll.tolist(),
        common_label_conditioned_q_mean_beat_F1=float(np.mean(qf, axis=0)[0]),
        common_label_conditioned_q_mean_downbeat_F1=float(np.mean(qf, axis=0)[1]),
        prior_kappa=pp[2].tolist(),
        recognition_kappa=qp[2].tolist(),
    )


def main():
    """Run the command-line tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hybrid-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--batch-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=256)
    a = p.parse_args()
    if a.output.exists() or a.draws < 8 or a.draws % 2:
        p.error("Fresh output and even draw count >=8 required")
    torch.set_num_threads(1)
    batch = torch.load(a.batch_cache, weights_only=True)
    rows = []
    for seed in (0, 1):
        hybrid = restore(a.hybrid_dir, seed)
        q = PhaseRecognition(hybrid)
        q.load_state_dict(
            torch.load(a.hybrid_dir / f"posterior_seed{seed}.pt", weights_only=True)["state"]
        )
        q.eval()
        for name, path in [("GSNN", a.baseline_dir), ("hybrid", a.hybrid_dir)]:
            result = dict(seed=seed, model=name, **estimate(restore(path, seed), q, batch, a.draws))
            rows.append(result)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "seed",
                            "model",
                            "mixture_IS_negative_CLL_mean",
                            "prior_MC_negative_CLL_mean",
                            "common_label_conditioned_q_mean_downbeat_F1",
                        )
                    }
                ),
                flush=True,
            )
    a.output.write_text(
        json.dumps(
            dict(
                scope=(
                    "Conditional likelihood comparison; common trained recognition pha"
                    "se proposal and independent prior-only prediction metrics"
                ),
                batch_cache=str(a.batch_cache),
                songs=batch["songs"],
                results=rows,
                limitations=(
                    "Finite-sample log-likelihood estimates; ESS reported. Recognition"
                    " uses labels only for likelihood evaluation and posterior diagnos"
                    "tics, never prior inference. Evaluation mixture is not a model pr"
                    "ior change."
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
