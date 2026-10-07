"""Matched phase-only Sohn hybrid control; original prior/decoder, no latent targets.

q(phi0,nu|h,y)=q(phi0|h,y,nu) p(nu|h). Velocity factors and the
centering shift conditional on velocity are identical, so joint KL is exactly
the von Mises phase KL. This is a partial recognition network, not full VBPM.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from diagnostics.audio_generator import AudioPhaseFirstGSNN, _VonMisesInvCDF
from diagnostics.audio_metrics import score
from vbpm.util.vonmises import kl_vonmises


class PhaseRecognition(nn.Module):
    """Infer phase from audio and observed N/B/D summaries."""

    def __init__(self, prior, observation_residual=False):
        super().__init__()
        assert prior.aligned_context and not prior.learn_tempo
        self.head = copy.deepcopy(prior.phase_head)
        self.head[0] = nn.Identity()
        old = self.head[1]
        expanded = nn.Linear(1545, 128)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :1536].copy_(old.weight)
            expanded.bias.copy_(old.bias)
        self.head[1] = expanded
        self.observation_head = None
        if observation_residual:
            self.observation_head = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, 2))
            # Preserve the copied posterior exactly while providing a label path
            # that does not pass through the saturated audio hidden layer.
            nn.init.zeros_(self.observation_head[-1].weight)
            nn.init.zeros_(self.observation_head[-1].bias)

    def parameters_for(self, prior, h, labels):
        """Return conditional phase, velocity, and concentration parameters."""
        guess = prior.prediction_for(h)
        phase = guess["phase"]
        x = torch.cat(
            (h.mean(1), (h * phase.cos()[..., None]).mean(1), (h * phase.sin()[..., None]).mean(1)),
            -1,
        )
        x = F.layer_norm(x.reshape(len(h), 3, 512), (512,)).reshape(len(h), 1536)
        observed = F.one_hot(labels, 3).to(h.dtype)
        count = observed.sum(1).clamp_min(1)
        observed = torch.cat(
            (
                observed.mean(1),
                (observed * phase.cos()[..., None]).sum(1) / count,
                (observed * phase.sin()[..., None]).sum(1) / count,
            ),
            -1,
        )
        raw = self.head(torch.cat((x, observed), -1))
        if self.observation_head is not None:
            raw = raw + self.observation_head(observed)
        velocity = prior.velocity_for(h, guess)
        relative = torch.cat((torch.zeros_like(velocity[:, :1]), velocity.cumsum(-1)), -1)
        phase0 = phase.mean(1) + math.pi * raw[:, 0] - relative.mean(1)
        return phase0, velocity, prior.transform_concentration(raw[:, 1])

    def loss(self, prior, h, labels, noise):
        """Return sampled negative observation log likelihood."""
        params = self.parameters_for(prior, h, labels)
        kappa = params[2].unsqueeze(0).expand_as(noise["uniform"])
        phase_noise = _VonMisesInvCDF.apply(kappa, noise["uniform"])
        phase, velocity = prior.trajectory(
            h,
            dict(phase=phase_noise, initial=noise["initial"], increments=noise["increments"]),
            parameters=params,
        )
        logits = prior.decoder(phase, velocity)
        recon = (
            F.cross_entropy(
                logits.reshape(-1, 3),
                labels.unsqueeze(0).expand_as(phase).reshape(-1),
                reduction="none",
            )
            .reshape_as(phase)
            .sum(-1)
            .mean()
        )
        p = prior.parameters_for(h)
        kl = kl_vonmises(params[0], params[2], p[0], p[2]).mean()
        return recon, kl


def noise(generator, count, frames):
    """Draw independent phase and velocity sampling noise."""
    return dict(
        uniform=torch.rand(4, count, generator=generator).clamp(1e-6, 1 - 1e-6),
        initial=torch.randn(4, count, generator=generator),
        increments=torch.randn(4, count, frames - 2, generator=generator),
    )


def initialize(parent, seed):
    """Restore the matched prior and reset experiment initialization."""
    torch.manual_seed(seed)
    prior = AudioPhaseFirstGSNN(
        aligned_context=True,
        smooth_concentration=True,
        bernoulli_clock_emission=True,
        proposal_min_probability=0.5,
    )
    prior.load_previous(torch.load(parent / f"seed{seed}.pt", weights_only=True)["state"])
    with torch.no_grad():
        prior.phase_head[-1].weight[-1].zero_()
        prior.phase_head[-1].bias[-1].fill_(math.log(0.5))
        fraction = (0.5 - 0.001) / (math.pi - 0.001)
        prior.decoder.raw_width.fill_(math.log(fraction / (1 - fraction)))
    return prior


def main():
    """Run the command-line tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume-dir", type=Path, required=True)
    p.add_argument("--batch-cache", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    if a.output.exists() or not 0 < a.alpha <= 1 or a.steps < 1:
        p.error("Fresh output and positive hybrid alpha/steps required")
    torch.set_num_threads(1)
    b = torch.load(a.batch_cache, weights_only=True)
    assert b["mask"].all()
    data = dict(
        x=b["h"],
        labels=b["labels"],
        phase=b["phi"],
        velocity=b["velocity"][:, :-1],
        beat_times=b["beat_times"],
        downbeat_times=b["downbeat_times"],
    )
    a.output.mkdir(parents=True)
    (a.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    report = dict(
        status="running",
        scope=(
            "Phase-only partial recognition control at full 30-second duration"
            "; original single-von-Mises prior, Gaussian velocity factors shar"
            "ed between q and p, fixed meter four"
        ),
        objective="alpha*(E_q[-log p_theta(y|z)] + KL(q||p)) + (1-alpha)*E_p[-log p_theta(y|z)]",
        alpha=a.alpha,
        beta=1,
        steps=a.steps,
        batch_size=a.batch_size,
        batch_cache=str(a.batch_cache),
        songs=b["songs"],
        parent=str(a.resume_dir),
        posterior=(
            "Independent copied phase network; added zero-initialized observat"
            "ion columns. No recognition tempo/meter change."
        ),
        truth_usage=(
            "N/B/D observations condition recognition and enter likelihood; ph"
            "ysical references used for scoring only"
        ),
        results=[],
    )

    def save():
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for seed in (0, 1):
        prior = initialize(a.resume_dir, seed)
        q = PhaseRecognition(prior)
        with torch.no_grad():
            pp = prior.parameters_for(b["h"])
            qp = q.parameters_for(prior, b["h"], b["labels"])
            assert all(torch.equal(x, y) for x, y in zip(pp, qp)), "Initial q must equal p exactly"
            assert torch.equal(kl_vonmises(qp[0], qp[2], pp[0], pp[2]), torch.zeros_like(pp[0]))
            proposal = prior.prediction_for(b["h"])
        parameters = list(prior.parameters()) + list(q.parameters())
        opt = torch.optim.Adam(parameters, lr=0.0003)
        initial = {n: v.detach().clone() for n, v in prior.named_parameters()}
        qinitial = {n: v.detach().clone() for n, v in q.named_parameters()}
        gp = torch.Generator().manual_seed(97000 + seed)
        gq = torch.Generator().manual_seed(277000 + seed)
        history = []
        last = {}
        for step in range(a.steps + 1):
            if step % 100 == 0 or step == a.steps:
                prior.cached_prediction = None
                row = dict(seed=seed, step=step, **score(prior, data), **last)
                history.append(row)
                print(json.dumps(row), flush=True)
            if step == a.steps:
                break
            count = min(a.batch_size, len(b["h"]))
            indices = (
                torch.arange(count)
                if count == len(b["h"])
                else torch.randperm(len(b["h"]), generator=gp)[:count]
            )
            prior.cached_prediction = {k: v[indices] for k, v in proposal.items()}
            np = noise(gp, count, b["h"].shape[1])
            nq = noise(gq, count, b["h"].shape[1])
            h = b["h"][indices]
            labels = b["labels"][indices]
            opt.zero_grad(set_to_none=True)
            rp = prior.loss(h, labels, np)
            rq, kl = q.loss(prior, h, labels, nq)
            loss = a.alpha * (rq + kl) + (1 - a.alpha) * rp
            loss.backward()
            if not torch.isfinite(loss) or any(
                v.grad is None or not torch.isfinite(v.grad).all() for v in parameters
            ):
                raise RuntimeError("Missing/nonfinite hybrid gradients")
            nn.utils.clip_grad_norm_(parameters, 10)
            opt.step()
            prior.cached_prediction = None
            last = dict(
                reconstruction_p=float(rp.detach()),
                reconstruction_q=float(rq.detach()),
                KL_phase=float(kl.detach()),
                hybrid_loss=float(loss.detach()),
            )
        changes = {n: float((v - initial[n]).norm()) for n, v in prior.named_parameters()}
        qchanges = {n: float((v - qinitial[n]).norm()) for n, v in q.named_parameters()}
        assert all(v > 0 for v in changes.values()) and all(v > 0 for v in qchanges.values())
        torch.save(
            dict(
                state=prior.state_dict(),
                seed=seed,
                learn_tempo=False,
                aligned_context=True,
                smooth_concentration=True,
                bernoulli_clock_emission=True,
                clock_mass_emission=True,
                frame_bin_emission=False,
                proposal_min_probability=0.5,
                tempo_basis="framewise",
                tempo_log_scale=0.1,
                hybrid_alpha=a.alpha,
            ),
            a.output / f"seed{seed}.pt",
        )
        torch.save(
            dict(state=q.state_dict(), seed=seed, alpha=a.alpha),
            a.output / f"posterior_seed{seed}.pt",
        )
        report["results"].append(
            dict(
                seed=seed,
                scores=history[-1],
                history=history,
                parameter_changes=changes,
                posterior_parameter_changes=qchanges,
            )
        )
        save()
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
