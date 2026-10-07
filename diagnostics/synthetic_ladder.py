"""Synthetic GSNN ladder: phase, tempo, sparse labels, then 50 Hz sampling.

Run with `python -m diagnostics.synthetic_ladder {phase,tempo,sparse,frames}`.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diagnostics import synthetic_reference as original
from diagnostics.synthetic_reference import CircularBernoulliDecoder, MLPDecoder
from vbpm.scoring.evaluation import decode_event_times, f_measure
from vbpm.util.vonmises import _VonMisesInvCDF

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent


class PhaseGSNN(nn.Module):
    """Synthetic phase-only conditional prior and observation decoder."""

    def __init__(self, decoder):
        super().__init__()
        prior = original.DistributionNetwork(2, 32)
        self.backbone = prior.backbone
        self.phase_head = prior.phase_head
        self.decoder = CircularBernoulliDecoder() if decoder == "thawed" else MLPDecoder()

    def phase(self, x, offsets=None):
        """Return the phase trajectory under fixed tempo."""
        (encoded, _) = self.backbone(x)
        direction = self.phase_head(torch.cat((x, encoded), -1)[:, 0])
        initial = torch.atan2(direction[:, 1], direction[:, 0] + 1e-08)
        if offsets is not None:
            initial = initial.unsqueeze(0) + offsets
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        return initial.unsqueeze(-1) + original.KNOWN_V * t

    def loss(self, x, y, offsets):
        """Return sampled negative observation log likelihood."""
        logits = self.decoder(self.phase(x, offsets))
        return (
            F.binary_cross_entropy_with_logits(
                logits, y.unsqueeze(0).expand_as(logits), reduction="none"
            )
            .sum((-1, -2))
            .mean()
        )


@torch.no_grad()
def phase_score(model, data):
    """Score the phase synthetic stage."""
    phase = model.phase(data["x"])
    delta = phase - data["phase"]
    offset = torch.atan2(delta.sin().mean(), delta.cos().mean())
    logits = model.decoder(phase)
    shuffled = model.decoder(phase.roll(1, 0))
    return dict(
        raw_phase_MAE_deg=float(original.circular_error(phase, data["phase"])),
        global_offset_deg=float(offset * 180 / torch.pi),
        aligned_phase_MAE_deg=float(original.circular_error(phase - offset, data["phase"])),
        mode_BCE_per_bit=float(F.binary_cross_entropy_with_logits(logits, data["y"])),
        shuffled_phase_BCE_per_bit=float(F.binary_cross_entropy_with_logits(shuffled, data["y"])),
        raw_phase_gate_passed=bool(original.circular_error(phase, data["phase"]) <= 3),
    )


def phase_main():
    """Run the phase synthetic stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--decoders", nargs="+", choices=["thawed", "mlp"], default=["thawed", "mlp"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1:
        parser.error("Fresh output directory and positive steps required")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    heldout = original.make_data("phase_only", 1024, 32, 932782)
    files = [SOURCE / "synthetic_reference.py", SOURCE / "synthetic_ladder.py"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = dict(
        status="running",
        steps=args.steps,
        results=[],
        source_hashes=hashes,
        objective=(
            "Sohn GSNN: mean sampled negative observation log likelihood; joint prior/decoder Adam"
        ),
        truth_usage=(
            "Synthetic generation and scoring only; model/loss receive x, y and independent noise"
        ),
        phase_prior="Learned original bidirectional GRU/head; fixed Von Mises concentration 40",
        fixed_factors="Known constant velocity 0.35 rad/frame; no meter latent",
        decoder_inputs="cos/sin sampled phase only; all decoder parameters trainable",
        limitation=(
            "Synthetic noisy cos/sin cues and eight dense Bernoulli channels; not real audio"
        ),
        gate="Final raw mode phase MAE <=3 degrees; aligned metric diagnostic only",
    )

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for seed in args.seeds:
        bank = original.noise_bank(args.steps, 4 * 64, 32, 34000 + seed)
        for decoder in args.decoders:
            torch.manual_seed(seed)
            model = PhaseGSNN(decoder)
            initial = {name: p.detach().clone() for (name, p) in model.named_parameters()}
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            history = []
            for step in range(args.steps + 1):
                if step % 250 == 0 or step == args.steps:
                    row = dict(seed=seed, decoder=decoder, step=step, **phase_score(model, heldout))
                    history.append(row)
                    print(json.dumps(row), flush=True)
                if step == args.steps:
                    break
                data = original.make_data("phase_only", 64, 32, 400000 + step)
                offsets = bank["phase"][step, 0].reshape(4, 64)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(data["x"], data["y"], offsets)
                loss.backward()
                if not torch.isfinite(loss) or any(
                    (p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters())
                ):
                    raise RuntimeError("Missing or nonfinite gradient")
                nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
            changes = {
                name: float((p.detach() - initial[name]).norm())
                for (name, p) in model.named_parameters()
            }
            assert all(
                (
                    value > 0
                    for (name, value) in changes.items()
                    if name.startswith(("decoder.", "phase_head."))
                )
            )
            assert any(
                (value > 0 for (name, value) in changes.items() if name.startswith("backbone."))
            )
            torch.save(
                dict(state=model.state_dict(), seed=seed, decoder=decoder),
                args.output / f"{decoder}_seed{seed}.pt",
            )
            report["results"].append(
                dict(
                    seed=seed,
                    decoder=decoder,
                    scores=history[-1],
                    history=history,
                    parameter_changes=changes,
                )
            )
            save()
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report["status"] = "complete"
    save()


class ConstantTempoGSNN(PhaseGSNN):
    """Synthetic GSNN with input-conditioned constant tempo."""

    def __init__(self):
        super().__init__("thawed")
        self.initial_velocity = nn.Linear(66, 2)
        with torch.no_grad():
            self.initial_velocity.weight.zero_()
            self.initial_velocity.bias.copy_(torch.tensor([0.0, -4.0]))

    def parameters_for(self, x):
        """Return conditional phase, velocity, and concentration parameters."""
        (encoded, _) = self.backbone(x)
        features = torch.cat((x, encoded), -1)[:, 0]
        direction = self.phase_head(features)
        phase = torch.atan2(direction[:, 1], direction[:, 0] + 1e-08)
        velocity = self.initial_velocity(features)
        return (phase, original.KNOWN_V + velocity[:, 0], velocity[:, 1].clamp(-7, -1))

    def trajectory(self, x, noise=None):
        """Integrate conditional phase and velocity trajectories."""
        (phase, velocity, log_sigma) = self.parameters_for(x)
        if noise is not None:
            phase = phase.unsqueeze(0) + noise["phase"]
            velocity = velocity.unsqueeze(0) + log_sigma.exp().unsqueeze(0) * noise["initial"]
        t = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
        return (phase.unsqueeze(-1) + velocity.unsqueeze(-1) * t, velocity)

    def loss(self, x, y, noise):
        """Return sampled negative observation log likelihood."""
        (phase, _) = self.trajectory(x, noise)
        logits = self.decoder(phase)
        return (
            F.binary_cross_entropy_with_logits(
                logits, y.unsqueeze(0).expand_as(logits), reduction="none"
            )
            .sum((-1, -2))
            .mean()
        )


class ChangingTempoGSNN(ConstantTempoGSNN):
    """Synthetic GSNN with Gaussian tempo innovations."""

    def __init__(self):
        super().__init__()
        self.velocity_increment = nn.Linear(66, 2)
        with torch.no_grad():
            self.velocity_increment.weight.zero_()
            self.velocity_increment.bias.copy_(torch.tensor([0.0, -6.0]))

    def trajectory(self, x, noise=None):
        """Integrate conditional phase and velocity trajectories."""
        (encoded, _) = self.backbone(x)
        features = torch.cat((x, encoded), -1)
        direction = self.phase_head(features[:, 0])
        initial_phase = torch.atan2(direction[:, 1], direction[:, 0] + 1e-08)
        initial = self.initial_velocity(features[:, 0])
        velocity0 = original.KNOWN_V + initial[:, 0]
        increments = self.velocity_increment(features[:, 1:-1])
        drift = increments[..., 0]
        if noise is not None:
            initial_phase = initial_phase.unsqueeze(0) + noise["phase"]
            velocity0 = (
                velocity0.unsqueeze(0)
                + initial[:, 1].clamp(-7, -1).exp().unsqueeze(0) * noise["initial"]
            )
            drift = (
                drift.unsqueeze(0)
                + increments[..., 1].clamp(-9, -3).exp().unsqueeze(0) * noise["increments"]
            )
        velocity = torch.cat(
            (velocity0.unsqueeze(-1), velocity0.unsqueeze(-1) + drift.cumsum(-1)), -1
        )
        phase = torch.cat(
            (initial_phase.unsqueeze(-1), initial_phase.unsqueeze(-1) + velocity.cumsum(-1)), -1
        )
        return (phase, velocity)


class CenteredChangingTempoGSNN(ChangingTempoGSNN):
    """Earlier centered mean coordinates, with a learned input encoder.

    A constant mean drift matches this diagnostic's synthetic trend law;
    independent Gaussian innovations still define a changing-tempo path.
    No input-to-phase shortcut or physical target is used.
    """

    def trajectory(self, x, noise=None):
        """Integrate conditional phase and velocity trajectories."""
        (encoded, _) = self.backbone(x)
        features = torch.cat((x, encoded), -1)
        pooled = features.mean(1)
        anchor_features = pooled
        if getattr(self, "anchor_features", "pooled") == "middle":
            middle = x.shape[1] // 2
            anchor_features = features[:, middle - 1 : middle + 1].mean(1)
        direction = self.phase_head(anchor_features)
        anchor = torch.atan2(direction[:, 1], direction[:, 0] + 1e-08)
        initial = self.initial_velocity(pooled)
        increments = self.velocity_increment(features[:, 1:-1])
        drift_mean = 0.003 * increments[..., 0].mean(1)
        velocity_mean = original.KNOWN_V + initial[:, 0]
        t = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
        c = t - t.mean()
        q = c.square() - c.square().mean()
        initial_phase = anchor + velocity_mean * c[0] + 0.5 * drift_mean * q[0]
        velocity0 = velocity_mean - drift_mean * ((x.shape[1] - 2) / 2)
        drift = drift_mean[:, None].expand(-1, x.shape[1] - 2)
        if noise is not None:
            delta_velocity = initial[:, 1].clamp(-7, -1).exp().unsqueeze(0) * noise["initial"]
            initial_phase = initial_phase.unsqueeze(0) + noise["phase"] - delta_velocity * t.mean()
            velocity0 = velocity0.unsqueeze(0) + delta_velocity
            drift = (
                drift.unsqueeze(0)
                + increments[..., 1].clamp(-9, -3).exp().unsqueeze(0) * noise["increments"]
            )
        velocity = torch.cat(
            (velocity0.unsqueeze(-1), velocity0.unsqueeze(-1) + drift.cumsum(-1)), -1
        )
        phase = torch.cat(
            (initial_phase.unsqueeze(-1), initial_phase.unsqueeze(-1) + velocity.cumsum(-1)), -1
        )
        return (phase, velocity)


@torch.no_grad()
def tempo_score(model, data):
    """Score the tempo synthetic stage."""
    (phase, velocity) = model.trajectory(data["x"])
    phase_error = float(original.circular_error(phase, data["phase"]))
    trajectory_velocity = (
        velocity[:, None].expand_as(data["velocity"]) if velocity.ndim == 1 else velocity
    )
    velocity_error = float((trajectory_velocity - data["velocity"]).square().mean().sqrt())
    result = dict(
        raw_phase_MAE_deg=phase_error,
        velocity_RMSE=velocity_error,
        mode_BCE_per_bit=float(F.binary_cross_entropy_with_logits(model.decoder(phase), data["y"])),
        negative_mode_velocity_fraction=float((velocity < 0).float().mean()),
        joint_gate_passed=phase_error <= 3
        and velocity_error <= 0.01
        and bool((velocity > 0).all()),
    )
    if isinstance(model, ChangingTempoGSNN):
        truth = data["velocity"]
        correlation = original.correlation(
            trajectory_velocity - trajectory_velocity.mean(1, keepdim=True),
            truth - truth.mean(1, keepdim=True),
        )
        result["within_sequence_velocity_correlation"] = correlation
        result["joint_gate_passed"] &= correlation is not None and correlation >= 0.9
    return result


def tempo_main():
    """Run the tempo synthetic stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-report", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["constant_tempo", "changing_tempo"], default="constant_tempo"
    )
    parser.add_argument("--constant-report", type=Path)
    parser.add_argument("--parameterization", choices=["initial", "centered"], default="initial")
    parser.add_argument("--anchor-features", choices=["pooled", "middle"], default="pooled")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1:
        parser.error("Fresh output and positive budget required")
    if args.parameterization == "centered" and args.stage != "changing_tempo":
        parser.error("Centered control applies to changing tempo only")
    phase_report = json.loads(args.phase_report.read_text())
    phase_results = [r for r in phase_report["results"] if r["decoder"] == "thawed"]
    if (
        phase_report["status"] != "complete"
        or len(phase_results) != 2
        or (not all((r["scores"]["raw_phase_gate_passed"] for r in phase_results)))
    ):
        parser.error("Both completed phase-only baseline seeds must pass first")
    if args.stage == "changing_tempo":
        if args.constant_report is None:
            parser.error("Changing tempo requires a completed constant-tempo report")
        previous = json.loads(args.constant_report.read_text())
        if (
            previous["status"] != "complete"
            or len(previous["results"]) != 2
            or (not all((r["scores"]["joint_gate_passed"] for r in previous["results"])))
        ):
            parser.error("Both constant-tempo seeds must pass first")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    heldout = original.make_data(args.stage, 1024, 32, 932782)
    report = dict(
        status="running",
        stage=args.stage,
        parameterization=args.parameterization,
        anchor_features=args.anchor_features,
        objective="100% GSNN sampled observation BCE; prior and decoder jointly trainable",
        initialization=(
            "Cold learned GRU/phase head and trainable calibrated decoder; no "
            "phase checkpoint loaded"
        ),
        change_from_phase_only=(
            "Learned initial Gaussian velocity head replaces known constant ve"
            "locity; same fixed initial-phase noise and integration"
        ),
        truth_usage="Generation and scoring only",
        phase_report=str(args.phase_report),
        gate=(
            "Final raw phase MAE <=3 degrees, velocity RMSE <=0.01 rad/frame, "
            "positive mode velocity"
        ),
        limitation=(
            "Synthetic noisy phase cues and dense eight-channel observations; not real audio"
        ),
        results=[],
    )
    if args.stage == "changing_tempo":
        report["change_from_phase_only"] = (
            "Constant-tempo baseline plus the original per-frame Gaussian velo"
            "city increment head and random-walk integration"
        )
        report["constant_report"] = str(args.constant_report)
        report["gate"] += "; within-sequence velocity correlation >=0.9"
    if args.parameterization == "centered":
        report["change_from_phase_only"] = (
            "Learned pooled phase anchor, mean tempo and constant mean drift i"
            "n centered coordinates; conditional initial-phase compensation fo"
            "r sampled initial velocity; original Gaussian innovation scales a"
            "nd decoder"
        )
        report["limitation"] += "; constant mean drift matches synthetic trend law"

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = ConstantTempoGSNN() if args.stage == "constant_tempo" else ChangingTempoGSNN()
        if args.parameterization == "centered":
            torch.manual_seed(seed)
            model = CenteredChangingTempoGSNN()
            model.anchor_features = args.anchor_features
        initial = {n: p.detach().clone() for (n, p) in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        bank = original.noise_bank(args.steps, 256, 32, 34000 + seed)
        history = []
        for step in range(args.steps + 1):
            if step % 250 == 0 or step == args.steps:
                row = dict(seed=seed, step=step, **tempo_score(model, heldout))
                history.append(row)
                print(json.dumps(row), flush=True)
            if step == args.steps:
                break
            data = original.make_data(args.stage, 64, 32, 400000 + step)
            names = (
                ("phase", "initial")
                if args.stage == "constant_tempo"
                else ("phase", "initial", "increments")
            )
            noise = {n: bank[n][step, 0].reshape(4, 64, *bank[n][step, 0].shape[1:]) for n in names}
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(data["x"], data["y"], noise)
            loss.backward()
            if not torch.isfinite(loss) or any(
                (p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters())
            ):
                raise RuntimeError("Missing or nonfinite GSNN gradient")
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        changes = {
            n: float((p.detach() - initial[n]).norm()) for (n, p) in model.named_parameters()
        }
        assert all(
            (
                v > 0
                for (n, v) in changes.items()
                if n.startswith(
                    ("phase_head.", "decoder.", "initial_velocity.", "velocity_increment.")
                )
            )
        )
        torch.save(dict(state=model.state_dict(), seed=seed), args.output / f"seed{seed}.pt")
        report["results"].append(
            dict(seed=seed, scores=history[-1], history=history, parameter_changes=changes)
        )
        save()
    report["status"] = "complete"
    save()


def frame_velocity(velocity):
    """Extend interval velocities to the final frame."""
    return torch.cat((velocity, velocity[..., -1:]), -1)


def sparse_make_data(count, seed):
    """Generate sparse synthetic observations."""
    data = original.make_data("changing_tempo", count, 32, seed)
    beat_index = torch.round(data["phase"] / (math.pi / 2))
    beat_phase = beat_index * (math.pi / 2)
    previous_velocity = torch.cat((data["velocity"][:, :1], data["velocity"]), -1)
    next_velocity = frame_velocity(data["velocity"])
    event = (beat_phase >= data["phase"] - 0.5 * previous_velocity) & (
        beat_phase < data["phase"] + 0.5 * next_velocity
    )
    down = event & (beat_index.remainder(4) == 0)
    data["labels"] = event.long() + down.long()
    return data


class SparseDecoder(nn.Module):
    """Trainable local circular categorical generation network.

    Angular centers, heights and widths all learn. Velocity converts angular
    distance to frame distance, so the observation kernel is one frame wide.
    The calibrated start supplies a phase convention, not training references.
    """

    def __init__(self):
        super().__init__()
        self.centers = nn.Parameter(torch.arange(4) * (math.pi / 2))
        self.height = nn.Parameter(torch.full((4,), 4.0))
        self.raw_width = nn.Parameter(torch.full((4,), math.log(math.expm1(0.16))))

    def forward(self, phase, velocity):
        """Evaluate the network on its input tensors."""
        delta = phase[..., None] - self.centers
        distance = torch.atan2(delta.sin(), delta.cos())
        width = 0.02 + F.softplus(self.raw_width)
        advance = frame_velocity(velocity).clamp_min(0.0001)
        peaks = self.height - 0.5 * (distance / (advance[..., None] * width)).square()
        return torch.stack(
            (torch.zeros_like(phase), torch.logsumexp(peaks[..., 1:], -1), peaks[..., 0]), -1
        )


class SparseGSNN(CenteredChangingTempoGSNN):
    """Generate sparse beat observations from synthetic phase/tempo cues."""

    def __init__(self, learn_concentration=False):
        super().__init__()
        self.anchor_features = "middle"
        self.decoder = SparseDecoder()
        self.learn_concentration = learn_concentration
        if learn_concentration:
            self.phase_concentration = nn.Linear(66, 1)
            with torch.no_grad():
                self.phase_concentration.weight.zero_()
                self.phase_concentration.bias.fill_(math.log(39.0))

    def concentration(self, x):
        """Return input-conditioned phase concentration."""
        if not self.learn_concentration:
            return x.new_full((len(x),), 40.0)
        (encoded, _) = self.backbone(x)
        features = torch.cat((x, encoded), -1)
        middle = x.shape[1] // 2
        raw = self.phase_concentration(features[:, middle - 1 : middle + 1].mean(1)).squeeze(-1)
        return 1 + raw.clamp(-9, math.log(100000000.0)).exp()

    def draw(self, x, noise):
        """Sample conditional phase and velocity trajectories."""
        kappa = self.concentration(x).unsqueeze(0).expand_as(noise["uniform"])
        offsets = _VonMisesInvCDF.apply(kappa, noise["uniform"])
        return self.trajectory(
            x, dict(phase=offsets, initial=noise["initial"], increments=noise["increments"])
        )

    def loss(self, x, labels, noise):
        """Return sampled negative observation log likelihood."""
        (phase, velocity) = self.draw(x, noise)
        logits = self.decoder(phase, velocity)
        targets = labels.unsqueeze(0).expand_as(phase)
        loss = F.cross_entropy(
            logits.reshape(-1, 3), targets.reshape(-1), reduction="none"
        ).reshape_as(phase)
        return loss.sum(-1).mean()


def f1(prediction, target):
    """Compute binary frame F1."""
    tp = (prediction & target).sum().item()
    return 2 * tp / max(1, prediction.sum().item() + target.sum().item())


@torch.no_grad()
def sparse_score(model, data):
    """Score the sparse synthetic stage."""
    model.eval()
    (phase, velocity) = model.trajectory(data["x"])
    logits = model.decoder(phase, velocity)
    prediction = logits.argmax(-1)
    phase_error = float(original.circular_error(phase, data["phase"]))
    velocity_error = float((velocity - data["velocity"]).square().mean().sqrt())
    truth = data["velocity"]
    correlation = original.correlation(
        velocity - velocity.mean(1, keepdim=True), truth - truth.mean(1, keepdim=True)
    )
    beat_f1 = f1(prediction > 0, data["labels"] > 0)
    down_f1 = f1(prediction == 2, data["labels"] == 2)
    physical = (
        phase_error <= 3
        and velocity_error <= 0.01
        and (correlation is not None)
        and (correlation >= 0.9)
        and bool((velocity > 0).all())
    )
    result = dict(
        raw_phase_MAE_deg=phase_error,
        velocity_RMSE=velocity_error,
        within_sequence_velocity_correlation=correlation,
        mode_CE=float(F.cross_entropy(logits.reshape(-1, 3), data["labels"].reshape(-1))),
        beat_frame_F1=beat_f1,
        downbeat_frame_F1=down_f1,
        phase_concentration_mean=float(model.concentration(data["x"]).mean()),
        physical_gate_passed=physical,
        joint_gate_passed=physical and beat_f1 >= 0.85 and (down_f1 >= 0.85),
    )
    model.train()
    return result


def sparse_main():
    """Run the sparse synthetic stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dense-dir", type=Path, default=ROOT / "runs/sohn_restart_centered_middle"
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--learn-concentration", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    previous = json.loads((args.dense_dir / "report.json").read_text())
    if previous["status"] != "complete" or not all(
        (r["scores"]["joint_gate_passed"] for r in previous["results"])
    ):
        parser.error("Passing dense observation prior required")
    if args.output.exists() or args.steps < 1:
        parser.error("Fresh output and positive steps required")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    source_files = [Path(__file__), Path(__file__), SOURCE / "synthetic_reference.py"]
    report = dict(
        status="running",
        objective="100% GSNN mean sampled categorical observation log likelihood; no posterior/KL",
        steps=args.steps,
        learn_concentration=args.learn_concentration,
        dense_initialization=str(args.dense_dir),
        change=(
            "Sparse frame-level N/B/D observations and a trainable local circu"
            "lar categorical decoder"
        ),
        decoder_trainability="All centers, heights and widths train; prior remains fully trainable",
        truth_usage=(
            "Synthetic generation and scoring only; loss accepts x, labels and independent noise"
        ),
        gate=(
            "Raw phase <=3 degrees, velocity RMSE <=0.01, within-sequence velo"
            "city correlation >=0.9, positive velocity, beat/downbeat frame F1"
            " >=0.85"
        ),
        limitation=(
            "32-frame synthetic cos/sin cues, fixed meter four, high synthetic"
            " beat density; not real audio"
        ),
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        results=[],
    )

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    heldout = sparse_make_data(1024, 932782)
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = SparseGSNN(args.learn_concentration)
        old = torch.load(args.dense_dir / f"seed{seed}.pt", weights_only=True)["state"]
        prior = {n: p for (n, p) in old.items() if not n.startswith("decoder.")}
        missing = model.load_state_dict(prior, strict=False)
        assert not missing.unexpected_keys
        assert all(
            (n.startswith(("decoder.", "phase_concentration.")) for n in missing.missing_keys)
        )
        initial = {n: p.detach().clone() for (n, p) in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        generator = torch.Generator().manual_seed(97000 + seed)
        history = []
        for step in range(args.steps + 1):
            if step % 250 == 0 or step == args.steps:
                row = dict(seed=seed, step=step, **sparse_score(model, heldout))
                history.append(row)
                print(json.dumps(row), flush=True)
            if step == args.steps:
                break
            data = sparse_make_data(64, 400000 + step)
            noise = dict(
                uniform=torch.rand(4, 64, generator=generator).clamp(1e-06, 1 - 1e-06),
                initial=torch.randn(4, 64, generator=generator),
                increments=torch.randn(4, 64, 30, generator=generator),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(data["x"], data["labels"], noise)
            loss.backward()
            if not torch.isfinite(loss) or any(
                (p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters())
            ):
                raise RuntimeError("Missing/nonfinite GSNN gradient")
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        changes = {
            n: float((p.detach() - initial[n]).norm()) for (n, p) in model.named_parameters()
        }
        assert all(
            (
                v > 0
                for (n, v) in changes.items()
                if n.startswith(
                    (
                        "phase_head.",
                        "initial_velocity.",
                        "velocity_increment.",
                        "decoder.",
                        "phase_concentration.",
                    )
                )
            )
        )
        torch.save(
            dict(state=model.state_dict(), seed=seed, learn_concentration=args.learn_concentration),
            args.output / f"seed{seed}.pt",
        )
        report["results"].append(
            dict(seed=seed, history=history, scores=history[-1], parameter_changes=changes)
        )
        save()
    report["status"] = "complete"
    save()


def frame_make_data(count, seed, factor=8):
    """Generate frame synthetic observations."""
    coarse = original.make_data("changing_tempo", count, 33, seed)
    time = torch.arange(32 * factor) / factor
    left = time.floor().long()
    fraction = time - left
    phase = coarse["phase"][:, left] + coarse["velocity"][:, left] * fraction
    velocity = phase[:, 1:] - phase[:, :-1]
    generator = torch.Generator().manual_seed(seed + 7000000)
    x = torch.stack((phase.cos(), phase.sin()), -1) + 0.08 * torch.randn(
        count, 32 * factor, 2, generator=generator
    )
    k = torch.round(phase / (math.pi / 2))
    peak = k * (math.pi / 2)
    previous = torch.cat((velocity[:, :1], velocity), -1)
    following = frame_velocity(velocity)
    event = (peak >= phase - 0.5 * previous) & (peak < phase + 0.5 * following)
    labels = event.long() + (event & (k.remainder(4) == 0)).long()
    return dict(x=x, phase=phase, velocity=velocity, labels=labels)


class TimedSparseGSNN(SparseGSNN):
    """Resample sparse synthetic dynamics onto the frontend frame grid."""

    def __init__(self, factor=8, smooth_residual=False):
        super().__init__(True)
        self.factor = factor
        self.smooth_residual = smooth_residual
        if smooth_residual:
            self.residual_model = nn.Sequential(
                nn.Linear(68, 32), nn.Tanh(), nn.Linear(32, 1, bias=False)
            )
            nn.init.zeros_(self.residual_model[-1].weight)

    def encode_features(self, x):
        """Encode cues at the original recurrent sampling rate."""
        coarse = x.reshape(len(x), -1, self.factor, 2).mean(2)
        (encoded, _) = self.backbone(coarse)
        return torch.cat((coarse, encoded), -1)

    def concentration(self, x):
        """Return input-conditioned phase concentration."""
        features = self.encode_features(x)
        middle = features.shape[1] // 2
        raw = self.phase_concentration(features[:, middle - 1 : middle + 1].mean(1)).squeeze(-1)
        return 1 + raw.clamp(-9, math.log(100000000.0)).exp()

    def base_trajectory(self, x, noise=None):
        """Integrate the mean trajectory and optional sampling noise."""
        features = self.encode_features(x)
        middle = features.shape[1] // 2
        direction = self.phase_head(features[:, middle - 1 : middle + 1].mean(1))
        anchor = torch.atan2(direction[:, 1], direction[:, 0] + 1e-08)
        initial = self.initial_velocity(features.mean(1))
        increments = self.velocity_increment(features[:, 1:-1])
        drift_mean = 0.003 * increments[..., 0].mean(1)
        velocity_mean = original.KNOWN_V + initial[:, 0]
        dt = 1 / self.factor
        t = torch.arange(x.shape[1], dtype=x.dtype, device=x.device) * dt
        c = t - t.mean()
        q = c.square() - c.square().mean()
        coarse_c = (
            torch.arange(features.shape[1], dtype=x.dtype, device=x.device)
            - (features.shape[1] - 1) / 2
        )
        anchor = anchor + 0.5 * drift_mean * (c.square().mean() - coarse_c.square().mean())
        phase0 = anchor + velocity_mean * c[0] + 0.5 * drift_mean * q[0]
        velocity0 = velocity_mean * dt - drift_mean * dt**2 * ((x.shape[1] - 2) / 2)
        drift = (drift_mean[:, None] * dt**2).expand(-1, x.shape[1] - 2)
        if noise is not None:
            delta = initial[:, 1].clamp(-7, -1).exp().unsqueeze(0) * noise["initial"]
            phase0 = phase0.unsqueeze(0) + noise["phase"] - delta * t.mean()
            velocity0 = velocity0.unsqueeze(0) + delta * dt
            sigma = F.interpolate(
                increments[..., 1].clamp(-9, -3).exp().unsqueeze(1),
                size=x.shape[1] - 2,
                mode="linear",
                align_corners=True,
            ).squeeze(1)
            drift = drift.unsqueeze(0) + sigma.unsqueeze(0) * dt**1.5 * noise["increments"]
        velocity = torch.cat(
            (velocity0.unsqueeze(-1), velocity0.unsqueeze(-1) + drift.cumsum(-1)), -1
        )
        phase = torch.cat((phase0.unsqueeze(-1), phase0.unsqueeze(-1) + velocity.cumsum(-1)), -1)
        return (phase, velocity)

    def trajectory(self, x, noise=None):
        """Integrate conditional phase and velocity trajectories."""
        (phase, velocity) = self.base_trajectory(x, noise)
        if not self.smooth_residual:
            return (phase, velocity)
        (mean_phase, _) = self.base_trajectory(x)
        features = self.encode_features(x)
        predicted = mean_phase.reshape(len(x), -1, self.factor).mean(-1)
        context = torch.cat(
            (features, predicted.cos().unsqueeze(-1), predicted.sin().unsqueeze(-1)), -1
        )
        residual = 0.25 * self.residual_model(context).squeeze(-1).tanh()
        residual = F.avg_pool1d(F.pad(residual.unsqueeze(1), (1, 1), mode="replicate"), 3, stride=1)
        residual = (
            F.interpolate(
                residual.unsqueeze(2), size=(1, x.shape[1]), mode="bicubic", align_corners=False
            )
            .squeeze(1)
            .squeeze(1)
        )
        residual = residual - residual.mean(-1, keepdim=True)
        return (phase + residual, velocity + residual[:, 1:] - residual[:, :-1])


@torch.no_grad()
def frame_score(model, data):
    """Score the frame synthetic stage."""
    model.eval()
    (phase, velocity) = model.trajectory(data["x"])
    logits = model.decoder(phase, velocity)
    prediction = logits.argmax(-1)
    phase_error = float(original.circular_error(phase, data["phase"]))
    velocity_error = float((velocity - data["velocity"]).square().mean().sqrt())
    truth = data["velocity"]
    correlation = original.correlation(
        velocity - velocity.mean(1, keepdim=True), truth - truth.mean(1, keepdim=True)
    )
    beat = f1(prediction > 0, data["labels"] > 0)
    down = f1(prediction == 2, data["labels"] == 2)
    event_scores = []
    raw_frame_scores = []
    probabilities = logits.softmax(-1).cpu().numpy()
    fps = 50 * model.factor / 8
    for i in range(len(phase)):
        (events, down_events) = decode_event_times(probabilities[i], fps)
        reference_beats = (
            np.asarray(data["beat_times"][i])
            if "beat_times" in data
            else (data["labels"][i] > 0).nonzero().flatten().cpu().numpy() / fps
        )
        reference_down = (
            np.asarray(data["downbeat_times"][i])
            if "downbeat_times" in data
            else (data["labels"][i] == 2).nonzero().flatten().cpu().numpy() / fps
        )
        event_scores.append(
            (f_measure(events, reference_beats)[0], f_measure(down_events, reference_down)[0])
        )
        raw_beats = (prediction[i] > 0).nonzero().flatten().cpu().numpy() / fps
        raw_down = (prediction[i] == 2).nonzero().flatten().cpu().numpy() / fps
        raw_frame_scores.append(
            (f_measure(raw_beats, reference_beats)[0], f_measure(raw_down, reference_down)[0])
        )
    event_scores = np.mean(event_scores, axis=0)
    raw_frame_scores = np.mean(raw_frame_scores, axis=0)
    physical = (
        phase_error <= 3
        and velocity_error <= 0.01 / model.factor
        and (correlation is not None)
        and (correlation >= 0.9)
        and bool((velocity > 0).all())
    )
    result = dict(
        raw_phase_MAE_deg=phase_error,
        velocity_RMSE_rad_per_frame=velocity_error,
        tempo_RMSE_BPM=velocity_error * 4 * 50 * 60 / (2 * math.pi),
        within_sequence_velocity_correlation=correlation,
        mode_CE=float(F.cross_entropy(logits.reshape(-1, 3), data["labels"].reshape(-1))),
        beat_frame_F1=beat,
        downbeat_frame_F1=down,
        beat_event_F1_70ms=float(event_scores[0]),
        downbeat_event_F1_70ms=float(event_scores[1]),
        uncollapsed_frame_F1_70ms=float(raw_frame_scores[0]),
        uncollapsed_downbeat_frame_F1_70ms=float(raw_frame_scores[1]),
        phase_concentration_mean=float(model.concentration(data["x"]).mean()),
        maximum_adjacent_velocity_change=float((velocity[:, 1:] - velocity[:, :-1]).abs().max()),
        negative_velocity_fraction=float((velocity <= 0).float().mean()),
        physical_gate_passed=physical,
        joint_gate_passed=physical and bool((event_scores >= 0.85).all()),
    )
    model.train()
    return result


def frame_main():
    """Run the frame synthetic stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--previous-dir", type=Path, default=ROOT / "runs/sohn_restart_sparse_learned"
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--smooth-residual", action="store_true")
    parser.add_argument("--resume-dir", type=Path)
    args = parser.parse_args()
    previous = json.loads((args.previous_dir / "report.json").read_text())
    if (
        previous["status"] != "complete"
        or len(previous["results"]) != 2
        or (not all((r["scores"]["joint_gate_passed"] for r in previous["results"])))
    ):
        parser.error("Both previous sparse observation seeds must pass")
    if args.output.exists() or min(args.steps, args.factor) < 1:
        parser.error("Fresh output and positive settings required")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    report = dict(
        status="running",
        objective=(
            "100% GSNN sampled categorical observation likelihood; prior and d"
            "ecoder fully trainable"
        ),
        change=(
            "Temporal resampling; the encoder retains its native grid through "
            "block averaging, while prior units and Brownian innovation units "
            "scale with dt"
        ),
        factor=args.factor,
        frames=32 * args.factor,
        steps=args.steps,
        previous_dir=str(args.previous_dir),
        smooth_residual=args.smooth_residual,
        resume_dir=str(args.resume_dir) if args.resume_dir else None,
        truth_usage="Synthetic generation and scoring only",
        limitation=(
            "Synthetic phase cues, fixed meter four and a quadratic mean traje"
            "ctory remain; not real audio"
        ),
        gate=(
            "Phase <=3 degrees; velocity RMSE <=0.01/factor rad/frame; within-"
            "sequence correlation >=0.9; positive velocity; one-to-one beat/do"
            "wnbeat event F1 within 70ms >=0.85"
        ),
        results=[],
    )

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    heldout = frame_make_data(512, 932782, args.factor)
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = TimedSparseGSNN(args.factor, args.smooth_residual)
        load_dir = args.resume_dir or args.previous_dir
        missing = model.load_state_dict(
            torch.load(load_dir / f"seed{seed}.pt", weights_only=True)["state"], strict=False
        )
        assert not missing.unexpected_keys
        assert all((n.startswith("residual_model.") for n in missing.missing_keys))
        initial = {n: p.detach().clone() for (n, p) in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        generator = torch.Generator().manual_seed(97000 + seed)
        history = []
        for step in range(args.steps + 1):
            if step % 250 == 0 or step == args.steps:
                row = dict(seed=seed, step=step, **frame_score(model, heldout))
                history.append(row)
                print(json.dumps(row), flush=True)
            if step == args.steps:
                break
            data = frame_make_data(32, 400000 + step, args.factor)
            noise = dict(
                uniform=torch.rand(4, 32, generator=generator).clamp(1e-06, 1 - 1e-06),
                initial=torch.randn(4, 32, generator=generator),
                increments=torch.randn(4, 32, 32 * args.factor - 2, generator=generator),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(data["x"], data["labels"], noise)
            loss.backward()
            if not torch.isfinite(loss) or any(
                (p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters())
            ):
                raise RuntimeError("Missing/nonfinite GSNN gradient")
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        changes = {
            n: float((p.detach() - initial[n]).norm()) for (n, p) in model.named_parameters()
        }
        assert all(
            (
                v > 0
                for (n, v) in changes.items()
                if n.startswith(
                    (
                        "phase_head.",
                        "initial_velocity.",
                        "velocity_increment.",
                        "decoder.",
                        "phase_concentration.",
                        "residual_model.",
                    )
                )
            )
        )
        torch.save(
            dict(
                state=model.state_dict(),
                factor=args.factor,
                smooth_residual=args.smooth_residual,
                seed=seed,
            ),
            args.output / f"seed{seed}.pt",
        )
        report["results"].append(
            dict(seed=seed, history=history, scores=history[-1], parameter_changes=changes)
        )
        save()
    report["status"] = "complete"
    save()


def main():
    """Run the command-line tool."""
    if len(sys.argv) < 2 or sys.argv[1] not in {"phase", "tempo", "sparse", "frames"}:
        print("Usage: python -m diagnostics.synthetic_ladder {phase,tempo,sparse,frames} [options]")
        return
    stage = sys.argv.pop(1)
    {"phase": phase_main, "tempo": tempo_main, "sparse": sparse_main, "frames": frame_main}[stage]()


if __name__ == "__main__":
    main()
