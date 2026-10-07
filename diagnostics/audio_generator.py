"""Real-audio GSNN ladder: correct bar phase before learning tempo correction.

Audio-only baseline predictions initialize physical mean coordinates. All
allocated learned prior and decoder parameters train with observation CE only.
There are no posterior parameters or latent target losses.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from diagnostics.audio_metrics import score
from diagnostics.emissions import (
    AngularFrameBinDecoder,
    BernoulliClockDecoder,
    ClockMassDecoder,
    FrameBinDecoder,
)
from diagnostics.real_event_references import attach_references
from diagnostics.synthetic_ladder import ROOT, SparseDecoder, _VonMisesInvCDF
from vbpm.audio_prediction_prior import AudioPredictionPrior
from vbpm.nets import LandmarkEmission


class StagedLandmarkDecoder(LandmarkEmission):
    """Use the existing trainable angular emission at known meter four."""

    def __init__(self):
        super().__init__((4,))

    def forward(self, phase, velocity):
        """Evaluate the network on its input tensors."""
        return super().forward(phase, velocity, phase.new_ones((*phase.shape, 1)), None, None)


class PhaseContextNormalization(nn.Module):
    """Normalize each pooled audio phase-context block."""

    def forward(self, context):
        """Evaluate the network on its input tensors."""
        return F.layer_norm(context.reshape(len(context), 3, 512), (512,)).reshape(
            len(context), 1536
        )


class AudioPhaseFirstGSNN(nn.Module):
    """Audio-only phase/tempo prior with a trainable observation decoder."""

    def __init__(
        self,
        learn_tempo=False,
        aligned_context=False,
        proposal_max_bpm=None,
        tempo_log_scale=0.1,
        tempo_basis="framewise",
        smooth_concentration=False,
        frame_bin_emission=False,
        angular_frame_bin_emission=False,
        landmark_emission=False,
        log_tempo_noise=False,
        clock_mass_emission=False,
        proposal_min_probability=None,
        tempo_feedback=False,
        bernoulli_clock_emission=False,
    ):
        super().__init__()
        self.factor = 8
        self.fps = 50.0
        self.proposal_meter = 4
        self.learn_tempo = learn_tempo
        self.aligned_context = aligned_context
        self.proposal_max_bpm = proposal_max_bpm
        self.proposal_min_probability = proposal_min_probability
        self.tempo_log_scale = tempo_log_scale
        self.tempo_basis = tempo_basis
        self.tempo_feedback = tempo_feedback
        self.smooth_concentration = smooth_concentration
        self.frame_bin_emission = frame_bin_emission
        self.log_tempo_noise = log_tempo_noise
        self.proposal_shift = None
        self.cached_prediction = None
        state = torch.load(
            Path.home() / ".cache/torch/hub/checkpoints/beat_this-final0.ckpt",
            weights_only=False,
            map_location="cpu",
        )["state_dict"]
        self.register_buffer(
            "prediction_weight", state["model.task_heads.beat_downbeat_lin.weight"].clone()
        )
        self.register_buffer(
            "prediction_bias", state["model.task_heads.beat_downbeat_lin.bias"].clone()
        )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(512, elementwise_affine=False),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
        )
        if aligned_context:
            self.phase_head[0] = PhaseContextNormalization()
            self.phase_head[1] = nn.Linear(1536, 128)
        with torch.no_grad():
            self.phase_head[-1].weight.zero_()
            self.phase_head[-1].bias.copy_(torch.tensor([0.0, math.log(39.0)]))
        self.log_initial_sigma = nn.Parameter(torch.tensor(-9.0))
        self.log_increment_sigma = nn.Parameter(torch.tensor(-12.0))
        self.decoder = (
            BernoulliClockDecoder()
            if bernoulli_clock_emission
            else (
                ClockMassDecoder()
                if clock_mass_emission
                else (
                    StagedLandmarkDecoder()
                    if landmark_emission
                    else (
                        AngularFrameBinDecoder()
                        if angular_frame_bin_emission
                        else (FrameBinDecoder() if frame_bin_emission else SparseDecoder())
                    )
                )
            )
        )
        if learn_tempo:
            self.tempo_head = nn.Sequential(
                nn.LayerNorm(512, elementwise_affine=False),
                nn.Linear(512, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.tempo_head[-1].weight)
            nn.init.zeros_(self.tempo_head[-1].bias)
            if tempo_feedback:
                original = self.tempo_head[1]
                expanded = nn.Linear(517, 64)
                with torch.no_grad():
                    expanded.weight.zero_()
                    expanded.weight[:, :512].copy_(original.weight)
                    expanded.bias.copy_(original.bias)
                self.tempo_head[1] = expanded

    def enable_temporal_phase_context(self):
        # Zero residual projection preserves the existing prior at initialization.
        """Add a zero-initialized temporal residual to phase conditioning."""
        self.temporal_phase_context = nn.Sequential(
            nn.Conv1d(512, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(64, 64, 5, dilation=8, padding=16),
            nn.Tanh(),
            nn.Conv1d(64, 64, 5, dilation=32, padding=64),
            nn.Tanh(),
            nn.Conv1d(64, 512, 1),
        )
        nn.init.zeros_(self.temporal_phase_context[-1].weight)
        nn.init.zeros_(self.temporal_phase_context[-1].bias)

    def parameters_for(self, h):
        """Return conditional phase, velocity, and concentration parameters."""
        guess = self.prediction_for(h)
        if self.proposal_shift is not None:
            guess["phase"] = guess["phase"] + self.proposal_shift[:, None]
        temporal_h = h
        if hasattr(self, "temporal_phase_context"):
            temporal_h = h + self.temporal_phase_context(h.transpose(1, 2)).transpose(1, 2)
        context = temporal_h.mean(1)
        if getattr(self, "periodic_context_only", False):
            # Remove constant audio content from the phase-weighted pools.
            # Finite windows otherwise leak their DC component into these pools.
            temporal_h = temporal_h - context[:, None]
            context = torch.zeros_like(context)
        if self.aligned_context:
            context = torch.cat(
                (
                    context,
                    (temporal_h * guess["phase"].cos().unsqueeze(-1)).mean(1),
                    (temporal_h * guess["phase"].sin().unsqueeze(-1)).mean(1),
                ),
                -1,
            )
        raw = self.phase_head(context)
        velocity = self.velocity_for(h, guess)
        relative = torch.cat((torch.zeros_like(velocity[:, :1]), velocity.cumsum(-1)), -1)
        anchor = guess["phase"].mean(1) + math.pi * raw[:, 0]
        phase0 = anchor - relative.mean(1)
        kappa = self.transform_concentration(raw[:, 1])
        return phase0, velocity, kappa

    def prediction_for(self, h):
        """Return cached or freshly computed audio-only proposals."""
        if self.cached_prediction is not None:
            return dict(self.cached_prediction)
        return AudioPredictionPrior.initial_prediction(
            self,
            h,
            h.new_ones(h.shape[:2]),
            max_bpm=self.proposal_max_bpm,
            min_probability=self.proposal_min_probability,
            recover_missing_beats=getattr(self, "recover_missing_beats", False),
        )

    def transform_concentration(self, raw):
        """Map raw network output to positive phase concentration."""
        if self.smooth_concentration:
            limit = math.log(1e8)
            return 1 + (limit - F.softplus(limit - raw)).exp()
        return 1 + raw.clamp(-9, math.log(1e8)).exp()

    def velocity_for(self, h, guess):
        """Return the audio-conditioned mean velocity trajectory."""
        velocity = guess["phase"][:, 1:] - guess["phase"][:, :-1]
        if self.learn_tempo:
            if getattr(self, "tempo_feedback", False):
                # Sohn-style initial-prediction feedback: the residual head
                # must observe the proposal whose error it is correcting.
                per_frame_velocity = torch.cat((velocity, velocity[:, -1:]), -1)
                reference = 2 * math.pi * 2 / (self.proposal_meter * self.fps)
                cues = torch.stack(
                    (
                        guess["beat_logits"].sigmoid(),
                        guess["downbeat_logits"].sigmoid(),
                        (4 * guess["phase"]).cos(),
                        (4 * guess["phase"]).sin(),
                        (per_frame_velocity / reference).clamp_min(1e-6).log(),
                    ),
                    -1,
                )
                context = torch.cat((self.tempo_head[0](h), cues), -1)
                correction = self.tempo_head[1:](context).squeeze(-1)
            else:
                correction = self.tempo_head(h).squeeze(-1)
            if self.tempo_basis == "constant-correction":
                # Keep the audio proposal's within-window tempo variation,
                # but prevent the learned mean from chasing individual frames.
                correction = correction.mean(-1, keepdim=True).expand_as(correction)
            else:
                correction = F.avg_pool1d(
                    F.pad(correction.unsqueeze(1), (8, 8), mode="replicate"), 17, stride=1
                ).squeeze(1)
            if self.tempo_basis == "bounded-correction":
                # The existing physical gate uses 20..300 BPM. Parameterize
                # that mean domain smoothly rather than clipping predictions.
                minimum = 20 * (2 * math.pi) / (4 * 60 * self.fps)
                span = 280 * (2 * math.pi) / (4 * 60 * self.fps)
                fraction = ((velocity - minimum) / span).clamp(1e-6, 1 - 1e-6)
                logit = fraction.log() - torch.log1p(-fraction)
                fraction = (-F.softplus(-logit - self.tempo_log_scale * correction[:, :-1])).exp()
                velocity = minimum + span * fraction
            else:
                velocity = velocity * (self.tempo_log_scale * correction[:, :-1].tanh()).exp()
        if self.tempo_basis == "linear":
            # The mean log-tempo has a level and slope, as in the original
            # prior's linear trajectory option. Innovation noise still varies
            # by frame; this restricts only the conditional mean trajectory.
            log_velocity = velocity.clamp_min(1e-6).log()
            centered_time = torch.linspace(
                -1.0, 1.0, velocity.shape[1], device=h.device, dtype=h.dtype
            )
            level = log_velocity.mean(-1, keepdim=True)
            slope = (log_velocity * centered_time).mean(
                -1, keepdim=True
            ) / centered_time.square().mean()
            velocity = (level + slope * centered_time).exp()
        return velocity

    def load_previous(self, state):
        """Load compatible parameters from a previous staged checkpoint."""
        state = dict(state)
        if isinstance(self.decoder, StagedLandmarkDecoder) and "decoder.event_bias" not in state:
            state = {n: p for n, p in state.items() if not n.startswith("decoder.")}
        if isinstance(self.decoder, ClockMassDecoder) and (
            "decoder.raw_width" not in state
            or state["decoder.raw_width"].shape != self.decoder.raw_width.shape
        ):
            state = {n: p for n, p in state.items() if not n.startswith("decoder.")}
        if self.aligned_context and state["phase_head.1.weight"].shape[1] == 512:
            expanded = torch.zeros_like(self.phase_head[1].weight)
            expanded[:, :512] = state["phase_head.1.weight"]
            state["phase_head.1.weight"] = expanded
        return self.load_state_dict(state, strict=False)

    def concentration(self, h):
        """Return input-conditioned phase concentration."""
        return self.parameters_for(h)[2]

    def trajectory(self, h, noise=None, parameters=None):
        """Integrate conditional phase and velocity trajectories."""
        phase0, velocity, _ = self.parameters_for(h) if parameters is None else parameters
        if noise is not None:
            delta0 = self.log_initial_sigma.clamp(-18, -1).exp() * noise["initial"]
            increments = self.log_increment_sigma.clamp(-18, -1).exp() * noise["increments"]
            offsets = torch.cat((delta0[..., None], delta0[..., None] + increments.cumsum(-1)), -1)
            if self.log_tempo_noise:
                mean_velocity = velocity.unsqueeze(0)
                velocity = mean_velocity * offsets.exp()
                drift = (velocity - mean_velocity).cumsum(-1)
                centered_drift = torch.cat((torch.zeros_like(drift[..., :1]), drift), -1).mean(-1)
                phase0 = phase0.unsqueeze(0) + noise["phase"] - centered_drift
            else:
                velocity = velocity.unsqueeze(0) + offsets
                phase0 = phase0.unsqueeze(0) + noise["phase"] - delta0 * ((h.shape[1] - 1) / 2)
        phase = torch.cat((phase0[..., None], phase0[..., None] + velocity.cumsum(-1)), -1)
        return phase, velocity

    def draw(self, h, noise):
        """Sample conditional phase and velocity trajectories."""
        parameters = self.parameters_for(h)
        kappa = parameters[2].unsqueeze(0).expand_as(noise["uniform"])
        phase_noise = _VonMisesInvCDF.apply(kappa, noise["uniform"])
        return self.trajectory(
            h,
            dict(phase=phase_noise, initial=noise["initial"], increments=noise["increments"]),
            parameters=parameters,
        )

    def loss(self, h, labels, noise):
        """Return sampled negative observation log likelihood."""
        phase, velocity = self.draw(h, noise)
        logits = self.decoder(phase, velocity)
        loss = F.cross_entropy(
            logits.reshape(-1, 3),
            labels.unsqueeze(0).expand_as(phase).reshape(-1),
            reduction="none",
        ).reshape_as(phase)
        return loss.sum(-1).mean()


class AudioPhaseAttentionGSNN(AudioPhaseFirstGSNN):
    """Learn phase origin relative to the current predicted tempo trajectory."""

    def __init__(
        self,
        learn_tempo=False,
        aligned_context=False,
        proposal_max_bpm=None,
        tempo_log_scale=0.1,
        tempo_basis="framewise",
        smooth_concentration=False,
        frame_bin_emission=False,
        phase_residual=False,
        angular_frame_bin_emission=False,
        landmark_emission=False,
        log_tempo_noise=False,
        clock_mass_emission=False,
        proposal_min_probability=None,
        tempo_feedback=False,
        bernoulli_clock_emission=False,
    ):
        if aligned_context:
            raise ValueError("Attention phase origin already uses the current trajectory directly")
        super().__init__(
            learn_tempo,
            False,
            proposal_max_bpm,
            tempo_log_scale,
            tempo_basis,
            smooth_concentration,
            frame_bin_emission,
            angular_frame_bin_emission,
            landmark_emission,
            log_tempo_noise,
            clock_mass_emission,
            proposal_min_probability,
            tempo_feedback,
            bernoulli_clock_emission,
        )
        self.phase_head[-1] = nn.Linear(128, 1)
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.constant_(self.phase_head[-1].bias, math.log(39.0))
        self.phase_attention = nn.Linear(512, 1, bias=False)
        with torch.no_grad():
            self.phase_attention.weight.copy_(self.prediction_weight[1:2])
        self.use_phase_residual = phase_residual
        if phase_residual:
            self.phase_residual = nn.Sequential(
                PhaseContextNormalization(), nn.Linear(1536, 128), nn.Tanh(), nn.Linear(128, 2)
            )
            nn.init.zeros_(self.phase_residual[-1].weight)
            nn.init.zeros_(self.phase_residual[-1].bias)

    def parameters_for(self, h):
        """Return conditional phase, velocity, and concentration parameters."""
        guess = self.prediction_for(h)
        velocity = self.velocity_for(h, guess)
        relative = torch.cat((torch.zeros_like(velocity[:, :1]), velocity.cumsum(-1)), -1)
        attention = self.phase_attention(h).squeeze(-1).softmax(-1)
        x = (attention * relative.cos()).sum(-1)
        y = (attention * relative.sin()).sum(-1)
        if self.use_phase_residual:
            context = torch.cat(
                (
                    h.mean(1),
                    (h * relative.cos().unsqueeze(-1)).mean(1),
                    (h * relative.sin().unsqueeze(-1)).mean(1),
                ),
                -1,
            )
            correction = self.phase_residual(context)
            x = x + correction[:, 0]
            y = y + correction[:, 1]
        phase0 = -torch.atan2(y, x + 1e-8)
        raw = self.phase_head(h.mean(1)).squeeze(-1)
        kappa = self.transform_concentration(raw)
        return phase0, velocity, kappa


def main():
    """Run the command-line tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learn-tempo", action="store_true")
    parser.add_argument("--phase-dir", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--aligned-context", action="store_true")
    parser.add_argument("--augment-origin", action="store_true")
    parser.add_argument(
        "--batch-cache", type=Path, default=ROOT / "runs/generator_isolation/oracle_batch.pt"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--proposal-max-bpm", type=float)
    parser.add_argument("--proposal-min-probability", type=float)
    parser.add_argument("--tempo-log-scale", type=float, default=0.1)
    parser.add_argument(
        "--fresh-audio-prior",
        action="store_true",
        help="Reuse only the decoder; initialize audio prior and concentration anew",
    )
    parser.add_argument(
        "--tempo-basis",
        choices=["framewise", "linear", "constant-correction", "bounded-correction"],
        default="framewise",
    )
    parser.add_argument("--phase-attention", action="store_true")
    parser.add_argument("--smooth-concentration", action="store_true")
    parser.add_argument("--frame-bin-emission", action="store_true")
    parser.add_argument("--angular-frame-bin-emission", action="store_true")
    parser.add_argument(
        "--landmark-emission",
        action="store_true",
        help="Use the existing trainable LandmarkEmission with structural metrical centers",
    )
    parser.add_argument(
        "--clock-mass-emission",
        action="store_true",
        help="Use normalized event mass per clock landmark with learned angular timing uncertainty",
    )
    parser.add_argument(
        "--landmark-initial-kappa",
        type=float,
        help="Initialize trainable emission concentrations after loading",
    )
    parser.add_argument("--phase-residual", action="store_true")
    parser.add_argument(
        "--reset-concentration",
        type=float,
        help="Initialize the concentration head to this value after loading; continue training it",
    )
    parser.add_argument(
        "--log-tempo-noise",
        action="store_true",
        help="Sample the tempo walk in log space to keep velocity positive",
    )
    parser.add_argument(
        "--initial-tempo-sigma",
        type=float,
        help=(
            "Reset the learned initial tempo standard deviation after loading;"
            " log units with log-tempo-noise"
        ),
    )
    parser.add_argument(
        "--clock-initial-width",
        type=float,
        help="Initialize learned normalized-clock angular timing widths after loading",
    )
    parser.add_argument(
        "--bernoulli-clock-emission",
        action="store_true",
        help="Normalized landmark mass with Bernoulli categorical link",
    )
    parser.add_argument(
        "--tempo-feedback",
        action="store_true",
        help="Condition tempo residual on audio initial-prediction cues",
    )
    parser.add_argument(
        "--recover-missing-beats",
        action="store_true",
        help="Recover ordinal count across isolated integer-multiple audio peak gaps",
    )
    parser.add_argument(
        "--periodic-context-only",
        action="store_true",
        help=(
            "Condition scalar phase head on centered phase-weighted audio pool"
            "s, removing global audio mean"
        ),
    )
    parser.add_argument(
        "--temporal-phase-context",
        action="store_true",
        help="Add order-sensitive temporal residual features only to the phase head",
    )
    args = parser.parse_args()
    if args.temporal_phase_context and (not args.aligned_context or args.phase_attention):
        parser.error("Temporal context control requires the existing aligned scalar phase head")
    if args.bernoulli_clock_emission:
        args.clock_mass_emission = True
    if args.angular_frame_bin_emission:
        args.frame_bin_emission = True
    if args.landmark_emission and args.frame_bin_emission:
        parser.error("Select one emission formulation")
    if args.clock_mass_emission and (args.landmark_emission or args.frame_bin_emission):
        parser.error("Select one emission formulation")
    if args.clock_initial_width is not None and (
        not args.clock_mass_emission or not 0.001 < args.clock_initial_width < math.pi
    ):
        parser.error(
            "Clock width must be between .001 and pi radians and requires normalized clock emission"
        )
    if args.landmark_initial_kappa is not None and (
        not args.landmark_emission or args.landmark_initial_kappa <= 0
    ):
        parser.error("Positive landmark initial kappa requires landmark emission")
    if args.output.exists() or args.steps < 1 or args.batch_size < 1:
        parser.error("Fresh output, positive steps and batch size required")
    if args.tempo_feedback and not args.learn_tempo:
        parser.error("Tempo feedback requires a learned tempo head")
    if args.periodic_context_only and not args.aligned_context:
        parser.error("Periodic-only context requires aligned context")
    if args.augment_origin and not args.aligned_context:
        parser.error("Origin augmentation requires the prior to receive the initial prediction")
    if args.proposal_max_bpm is not None and args.proposal_max_bpm <= 0:
        parser.error("Proposal maximum BPM must be positive")
    if args.proposal_min_probability is not None and not 0 <= args.proposal_min_probability < 1:
        parser.error("Proposal probability threshold must be in [0,1)")
    if args.tempo_log_scale <= 0:
        parser.error("Tempo log correction scale must be positive")
    if args.phase_attention and (args.aligned_context or args.augment_origin):
        parser.error(
            (
                "Attention phase origin uses the current trajectory directly; omit"
                " aligned context and augmentation"
            )
        )
    if args.phase_residual and not args.phase_attention:
        parser.error("The phase residual extends the attention head; enable phase attention")
    if args.reset_concentration is not None and args.reset_concentration <= 1:
        parser.error("Reset concentration must exceed one")
    if args.initial_tempo_sigma is not None and not math.exp(
        -18
    ) < args.initial_tempo_sigma < math.exp(-1):
        parser.error("Initial tempo sigma must be inside the existing learned sigma bounds")
    if args.learn_tempo:
        if args.phase_dir is None:
            parser.error("Tempo correction requires a passing phase-only checkpoint")
        acceptance_path = args.phase_dir / "acceptance_70ms.json"
        previous = json.loads(
            (
                acceptance_path if acceptance_path.exists() else args.phase_dir / "report.json"
            ).read_text()
        )
        if (
            previous["status"] != "complete"
            or len(previous["results"]) != 2
            or not all(r["scores"]["joint_gate_passed"] for r in previous["results"])
        ):
            parser.error("Both audio phase-only seeds must pass first")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if args.clock_mass_emission:
        source = Path(__file__).with_name("sohn_clock_mass_emission.py")
        (args.output / source.name).write_bytes(source.read_bytes())
    batch = torch.load(args.batch_cache, weights_only=True)
    assert bool(batch["mask"].all()), "This staged diagnostic requires fully observed windows"
    data = dict(
        x=batch["h"], phase=batch["phi"], velocity=batch["velocity"][:, :-1], labels=batch["labels"]
    )
    if "beat_times" in batch:
        data.update(beat_times=batch["beat_times"], downbeat_times=batch["downbeat_times"])
    else:
        attach_references(data, batch)
    report = dict(
        status="running",
        objective=(
            "100% GSNN sampled categorical observation CE; all allocated prior"
            " and decoder parameters trainable"
        ),
        learn_tempo=args.learn_tempo,
        steps=args.steps,
        phase_dir=str(args.phase_dir) if args.phase_dir else None,
        aligned_context=args.aligned_context,
        periodic_context_only=args.periodic_context_only,
        recover_missing_beats=args.recover_missing_beats,
        tempo_feedback=args.tempo_feedback,
        augment_origin=args.augment_origin,
        resume_dir=str(args.resume_dir) if args.resume_dir else None,
        scope=(
            f"{len(batch['h'])}"
            " fixed real-audio training windows at 50Hz; fixed meter four; not"
            " heldout"
        ),
        batch_cache=str(args.batch_cache),
        batch_size=args.batch_size,
        songs=batch["songs"],
        proposal_max_bpm=args.proposal_max_bpm,
        proposal_min_probability=args.proposal_min_probability,
        tempo_log_scale=args.tempo_log_scale,
        fresh_audio_prior=args.fresh_audio_prior,
        tempo_basis=args.tempo_basis,
        phase_attention=args.phase_attention,
        smooth_concentration=args.smooth_concentration,
        frame_bin_emission=args.frame_bin_emission,
        angular_frame_bin_emission=args.angular_frame_bin_emission,
        landmark_emission=args.landmark_emission,
        clock_mass_emission=args.clock_mass_emission,
        temporal_phase_context=args.temporal_phase_context,
        bernoulli_clock_emission=args.bernoulli_clock_emission,
        clock_initial_width=args.clock_initial_width,
        landmark_initial_kappa=args.landmark_initial_kappa,
        phase_residual=args.phase_residual,
        reset_concentration=args.reset_concentration,
        log_tempo_noise=args.log_tempo_noise,
        initial_tempo_sigma=args.initial_tempo_sigma,
        tempo_noise=(
            "Gaussian log-tempo walk, positive velocity; deterministic traject"
            "ory is zero-noise/median tempo"
        )
        if args.log_tempo_noise
        else "Additive Gaussian velocity initial/increment noise",
        phase=(
            "Learned audio attention determines circular origin relative to th"
            "e current predicted tempo; h predicts log concentration"
        )
        if args.phase_attention
        else (
            "Unbounded scalar correction to audio-proposed phase anchor, and l"
            "og concentration, predicted from h"
        ),
        tempo="Audio-only proposal mean; Gaussian initial/increment variances learned"
        if not args.learn_tempo
        else (
            (
                "Smooth 20..300 BPM conditional mean initialized from audio; unres"
                "tricted logit correction; Gaussian variances learned"
            )
            if args.tempo_basis == "bounded-correction"
            else (
                "Learned smooth log-tempo correction around audio proposal, bounde"
                "d +/-"
                f"{args.tempo_log_scale}"
                " in log units; Gaussian variances learned"
            )
        ),
        decoder=(
            "Normalized metrical pulse masses, learned angular timing widths, "
            "Bernoulli union N/B/D probabilities with downbeat precedence; no "
            "audio input"
        )
        if args.bernoulli_clock_emission
        else (
            (
                "Normalized metrical clock-event mass, learned angular timing widt"
                "hs, competing Poisson N/B/D probabilities; no audio input"
            )
            if args.clock_mass_emission
            else (
                (
                    "Existing LandmarkEmission: trainable event biases and angular con"
                    "centrations, structural metrical centers, no audio input"
                )
                if args.landmark_emission
                else (
                    (
                        "Angular Gaussian mass integrated over physical frame intervals; "
                        if args.angular_frame_bin_emission
                        else (
                            "Normalized probability mass integrated over frame bins; "
                            if args.frame_bin_emission
                            else "Pointwise kernels; "
                        )
                    )
                    + (
                        "all peak centers, heights and widths trained jointly; no audio in"
                        "put to decoder"
                    )
                )
            )
        ),
        truth_usage=(
            "Beat/downbeat labels only in likelihood; references scoring only;"
            " frozen baseline frontend already pretrained"
        ),
        gate=(
            "Phase time error p95 <=70ms; observed interbeat tempo RMSE <=3 BP"
            "M; positive bounded tempo; beat/downbeat event F1 +/-70ms >=0.85;"
            " raw angular and instantaneous-interpolation errors retained as d"
            "iagnostics"
        ),
        results=[],
    )

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for seed in (0, 1):
        torch.manual_seed(seed)
        model_class = AudioPhaseAttentionGSNN if args.phase_attention else AudioPhaseFirstGSNN
        extra = dict(
            angular_frame_bin_emission=args.angular_frame_bin_emission,
            landmark_emission=args.landmark_emission,
            log_tempo_noise=args.log_tempo_noise,
            clock_mass_emission=args.clock_mass_emission,
            proposal_min_probability=args.proposal_min_probability,
            tempo_feedback=args.tempo_feedback,
            bernoulli_clock_emission=args.bernoulli_clock_emission,
        )
        if args.phase_attention:
            extra["phase_residual"] = args.phase_residual
        model = model_class(
            args.learn_tempo,
            args.aligned_context,
            args.proposal_max_bpm,
            args.tempo_log_scale,
            args.tempo_basis,
            args.smooth_concentration,
            args.frame_bin_emission,
            **extra,
        )
        if args.temporal_phase_context:
            model.enable_temporal_phase_context()
        model.periodic_context_only = args.periodic_context_only
        model.recover_missing_beats = args.recover_missing_beats
        source_was_frame_bin = False
        if args.learn_tempo:
            load_dir = args.resume_dir or args.phase_dir
            previous_checkpoint = torch.load(load_dir / f"seed{seed}.pt", weights_only=True)
            previous_state = previous_checkpoint["state"]
            source_was_frame_bin = previous_checkpoint.get("frame_bin_emission", False)
            if args.fresh_audio_prior or (
                args.phase_attention and "phase_attention.weight" not in previous_state
            ):
                if not (args.landmark_emission or args.clock_mass_emission):
                    model.decoder.load_state_dict(
                        {
                            n.removeprefix("decoder."): p
                            for n, p in previous_state.items()
                            if n.startswith("decoder.")
                        }
                    )
            else:
                missing = model.load_previous(previous_state)
                assert not missing.unexpected_keys and all(
                    n.startswith("temporal_phase_context.")
                    or n.startswith("tempo_head.")
                    or (args.phase_residual and n.startswith("phase_residual."))
                    or (
                        (args.landmark_emission or args.clock_mass_emission)
                        and n.startswith("decoder.")
                    )
                    for n in missing.missing_keys
                )
        else:
            if args.resume_dir is not None:
                previous_checkpoint = torch.load(
                    args.resume_dir / f"seed{seed}.pt", weights_only=True
                )
                if args.fresh_audio_prior:
                    # A replacement emission has its own parameterization;
                    # initialize it independently, as in the joint path.
                    if not (args.landmark_emission or args.clock_mass_emission):
                        model.decoder.load_state_dict(
                            {
                                n.removeprefix("decoder."): p
                                for n, p in previous_checkpoint["state"].items()
                                if n.startswith("decoder.")
                            }
                        )
                else:
                    missing = model.load_previous(previous_checkpoint["state"])
                    assert not missing.unexpected_keys and all(
                        n.startswith("temporal_phase_context.")
                        or (args.phase_residual and n.startswith("phase_residual."))
                        or (
                            (args.landmark_emission or args.clock_mass_emission)
                            and n.startswith("decoder.")
                        )
                        for n in missing.missing_keys
                    )
                source_was_frame_bin = previous_checkpoint.get("frame_bin_emission", False)
            else:
                state = torch.load(
                    ROOT / f"runs/sohn_restart_frame_rate_native_encoder/seed{seed}.pt",
                    weights_only=True,
                )["state"]
                if not (args.landmark_emission or args.clock_mass_emission):
                    model.decoder.load_state_dict(
                        {
                            n.removeprefix("decoder."): p
                            for n, p in state.items()
                            if n.startswith("decoder.")
                        }
                    )
        if args.frame_bin_emission and not source_was_frame_bin:
            model.decoder.preserve_point_peak_heights()
        if args.reset_concentration is not None:
            with torch.no_grad():
                model.phase_head[-1].weight[-1].zero_()
                model.phase_head[-1].bias[-1].fill_(math.log(args.reset_concentration - 1))
        if args.landmark_initial_kappa is not None:
            with torch.no_grad():
                model.decoder.log_kappa.fill_(math.log(args.landmark_initial_kappa))
        if args.clock_initial_width is not None:
            fraction = (args.clock_initial_width - 0.001) / (math.pi - 0.001)
            with torch.no_grad():
                model.decoder.raw_width.fill_(math.log(fraction / (1 - fraction)))
        if args.initial_tempo_sigma is not None:
            with torch.no_grad():
                model.log_initial_sigma.fill_(math.log(args.initial_tempo_sigma))
        with torch.no_grad():
            baseline_prediction = model.prediction_for(data["x"])
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        generator = torch.Generator().manual_seed(97000 + seed)
        origin_generator = torch.Generator().manual_seed(147000 + seed)
        history = []
        for step in range(args.steps + 1):
            if step % 100 == 0 or step == args.steps:
                model.proposal_shift = None
                row = dict(seed=seed, step=step, **score(model, data))
                history.append(row)
                print(json.dumps(row), flush=True)
            if step == args.steps:
                break
            count = min(args.batch_size, len(data["x"]))
            indices = (
                torch.arange(count)
                if count == len(data["x"])
                else torch.randperm(len(data["x"]), generator=generator)[:count]
            )
            if args.augment_origin:
                model.proposal_shift = torch.randint(4, (count,), generator=origin_generator) * (
                    math.pi / 2
                )
            model.cached_prediction = {
                key: value[indices] for key, value in baseline_prediction.items()
            }
            noise = dict(
                uniform=torch.rand(4, count, generator=generator).clamp(1e-6, 1 - 1e-6),
                initial=torch.randn(4, count, generator=generator),
                increments=torch.randn(4, count, data["x"].shape[1] - 2, generator=generator),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(data["x"][indices], data["labels"][indices], noise)
            loss.backward()
            if not torch.isfinite(loss) or any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise RuntimeError("Missing/nonfinite GSNN gradient")
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            model.cached_prediction = None
        changes = {n: float((p.detach() - initial[n]).norm()) for n, p in model.named_parameters()}
        assert all(v > 0 for v in changes.values()), "Every allocated learned tensor must update"
        model.proposal_shift = None
        torch.save(
            dict(
                state=model.state_dict(),
                seed=seed,
                temporal_phase_context=args.temporal_phase_context,
                bernoulli_clock_emission=args.bernoulli_clock_emission,
                tempo_feedback=args.tempo_feedback,
                recover_missing_beats=args.recover_missing_beats,
                periodic_context_only=args.periodic_context_only,
                learn_tempo=args.learn_tempo,
                aligned_context=args.aligned_context,
                proposal_max_bpm=args.proposal_max_bpm,
                proposal_min_probability=args.proposal_min_probability,
                tempo_log_scale=args.tempo_log_scale,
                tempo_basis=args.tempo_basis,
                phase_attention=args.phase_attention,
                smooth_concentration=args.smooth_concentration,
                frame_bin_emission=args.frame_bin_emission,
                angular_frame_bin_emission=args.angular_frame_bin_emission,
                landmark_emission=args.landmark_emission,
                clock_mass_emission=args.clock_mass_emission,
                log_tempo_noise=args.log_tempo_noise,
                phase_residual=args.phase_residual,
            ),
            args.output / f"seed{seed}.pt",
        )
        report["results"].append(
            dict(seed=seed, scores=history[-1], history=history, parameter_changes=changes)
        )
        save()
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
