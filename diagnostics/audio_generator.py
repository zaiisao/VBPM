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
from diagnostics.audio_metrics import score
from diagnostics.real_event_references import attach_references
from diagnostics.synthetic_ladder import ROOT
from vbpm.util.vonmises import _VonMisesInvCDF as _VonMisesInvCDF
from diagnostics.generator_variants import AttentionGenerator as AudioPhaseAttentionGSNN
from diagnostics.generator_variants import Generator as AudioPhaseFirstGSNN
from diagnostics.generator_variants import PhaseContextNormalization as PhaseContextNormalization


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
        "--clock-mass-emission",
        action="store_true",
        help="Use normalized event mass per clock landmark with learned angular timing uncertainty",
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
    if args.clock_mass_emission and args.frame_bin_emission:
        parser.error("Select one emission formulation")
    if args.clock_initial_width is not None and (
        not args.clock_mass_emission or not 0.001 < args.clock_initial_width < math.pi
    ):
        parser.error(
            "Clock width must be between .001 and pi radians and requires normalized clock emission"
        )
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
        clock_mass_emission=args.clock_mass_emission,
        temporal_phase_context=args.temporal_phase_context,
        bernoulli_clock_emission=args.bernoulli_clock_emission,
        clock_initial_width=args.clock_initial_width,
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
                if not args.clock_mass_emission:
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
                        args.clock_mass_emission
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
                    if not args.clock_mass_emission:
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
                            args.clock_mass_emission
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
                if not args.clock_mass_emission:
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
