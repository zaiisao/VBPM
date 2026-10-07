"""Import checkpoints from staged GSNN experiments."""

import math

import torch


def import_staged_checkpoint(model, path):
    """Load experimental prior and emission weights into VBPM."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not checkpoint.get("bernoulli_clock_emission") or not checkpoint.get("aligned_context"):
        raise ValueError("Checkpoint must use the aligned Bernoulli-clock generator")
    if checkpoint.get("log_tempo_noise") or checkpoint.get("temporal_phase_context"):
        raise ValueError("This integration uses the unadapted additive Gaussian staged dynamics")
    if checkpoint.get("tempo_basis", "framewise") != "framewise":
        raise ValueError("Checkpoint has a different tempo basis")
    if (
        checkpoint.get("phase_attention")
        or checkpoint.get("tempo_feedback")
        or checkpoint.get("periodic_context_only")
    ):
        raise ValueError("Checkpoint uses an unsupported attention or feedback variant")
    if not checkpoint.get("smooth_concentration", False):
        raise ValueError("Checkpoint must use smooth phase concentration")
    model.prior_model.load_state_dict(
        {k: v for k, v in checkpoint["state"].items() if not k.startswith("decoder.")},
        strict=True,
    )
    for name, default in (
        ("recover_missing_beats", False),
        ("proposal_max_bpm", None),
        ("proposal_min_probability", None),
        ("tempo_log_scale", 0.1),
    ):
        value = checkpoint.get(name, default)
        if name == "proposal_max_bpm" and value is None:
            value = math.inf
        if name == "proposal_min_probability" and value is None:
            value = 0.3
        setattr(model.prior_model, name, value)

    model.emission_model.load_state_dict(
        {
            k.removeprefix("decoder."): v
            for k, v in checkpoint["state"].items()
            if k.startswith("decoder.")
        },
        strict=True,
    )


def prior_settings(prior):
    """Proposal and tempo settings stored alongside model weights."""
    return {
        "recover_missing_beats": prior.recover_missing_beats,
        "proposal_max_bpm": prior.proposal_max_bpm,
        "proposal_min_probability": prior.proposal_min_probability,
        "tempo_log_scale": prior.tempo_log_scale,
    }
