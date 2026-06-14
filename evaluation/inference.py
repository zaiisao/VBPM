"""Inference utilities and CLI for CHART.

At inference time, beat annotations (b_{1:T}) are unavailable. The model
uses the PRIOR (not posterior) to generate latent trajectories from audio
alone. The prior means are audio-driven (g^φ_ψ(h_t), g^φ̇_ψ(h_t) corrections
on the bar-pointer recursion), so the rolled-out dynamics track the signal
rather than free-running an uninformed random walk.

Two beat read-outs are available and reported side-by-side:
  - phase-wrap: beats from phase wrap-arounds of the rolled-out latent
    (the bar pointer's own dynamics);
  - decoder:    beats from the decoder's beat channel (also audio-driven).

This mirrors Stable Diffusion's approach: the encoder (posterior) provides
structured training targets, but inference uses the generative model
(prior) without the encoder.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from models.svt_core import SVTModel
from evaluation.phase_converter import (
    extract_beat_timestamps,
    extract_beats_from_phase_trajectory,
)

TWO_PI = 2.0 * math.pi


def _select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = bool(getattr(torch, "has_mps", False))
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CHART inference and save predictions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_npy", type=str, required=True,
                        help="Input acoustic activations (.npy) with shape [T,2] or [B,T,2]")
    parser.add_argument("--output_npy", type=str, required=True,
                        help="Path to save predictions (.npy)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_meter_classes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Gumbel-Softmax temperature for meter (low = more discrete)")
    parser.add_argument("--fps", type=float, default=86.1328125,
                        help="Frames per second of the input activations")
    return parser


def _load_svt_model(
    checkpoint_path: str,
    device: torch.device,
    hidden_dim: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    num_meter_classes: int = 8,
) -> SVTModel:
    ckpt: Any = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model args from checkpoint if available
    if isinstance(ckpt, dict) and "args" in ckpt:
        saved_args = ckpt["args"]
        num_meter_classes = saved_args.get("num_meter_classes", num_meter_classes)

    svt_state = ckpt.get("svt_model", ckpt) if isinstance(ckpt, dict) else ckpt

    model = SVTModel(
        hidden_dim=hidden_dim, nhead=nhead, num_layers=num_layers,
        num_meter_classes=num_meter_classes,
    ).to(device)
    model.load_state_dict(svt_state, strict=True)
    model.eval()
    return model


def _prepare_activations(input_path: str, device: torch.device) -> Tensor:
    activations_np = np.load(input_path)
    activations = torch.as_tensor(activations_np, dtype=torch.float32)

    if activations.ndim == 2:
        if activations.shape[1] != 2:
            raise ValueError(f"Expected [T,2] input when ndim=2, got {tuple(activations.shape)}")
        activations = activations.unsqueeze(0)
    elif activations.ndim == 3:
        if activations.shape[2] != 2:
            raise ValueError(f"Expected [B,T,2] input when ndim=3, got {tuple(activations.shape)}")
    else:
        raise ValueError(f"Expected input shape [T,2] or [B,T,2], got {tuple(activations.shape)}")

    return activations.to(device)


@torch.no_grad()
def run_inference(
    model: SVTModel,
    acoustic_activations: Tensor,
    *,
    temperature: float = 0.1,
    fps: float = 86.1328125,
) -> dict[str, Tensor | np.ndarray]:
    """Run CHART inference using the PRIOR (no beat annotations needed).

    The prior encoder processes the audio to learn uncertainty parameters.
    The sequential loop samples from the prior's transition model
    (bar pointer dynamics) to generate phase/tempo/meter trajectories.
    Beats are extracted from phase wrap-arounds.

    Args:
        model: Trained SVT model.
        acoustic_activations: ``[B, T, 2]`` acoustic features.
        temperature: Gumbel-Softmax temperature for meter.
        fps: Frames per second for beat timestamp extraction.

    Returns:
        Dict with:
        - ``beat_times``: list of 1-D arrays of beat timestamps (seconds) per batch
        - ``phase``: ``[B, T]`` phase trajectory
        - ``log_tempo``: ``[B, T]`` log-tempo trajectory
        - ``tempo``: ``[B, T]`` tempo in rad/frame
        - ``meter``: ``[B, T]`` meter class index
        - ``beat_logits``: ``[B, T]`` decoder beat logits (for comparison)
        - ``beat_probs``: ``[B, T]`` sigmoid of beat_logits
    """
    B = acoustic_activations.shape[0]

    # Sample latent trajectory from the prior alone (Algorithm 1, prior-only).
    out = model.sample_from_prior(acoustic_activations, temperature=temperature)

    phase_traj = out["phase"]                                      # [B, T]
    log_tempo_traj = out["log_tempo"]                              # [B, T]
    meter_traj = out["meter_onehot"].argmax(dim=-1)                # [B, T]
    beat_logits = out["beat_logits"][:, :, 0]                      # [B, T]
    db_logits = out["beat_logits"][:, :, 1]                        # [B, T]

    # Extract beats two ways: phase wrap-arounds (dynamics) and the decoder
    # beat channel (also audio-driven). Downbeats: decoder downbeat channel.
    phase_np = phase_traj.cpu().numpy()
    beat_probs_np = torch.sigmoid(beat_logits).cpu().numpy()
    db_probs_np = torch.sigmoid(db_logits).cpu().numpy()
    beat_times_list, beat_times_decoder_list, downbeat_times_list = [], [], []
    for b in range(B):
        beat_times_list.append(extract_beats_from_phase_trajectory(phase_np[b], fps=fps))
        beat_times_decoder_list.append(extract_beat_timestamps(beat_probs_np[b], fps=fps))
        downbeat_times_list.append(extract_beat_timestamps(db_probs_np[b], fps=fps))

    return {
        "beat_times": beat_times_list,             # phase-wrap read-out (primary)
        "beat_times_decoder": beat_times_decoder_list,
        "downbeat_times": downbeat_times_list,
        "phase": phase_traj,
        "log_tempo": log_tempo_traj,
        "tempo": log_tempo_traj.exp(),
        "meter": meter_traj,
        "beat_logits": beat_logits,
        "beat_probs": torch.sigmoid(beat_logits),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    device = _select_device(args.device)
    model = _load_svt_model(
        checkpoint_path=args.checkpoint,
        device=device,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_meter_classes=args.num_meter_classes,
    )

    activations = _prepare_activations(args.input_npy, device)
    results = run_inference(
        model=model,
        acoustic_activations=activations,
        temperature=args.temperature,
        fps=args.fps,
    )

    # Save phase-based beat timestamps (primary output)
    beat_times = results["beat_times"][0]  # first batch element
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, beat_times)

    beat_times_decoder = results["beat_times_decoder"][0]
    downbeat_times = results["downbeat_times"][0]
    print(f"Saved {len(beat_times)} phase-wrap beat timestamps to: {output_path}")
    print(f"  decoder read-out: {len(beat_times_decoder)} beats | {len(downbeat_times)} downbeats")
    if len(beat_times) > 1:
        avg_bpm = 60.0 / np.mean(np.diff(beat_times))
        print(f"Estimated tempo (phase-wrap): {avg_bpm:.1f} BPM")

    # Also save full trajectories
    traj_path = output_path.with_suffix(".trajectories.npz")
    np.savez(
        traj_path,
        beat_times=beat_times,
        beat_times_decoder=beat_times_decoder,
        downbeat_times=downbeat_times,
        phase=results["phase"][0].cpu().numpy(),
        tempo=results["tempo"][0].cpu().numpy(),
        meter=results["meter"][0].cpu().numpy(),
        beat_probs=results["beat_probs"][0].cpu().numpy(),
    )
    print(f"Saved full trajectories to: {traj_path}")


if __name__ == "__main__":
    main()
