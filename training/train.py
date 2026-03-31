"""Training entrypoints for CHART."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping, Sequence

import heapq
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from models.svt_core import SVTModel
from models.loss import compute_elbo_loss
from evaluation.phase_converter import extract_beat_timestamps, extract_downbeat_timestamps, extract_beats_from_phase_trajectory
from evaluation.score import evaluate_beats, evaluate_downbeats, frames_to_beat_times
from training.dataset import ActivationDataset
from training.extractors import get_extractor_backend, list_extractor_backends
from training.extractors.base import ExtractorBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = bool(getattr(torch, "has_mps", False))
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def _gumbel_temperature(epoch: int, num_epochs: int, start: float, end: float) -> float:
    """Linear annealing of Gumbel-Softmax temperature."""
    if num_epochs <= 1:
        return end
    t = min(epoch / max(num_epochs - 1, 1), 1.0)
    return start + (end - start) * t


def _kl_beta(epoch: int, anneal_epochs: int) -> float:
    """Linear KL annealing from 0 to 1."""
    if anneal_epochs <= 0:
        return 1.0
    return min(epoch / anneal_epochs, 1.0)


def _build_z_prev(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    """Extract z_prev dict from a batch mapping."""
    return {
        "phase": batch["phase_prev"].to(device),
        "log_tempo": batch["log_tempo_prev"].to(device),
        "meter_onehot": batch["meter_onehot_prev"].to(device),
    }


def _center_crop_seq_dim(x: Tensor, length: int) -> Tensor:
    if x.shape[1] == length:
        return x
    if x.shape[1] < length:
        raise ValueError(f"Cannot crop sequence length {x.shape[1]} to larger length {length}")
    start = (x.shape[1] - length) // 2
    return x[:, start : start + length]


def _center_crop_seq_dim_1d(x: Tensor, length: int) -> Tensor:
    """Center-crop a 1D or 2D tensor along dim 0 (seq dimension)."""
    if x.shape[0] == length:
        return x
    if x.shape[0] < length:
        raise ValueError(f"Cannot crop length {x.shape[0]} to {length}")
    start = (x.shape[0] - length) // 2
    return x[start : start + length]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_beat_viz(
    out: dict,
    beat_targets: Tensor,
    epoch: int,
    log_dir: str,
    fps: float = 86.1328125,
    gt_phase: np.ndarray | None = None,
    gt_z_prev: dict[str, Tensor] | None = None,
    downbeat_targets: Tensor | None = None,
) -> str | None:
    """Save detailed diagnostic visualization for the first sample."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from models.distributions import von_mises_kl, lognormal_kl
    except ImportError:
        return None

    TWO_PI = 2 * np.pi
    beat_logits_all = out["beat_logits"][0].detach().cpu().numpy()  # [T, 2]
    logits = beat_logits_all[:, 0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    db_logits = beat_logits_all[:, 1] if beat_logits_all.shape[1] > 1 else None
    db_probs = 1.0 / (1.0 + np.exp(-db_logits)) if db_logits is not None else None
    gt = beat_targets[0].detach().cpu().numpy()
    phase = out["samples"]["phase"][0].detach().cpu().numpy()
    log_tempo = out["samples"]["log_tempo"][0].detach().cpu().numpy()
    T = len(probs)

    beat_frames = np.where(gt > 0.5)[0]
    db_frames = list(np.where(gt_phase[:T] > 0.5)[0]) if gt_phase is not None else []
    tempo = np.exp(np.clip(log_tempo, -10, 10))
    bpm = tempo * 60 * fps / TWO_PI
    phase_wrapped = phase % TWO_PI

    # GT sawtooth from z_prev if available
    gt_phase_rad = None
    gt_bpm = None
    if gt_z_prev is not None and "phase" in gt_z_prev:
        gp = gt_z_prev["phase"][0, :T].detach().cpu().numpy()
        gt_phase_rad = (gp[:, 0] if gp.ndim == 2 else gp) % TWO_PI
        glt = gt_z_prev["log_tempo"][0, :T].detach().cpu().numpy()
        gt_log_tempo = glt[:, 0] if glt.ndim == 2 else glt
        gt_bpm = np.exp(np.clip(gt_log_tempo, -10, 10)) * 60 * fps / TWO_PI

    # Prior vs posterior
    post_phase_mu = out["posterior"]["phase_mu"][0].detach().cpu().numpy() % TWO_PI
    prior_phase_mu = out["prior"]["phase_mu"][0].detach().cpu().numpy() % TWO_PI
    prior_kappa = out["prior"]["phase_kappa"][0].detach().cpu().numpy()
    prior_sigma = out["prior"]["tempo_sigma"][0].detach().cpu().numpy()

    # Per-frame KL
    kl_phase_t = von_mises_kl(
        out["posterior"]["phase_mu"][0], torch.exp(out["posterior"]["phase_log_kappa"][0]),
        out["prior"]["phase_mu"][0], out["prior"]["phase_kappa"][0],
    ).detach().cpu().numpy()
    kl_tempo_t = lognormal_kl(
        out["posterior"]["tempo_mu"][0], torch.exp(out["posterior"]["tempo_log_sigma"][0]),
        out["prior"]["tempo_mu"][0], out["prior"]["tempo_sigma"][0],
    ).detach().cpu().numpy()

    fig, axes = plt.subplots(7, 1, figsize=(22, 24), sharex=True)

    # 0. Beat probability + GT beats/downbeats
    axes[0].plot(probs, "b-", lw=0.8, label="P(beat)")
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.3)
    for bf in beat_frames:
        color = "r" if bf in db_frames else "g"
        axes[0].axvline(bf, color=color, alpha=0.4, lw=1.5 if bf in db_frames else 1)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"Epoch {epoch}: Beat P (blue) | GT beats (green) | GT downbeats (red)")
    axes[0].legend()

    # 1. Downbeat probability + GT downbeats
    if db_probs is not None:
        axes[1].plot(db_probs, "b-", lw=0.8, label="P(downbeat)")
        axes[1].axhline(0.5, color="gray", ls="--", alpha=0.3)
    if downbeat_targets is not None:
        db_gt = downbeat_targets[0].detach().cpu().numpy()[:T]
        db_gt_frames = np.where(db_gt > 0.5)[0]
        for df in db_gt_frames:
            axes[1].axvline(df, color="r", alpha=0.5, lw=1.5)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Downbeat P (blue) | GT downbeats (red)")
    axes[1].legend()

    # 2. Phase: model vs GT sawtooth
    axes[2].plot(phase_wrapped, "purple", lw=0.5, alpha=0.7, label="model phase")
    if gt_phase_rad is not None:
        axes[2].plot(gt_phase_rad, "green", lw=0.5, alpha=0.5, label="GT phase (sawtooth)")
    axes[2].axhline(0, color="k", ls=":", alpha=0.2)
    axes[2].axhline(TWO_PI, color="k", ls=":", alpha=0.2)
    axes[2].set_ylabel("Phase mod 2π")
    axes[2].set_title("Phase: model (purple) vs GT sawtooth (green)")
    axes[2].legend()

    # 3. Prior vs Posterior phase mu
    axes[3].plot(post_phase_mu, "blue", lw=0.5, alpha=0.7, label="posterior μ_φ")
    axes[3].plot(prior_phase_mu, "red", lw=0.5, alpha=0.7, label="prior μ_φ (from GT z_prev)")
    if gt_phase_rad is not None:
        axes[3].plot(gt_phase_rad, "green", lw=0.3, alpha=0.3, label="GT phase")
    axes[3].set_ylabel("μ_φ mod 2π")
    axes[3].set_title("Prior mean (red, from GT) vs Posterior mean (blue) — should overlap")
    axes[3].legend()

    # 4. Tempo BPM: model vs GT
    axes[4].plot(bpm, "orange", lw=0.5, alpha=0.7, label="model BPM")
    if gt_bpm is not None:
        axes[4].plot(gt_bpm, "green", lw=0.5, alpha=0.5, label="GT BPM")
    axes[4].set_ylabel("BPM")
    axes[4].set_title(f"Tempo: model mean={bpm.mean():.0f} BPM" + (f", GT mean={gt_bpm.mean():.0f}" if gt_bpm is not None else ""))
    axes[4].legend()

    # 5. Per-frame KL
    axes[5].plot(kl_phase_t, "purple", lw=0.5, label=f"KL phase (mean={kl_phase_t.mean():.2f})")
    axes[5].plot(kl_tempo_t, "orange", lw=0.5, label=f"KL tempo (mean={kl_tempo_t.mean():.2f})")
    axes[5].set_ylabel("KL (nats)")
    axes[5].set_title("Per-frame KL — high means posterior disagrees with GT-anchored prior")
    axes[5].legend()

    # 6. Prior uncertainty
    ax6b = axes[6].twinx()
    axes[6].plot(prior_kappa, "darkblue", lw=0.8, label=f"κ (mean={prior_kappa.mean():.0f})")
    ax6b.plot(prior_sigma, "darkred", lw=0.8, label=f"σ (mean={prior_sigma.mean():.4f})")
    axes[6].set_ylabel("κ_φ", color="darkblue")
    ax6b.set_ylabel("σ_tempo", color="darkred")
    axes[6].set_title("Prior uncertainty: κ should be HIGH (concentrated), σ should be LOW (stable tempo)")
    axes[6].legend(loc="upper left"); ax6b.legend(loc="upper right")
    axes[6].set_xlabel("Frame")

    plt.tight_layout()
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"beat_viz_ep{epoch:03d}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _tensor_stats(t: Tensor) -> str:
    """One-line summary: min, max, mean, nan/inf counts."""
    n = t.numel()
    nans = int(torch.isnan(t).sum().item())
    infs = int(torch.isinf(t).sum().item())
    if nans == n:
        return f"ALL_NAN({n})"
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return f"nan={nans} inf={infs} no_finite({n})"
    return (f"[{finite.min().item():.4g}, {finite.max().item():.4g}] "
            f"μ={finite.mean().item():.4g} nan={nans} inf={infs}")


def _dump_diagnostics(
    out: dict,
    z_prev_sampled: dict[str, Tensor],
    svt_total: Tensor,
    components: dict[str, Tensor],
    epoch: int,
    step: int,
) -> None:
    """Print detailed diagnostics when something goes wrong."""
    sys.stdout.write(f"\n{'='*70}\n")
    sys.stdout.write(f"  DIAGNOSTICS: epoch={epoch} step={step}\n")
    sys.stdout.write(f"{'='*70}\n")

    # Loss components
    sys.stdout.write(f"  svt_total = {svt_total.item() if torch.isfinite(svt_total) else svt_total.item()}\n")
    for k, v in components.items():
        sys.stdout.write(f"  {k} = {v.item()}\n")

    # z_prev_sampled (input to pass 2)
    sys.stdout.write(f"  --- z_prev_sampled (pass 1 rollout → pass 2 input) ---\n")
    for k, v in z_prev_sampled.items():
        sys.stdout.write(f"  z_prev.{k}: {_tensor_stats(v)}\n")

    # Posterior parameters
    sys.stdout.write(f"  --- Posterior parameters ---\n")
    post = out["posterior"]
    for k, v in post.items():
        sys.stdout.write(f"  post.{k}: {_tensor_stats(v)}\n")
    # Derived values
    sys.stdout.write(f"  post.kappa_q (exp log_kappa): {_tensor_stats(post['phase_log_kappa'].exp())}\n")
    sys.stdout.write(f"  post.sigma_q (exp log_sigma): {_tensor_stats(post['tempo_log_sigma'].exp())}\n")

    # Prior parameters
    sys.stdout.write(f"  --- Prior parameters ---\n")
    pri = out["prior"]
    for k, v in pri.items():
        sys.stdout.write(f"  prior.{k}: {_tensor_stats(v)}\n")

    # Samples (from pass 2)
    sys.stdout.write(f"  --- Samples (pass 2) ---\n")
    samp = out["samples"]
    for k, v in samp.items():
        sys.stdout.write(f"  samp.{k}: {_tensor_stats(v)}\n")

    # Beat logits
    sys.stdout.write(f"  beat_logits: {_tensor_stats(out['beat_logits'])}\n")

    # Per-element KL check (not reduced)
    from models.distributions import von_mises_kl, lognormal_kl, categorical_kl
    kl_m_raw = categorical_kl(post["meter_logits"], pri["meter_logits"])
    kl_phi_raw = von_mises_kl(post["phase_mu"], post["phase_log_kappa"].exp(),
                               pri["phase_mu"], pri["phase_kappa"])
    kl_t_raw = lognormal_kl(post["tempo_mu"], post["tempo_log_sigma"].exp(),
                             pri["tempo_mu"], pri["tempo_sigma"])
    sys.stdout.write(f"  --- Raw KL (per element, before free_bits) ---\n")
    sys.stdout.write(f"  kl_meter:  {_tensor_stats(kl_m_raw)}\n")
    sys.stdout.write(f"  kl_phase:  {_tensor_stats(kl_phi_raw)}\n")
    sys.stdout.write(f"  kl_tempo:  {_tensor_stats(kl_t_raw)}\n")

    sys.stdout.write(f"{'='*70}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Training epochs
# ---------------------------------------------------------------------------

def train_epoch(
    model: SVTModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    temperature: float = 1.0,
    beta: float = 1.0,
    pos_weight: float = 20.0,
    max_grad_norm: float = 1.0,
    epoch: int = 1,
    num_epochs: int = 1,
    log_interval: int = 1,
) -> tuple[float, dict[str, float]]:
    """Run one training epoch (teacher-forced parallel mode).

    Returns:
        Tuple of (avg_total_loss, avg_component_dict).
    """
    model.train()

    total_sum = 0.0
    comp_sums: dict[str, float] = {}
    num_batches = 0
    num_total_batches = max(1, len(dataloader))

    for batch_idx, batch in enumerate(dataloader, start=1):
        activations = batch["activations"].to(device)
        beat_targets = batch["beat_targets"].to(device)
        z_prev = _build_z_prev(batch, device)

        optimizer.zero_grad(set_to_none=True)

        # Algorithm 1: sequential autoregressive (seed from first frame)
        z_prev_init = {
            "phase": z_prev["phase"][:, :1, :],
            "log_tempo": z_prev["log_tempo"][:, :1, :],
            "meter_onehot": z_prev["meter_onehot"][:, :1, :],
        }
        out = model(activations, z_prev_init, temperature=temperature)

        total_loss, components = compute_elbo_loss(
            beat_logits=out["beat_logits"],
            beat_targets=beat_targets,
            posterior=out["posterior"],
            prior=out["prior"],
            beta=beta,
            pos_weight=pos_weight,
        )

        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_sum += float(total_loss.detach().item())
        for k, v in components.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
        num_batches += 1

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_total_batches):
            avg = total_sum / num_batches
            sys.stdout.write(
                f"\r[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"step {batch_idx:04d}/{num_total_batches:04d} "
                f"total={avg:.6f}"
            )
            sys.stdout.flush()

    if num_batches > 0:
        sys.stdout.write("\n")
    if num_batches == 0:
        return 0.0, {}

    avg_total = total_sum / num_batches
    avg_comps = {k: v / num_batches for k, v in comp_sums.items()}
    return avg_total, avg_comps


def train_epoch_end_to_end(
    extractor_model: torch.nn.Module,
    extractor_backend: ExtractorBackend,
    svt_model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    temperature: float = 1.0,
    beta: float = 1.0,
    pos_weight: float = 20.0,
    free_bits: float = 0.0,
    free_bits_meter: float | None = None,
    free_bits_phase: float | None = None,
    free_bits_tempo: float | None = None,
    max_grad_norm: float = 1.0,
    extractor_loss_weight: float = 1.0,
    svt_loss_weight: float = 1.0,
    epoch: int = 1,
    num_epochs: int = 1,
    log_interval: int = 1,
    is_main: bool = True,
    smooth_sigma: float = 3.0,
    smooth_sigma_db: float = 5.0,
) -> tuple[float, float, float, dict[str, float]]:
    """Run one end-to-end training epoch.

    Returns:
        (avg_total, avg_extractor, avg_svt, avg_components).
    """
    extractor_model.train()
    svt_model.train()

    total_sum = 0.0
    extractor_sum = 0.0
    svt_sum = 0.0
    comp_sums: dict[str, float] = {}
    num_batches = 0
    num_total_batches = max(1, len(dataloader))

    for batch_idx, batch in enumerate(dataloader, start=1):
        audio = batch["audio"].to(device)
        extractor_target = batch["extractor_target"].to(device)
        beat_targets = batch["beat_targets"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Stage 1: Extractor forward
        extractor_loss, activations = extractor_backend.compute_loss_and_activations(
            model=extractor_model, audio=audio, target=extractor_target,
        )

        # Crop structured targets to match extractor output length
        T_act = activations.shape[1]
        beat_targets_cropped = _center_crop_seq_dim(beat_targets.unsqueeze(-1), T_act).squeeze(-1)

        # Downbeat targets from extractor_target channel 1
        downbeat_targets_cropped = _center_crop_seq_dim(
            extractor_target[:, 1:2, :].transpose(1, 2), T_act
        ).squeeze(-1)

        # Full GT z_prev trajectory (prior anchored to GT dynamics)
        z_prev_gt = {
            "phase": _center_crop_seq_dim(batch["phase_prev"].to(device), T_act),
            "log_tempo": _center_crop_seq_dim(batch["log_tempo_prev"].to(device), T_act),
            "meter_onehot": _center_crop_seq_dim(batch["meter_onehot_prev"].to(device), T_act),
        }

        # Parallel forward: prior uses GT z_prev, posterior sampled independently
        out = svt_model(activations, z_prev_gt, temperature=temperature,
                        beat_targets=beat_targets_cropped,
                        downbeat_targets=downbeat_targets_cropped)
        svt_total, components = compute_elbo_loss(
            beat_logits=out["beat_logits"],
            beat_targets=beat_targets_cropped,
            posterior=out["posterior"],
            prior=out["prior"],
            beta=beta,
            pos_weight=pos_weight,
            free_bits=free_bits,
            free_bits_meter=free_bits_meter,
            free_bits_phase=free_bits_phase,
            free_bits_tempo=free_bits_tempo,
            downbeat_targets=downbeat_targets_cropped,
            smooth_sigma=smooth_sigma,
            smooth_sigma_db=smooth_sigma_db,
        )

        total_loss = extractor_loss_weight * extractor_loss + svt_loss_weight * svt_total

        # Guard against NaN/Inf — dump diagnostics on first occurrence, then skip
        if not torch.isfinite(total_loss):
            optimizer.zero_grad(set_to_none=True)
            if is_main:
                sys.stdout.write(f"\n  [WARN] NaN/Inf loss at epoch {epoch} step {batch_idx}\n")
                _dump_diagnostics(out, z_prev_gt, svt_total, components, epoch, batch_idx)
            continue

        total_loss.backward()
        all_params = list(extractor_model.parameters()) + list(svt_model.parameters())

        # Check for NaN/Inf in gradients BEFORE clipping/stepping.
        # clip_grad_norm_ does not handle NaN — it propagates them.
        bad_grad = False
        for name, p in list(extractor_model.named_parameters()) + list(svt_model.named_parameters()):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                if is_main and not bad_grad:
                    nan_ct = int(torch.isnan(p.grad).sum().item())
                    inf_ct = int(torch.isinf(p.grad).sum().item())
                    sys.stdout.write(
                        f"\n  [GRAD] NaN/Inf gradient at epoch {epoch} step {batch_idx}: "
                        f"{name} (nan={nan_ct} inf={inf_ct} / {p.grad.numel()})\n"
                    )
                    sys.stdout.flush()
                bad_grad = True
                break

        if bad_grad:
            optimizer.zero_grad(set_to_none=True)
            continue

        clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()

        total_sum += float(total_loss.detach().item())
        extractor_sum += float(extractor_loss.detach().item())
        svt_sum += float(svt_total.detach().item())
        for k, v in components.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
        num_batches += 1

        if is_main and log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_total_batches):
            sys.stdout.write(
                f"\r[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"step {batch_idx:04d}/{num_total_batches:04d} "
                f"total={total_sum / num_batches:.6f} "
                f"ext={extractor_sum / num_batches:.6f} "
                f"svt={svt_sum / num_batches:.6f}"
            )
            sys.stdout.flush()
            if _WANDB_AVAILABLE and _wandb.run is not None:
                global_step = (epoch - 1) * num_total_batches + batch_idx
                step_log: dict = {
                    "global_step": global_step,
                    "train_step/total_loss": total_sum / num_batches,
                    "train_step/ext_loss": extractor_sum / num_batches,
                    "train_step/svt_loss": svt_sum / num_batches,
                }
                for k, v in comp_sums.items():
                    step_log[f"train_step/{k}"] = v / num_batches
                # Log parameter range diagnostics per step
                post = out["posterior"]
                pri = out["prior"]
                samp2 = out["samples"]
                step_log["diag/post_tempo_log_sigma_max"] = float(post["tempo_log_sigma"].max().item())
                step_log["diag/post_tempo_log_sigma_min"] = float(post["tempo_log_sigma"].min().item())
                step_log["diag/post_phase_log_kappa_max"] = float(post["phase_log_kappa"].max().item())
                step_log["diag/prior_tempo_sigma_max"] = float(pri["tempo_sigma"].max().item())
                step_log["diag/prior_tempo_sigma_min"] = float(pri["tempo_sigma"].min().item())
                step_log["diag/prior_phase_kappa_max"] = float(pri["phase_kappa"].max().item())
                step_log["diag/samp_log_tempo_max"] = float(samp2["log_tempo"].max().item())
                step_log["diag/samp_log_tempo_min"] = float(samp2["log_tempo"].min().item())
                step_log["diag/samp_log_tempo_max2"] = float(out["samples"]["log_tempo"].max().item())
                step_log["diag/samp_log_tempo_min2"] = float(out["samples"]["log_tempo"].min().item())
                step_log["diag/prior_tempo_mu_max"] = float(pri["tempo_mu"].max().item())
                step_log["diag/prior_tempo_mu_min"] = float(pri["tempo_mu"].min().item())
                step_log["diag/post_tempo_mu_max"] = float(post["tempo_mu"].max().item())
                step_log["diag/post_tempo_mu_min"] = float(post["tempo_mu"].min().item())
                _wandb.log(step_log)

    if is_main and num_batches > 0:
        sys.stdout.write("\n")
    if num_batches == 0:
        return 0.0, 0.0, 0.0, {}

    avg_comps = {k: v / num_batches for k, v in comp_sums.items()}
    return (
        total_sum / num_batches,
        extractor_sum / num_batches,
        svt_sum / num_batches,
        avg_comps,
    )


@torch.no_grad()
def val_epoch_end_to_end(
    extractor_model: torch.nn.Module,
    extractor_backend: ExtractorBackend,
    svt_model: SVTModel,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    beta: float = 1.0,
    pos_weight: float = 20.0,
    free_bits: float = 0.0,
    free_bits_meter: float | None = None,
    free_bits_phase: float | None = None,
    free_bits_tempo: float | None = None,
    extractor_loss_weight: float = 1.0,
    svt_loss_weight: float = 1.0,
    fps: float = 86.1328125,
    smooth_sigma: float = 3.0,
    smooth_sigma_db: float = 5.0,
) -> tuple[float, float, float, dict[str, float], dict[str, float]]:
    """Run one validation epoch (no gradient updates).

    Returns:
        (avg_total, avg_extractor, avg_svt, avg_loss_components, avg_mir_eval_metrics).
    """
    extractor_model.eval()
    svt_model.eval()

    total_sum = 0.0
    extractor_sum = 0.0
    svt_sum = 0.0
    comp_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    num_batches = 0
    num_eval_samples = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        extractor_target = batch["extractor_target"].to(device)
        beat_targets = batch["beat_targets"].to(device)

        extractor_loss, activations = extractor_backend.compute_loss_and_activations(
            model=extractor_model, audio=audio, target=extractor_target,
        )

        # The extractor center-crops its output. First, center-crop all targets
        # to match the extractor's full output length, THEN cap for sequential loop.
        T_ext = activations.shape[1]  # extractor output length (center-valid)
        beat_targets_aligned = _center_crop_seq_dim(beat_targets.unsqueeze(-1), T_ext).squeeze(-1)
        phase_prev_aligned = _center_crop_seq_dim(batch["phase_prev"].to(device), T_ext)
        log_tempo_prev_aligned = _center_crop_seq_dim(batch["log_tempo_prev"].to(device), T_ext)
        meter_prev_aligned = _center_crop_seq_dim(batch["meter_onehot_prev"].to(device), T_ext)

        # Downbeat targets from extractor_target channel 1
        downbeat_targets_aligned = _center_crop_seq_dim(
            extractor_target[:, 1:2, :].transpose(1, 2), T_ext
        ).squeeze(-1)

        # Cap sequence length for sequential Algorithm 1
        _MAX_VAL_FRAMES = 512
        T_act = min(T_ext, _MAX_VAL_FRAMES)
        activations = activations[:, :T_act, :]
        beat_targets_cropped = beat_targets_aligned[:, :T_act]
        downbeat_targets_cropped = downbeat_targets_aligned[:, :T_act]

        # Full GT z_prev for val
        z_prev_gt = {
            "phase": phase_prev_aligned[:, :T_act, :],
            "log_tempo": log_tempo_prev_aligned[:, :T_act, :],
            "meter_onehot": meter_prev_aligned[:, :T_act, :],
        }
        out = svt_model(activations, z_prev_gt, temperature=temperature,
                        beat_targets=beat_targets_cropped,
                        downbeat_targets=downbeat_targets_cropped)
        svt_total, components = compute_elbo_loss(
            beat_logits=out["beat_logits"],
            beat_targets=beat_targets_cropped,
            posterior=out["posterior"],
            prior=out["prior"],
            beta=beta,
            pos_weight=pos_weight,
            free_bits=free_bits,
            free_bits_meter=free_bits_meter,
            free_bits_phase=free_bits_phase,
            free_bits_tempo=free_bits_tempo,
            downbeat_targets=downbeat_targets_cropped,
            smooth_sigma=smooth_sigma,
            smooth_sigma_db=smooth_sigma_db,
        )

        total_loss = extractor_loss_weight * extractor_loss + svt_loss_weight * svt_total

        total_sum += float(total_loss.item())
        extractor_sum += float(extractor_loss.item())
        svt_sum += float(svt_total.item())
        for k, v in components.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
        num_batches += 1

        # --- mir_eval beat/downbeat metrics per sample in batch ---
        beat_probs = torch.sigmoid(out["beat_logits"][:, :, 0]).cpu().numpy()  # [B, T]
        bt_ref_np = beat_targets_cropped.cpu().numpy()  # [B, T]
        phase_np = out["samples"]["phase"].cpu().numpy()  # [B, T]
        B = beat_probs.shape[0]

        for b in range(B):
            ref_beats = frames_to_beat_times(bt_ref_np[b], fps)
            if len(ref_beats) < 2:
                continue

            # --- Decoder-based beat extraction ---
            est_beats = extract_beat_timestamps(beat_probs[b], fps=fps)
            if len(est_beats) > 0:
                beat_scores = evaluate_beats(ref_beats, est_beats)
                for k, v in beat_scores.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v

                # Downbeat metrics
                est_downbeats = extract_downbeat_timestamps(est_beats, phase_np[b], fps=fps)
                ref_downbeats = extract_downbeat_timestamps(
                    ref_beats,
                    _center_crop_seq_dim_1d(
                        batch["phase"][b].squeeze(-1), T_act
                    ).numpy() / 1.0,
                    fps=fps,
                )
                if len(ref_downbeats) >= 2 and len(est_downbeats) >= 2:
                    db_scores = evaluate_downbeats(ref_downbeats, est_downbeats)
                    for k, v in db_scores.items():
                        metric_sums[k] = metric_sums.get(k, 0.0) + v

            # --- Phase-based beat extraction (bar pointer wraps) ---
            est_beats_phase = extract_beats_from_phase_trajectory(phase_np[b], fps=fps)
            if len(est_beats_phase) > 0:
                phase_beat_scores = evaluate_beats(ref_beats, est_beats_phase)
                for k, v in phase_beat_scores.items():
                    metric_sums[f"phase_{k}"] = metric_sums.get(f"phase_{k}", 0.0) + v

            num_eval_samples += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0, {}, {}

    avg_comps = {k: v / num_batches for k, v in comp_sums.items()}
    avg_metrics = {k: v / max(num_eval_samples, 1) for k, v in metric_sums.items()}
    return (
        total_sum / num_batches,
        extractor_sum / num_batches,
        svt_sum / num_batches,
        avg_comps,
        avg_metrics,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CHART SVT model")
    parser.add_argument("--mode", choices=["activation", "end2end"], default="activation")
    parser.add_argument("--extractor", type=str, default="wavebeat", choices=list_extractor_backends())
    parser.add_argument("--activations_dir", type=str, default=None)
    parser.add_argument("--phases_dir", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Structured latent / ELBO
    parser.add_argument("--num_meter_classes", type=int, default=8)
    parser.add_argument("--gumbel_temp_start", type=float, default=1.0)
    parser.add_argument("--gumbel_temp_end", type=float, default=0.1)
    parser.add_argument("--kl_anneal_epochs", type=int, default=10)
    parser.add_argument(
        "--bce_pos_weight",
        type=float,
        default=20.0,
        help="BCE positive-class weight to compensate for beat-frame class imbalance (~1%% positive rate). Default: 20.0",
    )
    parser.add_argument(
        "--free_bits",
        type=float,
        default=0.0,
        help="Free-bits threshold λ (nats) applied per latent per sample. Default for all latents. Default: 0.0 (disabled).",
    )
    parser.add_argument("--free_bits_meter", type=float, default=None,
                        help="Per-latent free_bits override for meter KL. Default: use --free_bits.")
    parser.add_argument("--free_bits_phase", type=float, default=None,
                        help="Per-latent free_bits override for phase KL. Default: use --free_bits.")
    parser.add_argument("--free_bits_tempo", type=float, default=None,
                        help="Per-latent free_bits override for tempo KL. Default: use --free_bits.")
    parser.add_argument("--smooth_sigma", type=float, default=3.0,
                        help="Gaussian kernel sigma for beat target smoothing (frames). Default: 3.0.")
    parser.add_argument("--smooth_sigma_db", type=float, default=5.0,
                        help="Gaussian kernel sigma for downbeat target smoothing (frames). Default: 5.0.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping. Default: 1.0.")
    parser.add_argument("--z_context", type=int, default=1,
                        help="Number of past z frames for posterior context. 1=Markov (paper), >1=extended context.")
    parser.add_argument("--h_prior_bottleneck", type=int, default=0,
                        help="Bottleneck dim for h_prior in decoder. 0=full (default), >0=compressed.")

    # End-to-end
    parser.add_argument("--extractor_ckpt", type=str, default=None)
    parser.add_argument("--freeze_extractor", action="store_true")
    parser.add_argument("--extractor_loss_weight", type=float, default=1.0)
    parser.add_argument("--svt_loss_weight", type=float, default=1.0)

    # Backward compat
    parser.add_argument("--wavebeat_ckpt", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--freeze_wavebeat", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wavebeat_loss_weight", type=float, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--save_ckpt_path", type=str, default=None)

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="chart")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    known_args, _ = parser.parse_known_args()
    extractor_backend = get_extractor_backend(known_args.extractor)
    extractor_backend.add_cli_args(parser)
    return parser


def _normalize_backward_compat_args(args: argparse.Namespace) -> None:
    if args.wavebeat_ckpt is not None and args.extractor_ckpt is None:
        args.extractor_ckpt = args.wavebeat_ckpt
    if bool(args.freeze_wavebeat):
        args.freeze_extractor = True
    if args.wavebeat_loss_weight is not None:
        args.extractor_loss_weight = args.wavebeat_loss_weight


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _normalize_backward_compat_args(args)

    # --- Distributed setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank >= 0
    if is_distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = _select_device()
        rank = 0
        world_size = 1
    is_main = rank == 0
    args.dist_rank = rank
    args.dist_world_size = world_size

    if is_main:
        print(f"Using device: {device}" + (f" (DDP world_size={world_size})" if is_distributed else ""))

    # --- Weights & Biases init ---
    use_wandb = is_main and _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        _wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            resume="allow",
        )
        # Define custom x-axes so step-level and epoch-level metrics don't conflict.
        _wandb.define_metric("global_step")
        _wandb.define_metric("train_step/*", step_metric="global_step")
        _wandb.define_metric("epoch")
        _wandb.define_metric("train/*", step_metric="epoch")
        _wandb.define_metric("val/*", step_metric="epoch")
        _wandb.define_metric("ckpt/*", step_metric="epoch")
    else:
        if not _WANDB_AVAILABLE:
            print("[wandb] not installed, skipping.")
        elif args.no_wandb:
            print("[wandb] disabled via --no_wandb.")

    K = args.num_meter_classes

    if args.mode == "activation":
        if args.activations_dir is None or args.phases_dir is None:
            raise ValueError("--activations_dir and --phases_dir are required for mode=activation")

        dataset = ActivationDataset(
            activations_dir=args.activations_dir,
            phases_dir=args.phases_dir,
            seq_len=args.seq_len,
            num_meter_classes=K,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = SVTModel(
            hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=K,
            z_context=args.z_context, h_prior_bottleneck=args.h_prior_bottleneck,
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        for epoch in range(1, args.num_epochs + 1):
            temp = _gumbel_temperature(
                epoch - 1, args.num_epochs, args.gumbel_temp_start, args.gumbel_temp_end,
            )
            beta = _kl_beta(epoch - 1, args.kl_anneal_epochs)

            avg_total, avg_comps = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                temperature=temp,
                beta=beta,
                pos_weight=args.bce_pos_weight,
                max_grad_norm=args.max_grad_norm,
                epoch=epoch,
                num_epochs=args.num_epochs,
                log_interval=args.log_interval,
            )

            comp_str = " | ".join(f"{k}={v:.6f}" for k, v in avg_comps.items())
            print(
                f"[Epoch {epoch:03d}/{args.num_epochs:03d}] "
                f"total={avg_total:.6f} | {comp_str} | tau={temp:.3f} beta={beta:.3f}"
            )
            if use_wandb:
                log = {"train/total_loss": avg_total, "train/gumbel_temp": temp, "train/kl_beta": beta}
                log.update({f"train/{k}": v for k, v in avg_comps.items()})
                _wandb.log(log, step=epoch)

        if args.save_ckpt_path:
            torch.save({"svt_model": model.state_dict(), "args": vars(args)}, args.save_ckpt_path)
        if use_wandb:
            _wandb.finish()
        return

    # --- End-to-end mode ---
    extractor_backend = get_extractor_backend(args.extractor)
    dataloader = extractor_backend.build_dataloader(args)
    val_dataloader = extractor_backend.build_val_dataloader(args)
    extractor_model = extractor_backend.build_model(args, device)
    extractor_backend.load_checkpoint(extractor_model, args, device)

    if args.freeze_extractor:
        for parameter in extractor_model.parameters():
            parameter.requires_grad = False

    svt_model = SVTModel(
        hidden_dim=128, nhead=4, num_layers=2, num_meter_classes=K,
    ).to(device)

    if is_distributed:
        svt_model = DDP(svt_model, device_ids=[local_rank], find_unused_parameters=False)

    trainable_parameters = [
        p for p in list(extractor_model.parameters()) + list(svt_model.parameters())
        if p.requires_grad
    ]
    optimizer = optim.AdamW(trainable_parameters, lr=args.lr)

    # Top-3 checkpoint tracking: heap of (score, epoch, path)
    # We use a min-heap so the worst of the top-3 is always heap[0].
    top_ckpts: list[tuple[float, int, str]] = []
    ckpt_dir = os.path.dirname(args.save_ckpt_path) if args.save_ckpt_path else "checkpoints"
    ckpt_stem = os.path.splitext(os.path.basename(args.save_ckpt_path))[0] if args.save_ckpt_path else "chart"
    os.makedirs(ckpt_dir or ".", exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        # Keep DistributedSampler in sync with epoch for proper shuffling
        if is_distributed and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        temp = _gumbel_temperature(
            epoch - 1, args.num_epochs, args.gumbel_temp_start, args.gumbel_temp_end,
        )
        beta = _kl_beta(epoch - 1, args.kl_anneal_epochs)

        avg_total, avg_ext, avg_svt, avg_comps = train_epoch_end_to_end(
            extractor_model=extractor_model,
            extractor_backend=extractor_backend,
            svt_model=svt_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            temperature=temp,
            beta=beta,
            pos_weight=args.bce_pos_weight,
            free_bits=args.free_bits,
            free_bits_meter=args.free_bits_meter,
            free_bits_phase=args.free_bits_phase,
            free_bits_tempo=args.free_bits_tempo,
            max_grad_norm=args.max_grad_norm,
            extractor_loss_weight=args.extractor_loss_weight,
            svt_loss_weight=args.svt_loss_weight,
            epoch=epoch,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
            is_main=is_main,
            smooth_sigma=args.smooth_sigma,
            smooth_sigma_db=args.smooth_sigma_db,
        )

        if is_main:
            comp_str = " | ".join(f"{k}={v:.6f}" for k, v in avg_comps.items())
            print(
                f"[Epoch {epoch:03d}/{args.num_epochs:03d}] "
                f"total={avg_total:.6f} | ext={avg_ext:.6f} | svt={avg_svt:.6f} | "
                f"{comp_str} | tau={temp:.3f} beta={beta:.3f}"
            )

            # Gradient norm (computed over all trainable params after last backward)
            grad_norm = 0.0
            for p in trainable_parameters:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            lr_current = optimizer.param_groups[0]["lr"]
            if use_wandb:
                train_log = {
                    "epoch": epoch,
                    "train/total_loss": avg_total,
                    "train/ext_loss": avg_ext,
                    "train/svt_loss": avg_svt,
                    "train/grad_norm": grad_norm,
                    "train/gumbel_temp": temp,
                    "train/kl_beta": beta,
                    "train/lr": lr_current,
                }
                train_log.update({f"train/{k}": v for k, v in avg_comps.items()})
                _wandb.log(train_log)

        val_f_measure = 0.0
        if is_main and val_dataloader is not None:
            val_fps = getattr(args, "audio_sample_rate", 22050) / getattr(args, "target_factor", 256)
            v_total, v_ext, v_svt, v_comps, v_metrics = val_epoch_end_to_end(
                extractor_model=extractor_model,
                extractor_backend=extractor_backend,
                svt_model=svt_model,
                dataloader=val_dataloader,
                device=device,
                temperature=temp,
                beta=beta,
                pos_weight=args.bce_pos_weight,
                free_bits=args.free_bits,
                free_bits_meter=args.free_bits_meter,
                free_bits_phase=args.free_bits_phase,
                free_bits_tempo=args.free_bits_tempo,
                extractor_loss_weight=args.extractor_loss_weight,
                svt_loss_weight=args.svt_loss_weight,
                fps=val_fps,
                smooth_sigma=args.smooth_sigma,
            smooth_sigma_db=args.smooth_sigma_db,
            )
            v_comp_str = " | ".join(f"{k}={v:.6f}" for k, v in v_comps.items())
            print(
                f"  [Val] total={v_total:.6f} | ext={v_ext:.6f} | svt={v_svt:.6f} | "
                f"{v_comp_str}"
            )
            val_f_measure = v_metrics.get("F-measure", 0.0)
            m_str = " | ".join(f"{k}={v:.4f}" for k, v in v_metrics.items()) if v_metrics else "F-measure=0.0000"
            print(f"  [Val mir_eval] {m_str}")

            # Visualize beat activations on first val batch
            try:
                viz_batch = next(iter(val_dataloader))
                with torch.no_grad():
                    viz_audio = viz_batch["audio"].to(device)
                    viz_ext_target = viz_batch["extractor_target"].to(device)
                    _, viz_act = extractor_backend.compute_loss_and_activations(
                        model=extractor_model, audio=viz_audio, target=viz_ext_target,
                    )
                    viz_T = min(viz_act.shape[1], 512)
                    viz_act = viz_act[:, :viz_T, :]
                    viz_bt = _center_crop_seq_dim(
                        viz_batch["beat_targets"].to(device).unsqueeze(-1), viz_T
                    ).squeeze(-1)
                    viz_z_gt = {
                        "phase": _center_crop_seq_dim(viz_batch["phase_prev"].to(device), viz_T),
                        "log_tempo": _center_crop_seq_dim(viz_batch["log_tempo_prev"].to(device), viz_T),
                        "meter_onehot": _center_crop_seq_dim(viz_batch["meter_onehot_prev"].to(device), viz_T),
                    }
                    viz_db = _center_crop_seq_dim(
                        viz_ext_target[:, 1:2, :].transpose(1, 2), viz_T
                    ).squeeze(-1)
                    viz_out = svt_model(viz_act, viz_z_gt, temperature=temp,
                                        beat_targets=viz_bt, downbeat_targets=viz_db)
                    # Get GT downbeats from extractor_target channel 1
                    viz_gt_db = None
                    if "extractor_target" in viz_batch:
                        ext_tgt = viz_batch["extractor_target"].to(device)
                        if ext_tgt.shape[1] >= 2:
                            viz_gt_db = _center_crop_seq_dim(
                                ext_tgt[:, 1:2, :].permute(0, 2, 1), viz_T
                            )[0, :, 0].cpu().numpy()
                viz_dir = os.path.join(ckpt_dir, "viz")
                viz_path = _save_beat_viz(viz_out, viz_bt, epoch, viz_dir, fps=val_fps,
                                          gt_phase=viz_gt_db, gt_z_prev=viz_z_gt,
                                          downbeat_targets=viz_db)
                if viz_path:
                    print(f"  [Viz] saved {viz_path}")
                    if use_wandb:
                        _wandb.log({"epoch": epoch, "val/beat_viz": _wandb.Image(viz_path)})
            except Exception as e:
                print(f"  [Viz] failed: {e}")

            if use_wandb:
                val_log = {
                    "epoch": epoch,
                    "val/total_loss": v_total,
                    "val/ext_loss": v_ext,
                    "val/svt_loss": v_svt,
                }
                val_log.update({f"val/{k}": v for k, v in v_comps.items()})
                val_log.update({f"val/{k}": v for k, v in v_metrics.items()})
                _wandb.log(val_log)

        # --- Save top-3 checkpoints by val beat F-measure (rank 0 only) ---
        if is_main and args.save_ckpt_path:
            svt_state = svt_model.module.state_dict() if is_distributed else svt_model.state_dict()
            ckpt_path = os.path.join(ckpt_dir, f"{ckpt_stem}_ep{epoch:03d}_f{val_f_measure:.4f}.pt")
            ckpt_data = {
                "epoch": epoch,
                "val_f_measure": val_f_measure,
                "extractor": args.extractor,
                "extractor_model": extractor_model.state_dict(),
                "svt_model": svt_state,
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            if len(top_ckpts) < 3:
                torch.save(ckpt_data, ckpt_path)
                heapq.heappush(top_ckpts, (val_f_measure, epoch, ckpt_path))
                print(f"  [Ckpt] saved {os.path.basename(ckpt_path)}")
                if use_wandb:
                    _wandb.log({"epoch": epoch, "ckpt/saved_epoch": epoch, "ckpt/val_f_measure": val_f_measure})
            elif val_f_measure > top_ckpts[0][0]:
                # Better than the worst of top-3: evict it
                _, _, old_path = heapq.heapreplace(top_ckpts, (val_f_measure, epoch, ckpt_path))
                torch.save(ckpt_data, ckpt_path)
                if os.path.exists(old_path):
                    os.remove(old_path)
                print(f"  [Ckpt] saved {os.path.basename(ckpt_path)} (replaced {os.path.basename(old_path)})")
                if use_wandb:
                    _wandb.log({"epoch": epoch, "ckpt/saved_epoch": epoch, "ckpt/val_f_measure": val_f_measure})

    if use_wandb:
        _wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
