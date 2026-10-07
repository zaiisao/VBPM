"""Peak-based proposals retained for historical diagnostic experiments."""

import math

import torch
from torch.nn import functional as F


def extract_audio_phase(
    proposal, h, mask, max_bpm=math.inf, min_probability=0.3, recover_missing_beats=False
):
    """Estimate bar phase from frontend beat peaks and downbeat scores."""
    if max_bpm <= 0:
        raise ValueError("Proposal maximum BPM must be positive")
    if not 0 <= min_probability < 1:
        raise ValueError("Proposal probability threshold must be in [0,1)")

    logits = F.linear(h, proposal.prediction_weight, proposal.prediction_bias)
    beat_logits = logits[..., 0] + logits[..., 1]
    downbeat_logits = logits[..., 1]

    batch_size, num_frames = mask.shape
    phase = h.new_zeros(batch_size, num_frames)
    velocity = h.new_full(
        (batch_size, num_frames), 2 * math.pi * 2 / (proposal.proposal_meter * proposal.fps)
    )

    bar_confidence = h.new_zeros(batch_size, proposal.proposal_meter)
    # Peak selection is fixed; return the logits separately for diagnostic callers.
    with torch.no_grad():
        probability = beat_logits.sigmoid()
        maxima = F.max_pool1d(probability[:, None], 7, stride=1, padding=3)[:, 0]
        frames = torch.arange(num_frames, device=h.device, dtype=h.dtype)
        for clip_index in range(batch_size):
            confident = probability[clip_index] > min_probability
            peaks = torch.nonzero(
                (probability[clip_index] >= maxima[clip_index]) & confident & (mask[clip_index] > 0)
            ).flatten()

            if peaks.numel() < 2:
                phase[clip_index] = frames * velocity[clip_index, 0]
                continue

            positions = peaks.to(h.dtype)
            # Refine the baseline peak using its logit curvature, without
            # looking at beat annotations or using true tempo.
            inside = (peaks > 0) & (peaks < num_frames - 1)
            peak_frames = peaks[inside]
            left, center, right = (
                beat_logits[clip_index, peak_frames - 1],
                beat_logits[clip_index, peak_frames],
                beat_logits[clip_index, peak_frames + 1],
            )
            curvature = left - 2 * center + right
            shift = torch.where(
                curvature.abs() > 1e-6,
                0.5 * (left - right) / curvature,
                torch.zeros_like(curvature),
            )
            positions[inside] += shift.clamp(-0.5, 0.5)
            if math.isfinite(max_bpm):
                # Keep the strongest peaks that satisfy the minimum beat spacing.
                selected = []
                for candidate in (
                    probability[clip_index, peaks].argsort(descending=True, stable=True).tolist()
                ):
                    if all(
                        abs(float(positions[candidate] - positions[j]))
                        >= 60 * proposal.fps / max_bpm
                        for j in selected
                    ):
                        selected.append(candidate)
                selected = sorted(selected)
                peaks = peaks[selected]
                positions = positions[selected]
                if len(selected) < 2:
                    phase[clip_index] = frames * velocity[clip_index, 0]
                    continue

            jumps = torch.ones(len(peaks) - 1, device=h.device, dtype=torch.long)
            if recover_missing_beats:
                # An isolated near-integer long gap, bounded by consistent
                # periods, can be a missed audio peak. Preserve ordinal
                # count rather than slowing the entire clock in that gap.
                gaps = positions[1:] - positions[:-1]
                for j in range(1, len(gaps) - 1):
                    neighbors = torch.cat(
                        (gaps[max(0, j - 2) : j], gaps[j + 1 : min(len(gaps), j + 3)])
                    )
                    period = neighbors.median()
                    ratio = gaps[j] / period.clamp_min(1.0)
                    count = int(ratio.round())
                    adjacent_consistent = bool(
                        ((gaps[j - 1 : j + 2 : 2] / period - 1).abs() <= 0.25).all()
                    )
                    if (
                        2 <= count <= 3
                        and abs(float(ratio / count - 1)) <= 0.15
                        and adjacent_consistent
                    ):
                        jumps[j] = count

            beat_numbers = torch.cat((jumps.new_zeros(1), jumps.cumsum(0)))
            scores = torch.stack(
                [
                    downbeat_logits[clip_index, peaks][
                        beat_numbers % proposal.proposal_meter == bar_position
                    ].sum()
                    for bar_position in range(proposal.proposal_meter)
                ]
            )
            bar_confidence[clip_index] = scores.softmax(0)
            downbeat_position = scores.argmax()
            interval_indices = (
                torch.searchsorted(positions.contiguous(), frames.contiguous()) - 1
            ).clamp(0, len(peaks) - 2)
            duration = (positions[interval_indices + 1] - positions[interval_indices]).clamp(
                min=1.0
            )
            beat_position = beat_numbers[interval_indices].to(h.dtype) + (
                frames - positions[interval_indices]
            ) / duration * jumps[interval_indices].to(h.dtype)
            phase[clip_index] = (beat_position - downbeat_position.to(h.dtype)) * (
                2 * math.pi / proposal.proposal_meter
            )
            velocity[clip_index] = (
                (2 * math.pi / proposal.proposal_meter)
                / duration
                * jumps[interval_indices].to(h.dtype)
            )
    return {
        "phase": phase,
        "physical_velocity": velocity,
        "beat_logits": beat_logits,
        "downbeat_logits": downbeat_logits,
        "bar_confidence": bar_confidence,
    }
