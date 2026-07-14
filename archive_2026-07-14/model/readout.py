"""Deployment read-outs + mir_eval scoring: geometric (phase-wrap -> event times) through the
prior/recognition rollouts, and the particle-filter read-out (MAP wraps + Bayesian activations).

Every function takes ``fps`` explicitly (from the active frontend config) -- no hardcoded frame
rate. Beats-per-bar always comes from the model's meter latent (per-song majority vote, or the MAP
particle's majority for the filter); a hardcoded value appears only as the labelled
``fallback_beats_per_bar`` on records that predate the meter read-out.
"""
import math

import numpy as np
import torch
import mir_eval
from scipy.signal import find_peaks

TWO_PI = 2.0 * math.pi
DEFAULT_EVAL_MAX_FRAMES = 1600


def phase_wrap_times(phase_trajectory, min_separation_seconds, fps):
    # Frames where an advancing phase wraps 2*pi -> 0 (a step more negative than -pi), de-duplicated.
    phase_steps = np.diff(phase_trajectory)
    wrap_frames = np.where(phase_steps < -math.pi)[0] + 1
    kept_frames = []
    min_separation_frames = int(min_separation_seconds * fps)
    for frame in wrap_frames:
        if not kept_frames or frame - kept_frames[-1] >= min_separation_frames:
            kept_frames.append(frame)
    return np.array(kept_frames) / fps


def beat_f_measure(reference_times, estimated_times, tolerance_seconds=0.07):
    if len(reference_times) == 0:
        return float("nan")
    if len(estimated_times) == 0:
        return 0.0
    return float(mir_eval.beat.f_measure(np.asarray(reference_times, dtype=float),
                                         np.asarray(estimated_times, dtype=float),
                                         f_measure_threshold=tolerance_seconds))


def peak_pick_times(activation, min_separation_seconds, fps, height=0.1):
    frames, _ = find_peaks(activation, height=height,
                           distance=max(1, int(min_separation_seconds * fps)))
    return frames / fps


def song_beats_per_bar(rollout):
    # Per-song beats-per-bar from the SOFT METER LATENT (majority vote of per-frame argmax) --
    # the meter-consequential read-out (never a hardcoded 4; class k = k+1 beats/bar).
    per_frame = rollout.meter_probabilities[0].argmax(-1).cpu().numpy()
    return int(np.bincount(per_frame).argmax()) + 1


def _reference_times(song, num_frames, fps):
    beats = np.where(song.beat_targets[:num_frames].numpy() > 0.5)[0] / fps
    downbeats = np.where(song.downbeat_targets[:num_frames].numpy() > 0.5)[0] / fps
    return beats, downbeats


def _score_phase_trajectory(phase_trajectory, beats_per_bar, reference_beats, reference_downbeats,
                            fps, beat_scores, downbeat_scores, coverages, rotation_ratios):
    beat_phase_trajectory = (beats_per_bar * phase_trajectory) % TWO_PI
    if len(reference_beats) >= 2:
        beat_scores.append(beat_f_measure(
            reference_beats, phase_wrap_times(beat_phase_trajectory, 0.10, fps)))
    if len(reference_downbeats) >= 2:
        downbeat_scores.append(beat_f_measure(
            reference_downbeats, phase_wrap_times(phase_trajectory, 0.30, fps)))
    histogram, _ = np.histogram(phase_trajectory, bins=16, range=(0.0, TWO_PI))
    coverages.append(float((histogram > 0).mean()))
    total_revolutions = float(np.abs(np.unwrap(phase_trajectory)[-1]
                                     - np.unwrap(phase_trajectory)[0]) / TWO_PI)
    rotation_ratios.append(total_revolutions / max(len(reference_downbeats) - 1, 1))


@torch.no_grad()
def evaluate_prior_readout(model, songs, fps, max_frames=DEFAULT_EVAL_MAX_FRAMES, device="cuda"):
    """Deployment the CVAE way (Sohn et al. 2015, sec. 4.1): z from the PRIOR network p(z|x) -- a
    deterministic rollout at the prior means -- then the geometric read-out. The event channels y
    are never an input to this pipeline."""
    model.eval()
    beat_scores, downbeat_scores, coverages, rotation_ratios = [], [], [], []
    example_phase_trajectory = None
    for song in songs:
        num_frames = min(song.features.shape[0], song.beat_targets.shape[0], max_frames)
        features = song.features[:num_frames].unsqueeze(0).to(device)
        rollout = model.rollout_prior(features, sample=False)
        phase_trajectory = rollout.bar_phase[0].cpu().numpy()
        if example_phase_trajectory is None:
            example_phase_trajectory = phase_trajectory
        reference_beats, reference_downbeats = _reference_times(song, num_frames, fps)
        _score_phase_trajectory(phase_trajectory, song_beats_per_bar(rollout),
                                reference_beats, reference_downbeats, fps,
                                beat_scores, downbeat_scores, coverages, rotation_ratios)
    model.train()
    return {"beat_f": float(np.nanmean(beat_scores)),
            "downbeat_f": float(np.nanmean(downbeat_scores)),
            "phase_coverage": float(np.mean(coverages)),
            "rotation_ratio": float(np.median(rotation_ratios)),
            "example_phase_trajectory": example_phase_trajectory}


@torch.no_grad()
def evaluate_geometric_readout(model, songs, fps, audio_condition="real",
                               max_frames=DEFAULT_EVAL_MAX_FRAMES, device="cuda"):
    """Recognition-network read-out with SILENT event channels -- an out-of-distribution diagnostic
    (kept for the leak table: real vs shuffle vs zero audio), NOT a deployment claim."""
    model.eval()
    beat_scores, downbeat_scores, coverages, rotation_ratios = [], [], [], []
    example_phase_trajectory = None
    for song_index, song in enumerate(songs):
        source = songs[(song_index + 1) % len(songs)] if audio_condition == "shuffle" else song
        num_frames = min(source.features.shape[0], song.beat_targets.shape[0], max_frames)
        if audio_condition == "zero":
            features = torch.zeros(1, num_frames, source.features.shape[-1], device=device)
        else:
            features = source.features[:num_frames].unsqueeze(0).to(device)
        silent_channel = torch.zeros(1, num_frames, device=device)
        rollout = model.rollout(features, silent_channel, silent_channel, sample=False, compute_kl=False)
        phase_trajectory = rollout.bar_phase[0].cpu().numpy()
        if example_phase_trajectory is None:
            example_phase_trajectory = phase_trajectory
        reference_beats, reference_downbeats = _reference_times(song, num_frames, fps)
        _score_phase_trajectory(phase_trajectory, song_beats_per_bar(rollout),
                                reference_beats, reference_downbeats, fps,
                                beat_scores, downbeat_scores, coverages, rotation_ratios)
    model.train()
    return {"beat_f": float(np.nanmean(beat_scores)),
            "downbeat_f": float(np.nanmean(downbeat_scores)),
            "phase_coverage": float(np.mean(coverages)),
            "rotation_ratio": float(np.median(rotation_ratios)),
            "example_phase_trajectory": example_phase_trajectory}


@torch.no_grad()
def evaluate_filter_readout(model, songs, fps, max_frames=DEFAULT_EVAL_MAX_FRAMES,
                            device="cuda", **filter_kwargs):
    """Particle-filter deployment (model/particle_filter.py). Two read-outs: wraps of the MAP
    particle's phase, and peak-picked Bayesian wrap activations (the ensemble read-out -- beats
    MAP by ~+0.13 on the same runs)."""
    model.eval()
    map_beat, map_downbeat, bayes_beat, bayes_downbeat, rotation_ratios = [], [], [], [], []
    for song in songs:
        num_frames = min(song.features.shape[0], song.beat_targets.shape[0], max_frames)
        features = song.features[:num_frames].unsqueeze(0).to(device)
        observations = song.frontend_activations[:num_frames].to(device)
        result = model.filter_deploy(features, observations, **filter_kwargs)
        map_phase = result["map_phase"]
        map_beats_per_bar = result["map_beats_per_bar"]
        reference_beats, reference_downbeats = _reference_times(song, num_frames, fps)
        beat_phase_trajectory = (map_beats_per_bar * map_phase) % TWO_PI
        if len(reference_beats) >= 2:
            map_beat.append(beat_f_measure(
                reference_beats, phase_wrap_times(beat_phase_trajectory, 0.10, fps)))
            bayes_beat.append(beat_f_measure(
                reference_beats, peak_pick_times(result["beat_activation"], 0.10, fps)))
        if len(reference_downbeats) >= 2:
            map_downbeat.append(beat_f_measure(
                reference_downbeats, phase_wrap_times(map_phase, 0.30, fps)))
            bayes_downbeat.append(beat_f_measure(
                reference_downbeats, peak_pick_times(result["downbeat_activation"], 0.30, fps)))
        total_revolutions = float(np.abs(np.unwrap(map_phase)[-1] - np.unwrap(map_phase)[0]) / TWO_PI)
        rotation_ratios.append(total_revolutions / max(len(reference_downbeats) - 1, 1))
    model.train()
    return {"beat_f": float(np.nanmean(map_beat)),
            "downbeat_f": float(np.nanmean(map_downbeat)),
            "beat_f_bayes": float(np.nanmean(bayes_beat)),
            "downbeat_f_bayes": float(np.nanmean(bayes_downbeat)),
            "rotation_ratio": float(np.median(rotation_ratios))}
