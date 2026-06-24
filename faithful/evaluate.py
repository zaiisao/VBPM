"""Deploy-path evaluation of the faithful model, scored with mir_eval.

The point of this evaluation is the COLLAPSE DIAGNOSTIC, not a leaderboard number. We
read beats off the free-running prior two ways and compare:

  * ``phase_wrap`` : beats = 2*pi->0 wraps of the deterministic phase-mean chain. This is
    the PURE LATENT read-out -- if the bar-pointer dynamics learned anything, it works.
  * ``decoder``    : beats = peak-picks of the Bernoulli decoder, which reads h (§5.4). This
    rides the audio and can look fine even when the latent is dead.
  * ``metronome``  : a constant-120-BPM floor.

The signature of posterior collapse is: decoder >> phase_wrap ~ metronome, while the
training KL terms decay to ~0. mir_eval's standard 70 ms F-measure window is used.
"""
from __future__ import annotations

import math

import mir_eval
import numpy as np
import torch

from .elbo import free_run


def _min_dist_filter(frames: np.ndarray, min_gap: float) -> np.ndarray:
    out, last = [], -1e9
    for f in frames:
        if f - last >= min_gap:
            out.append(f); last = f
    return np.asarray(out, dtype=float)


def beats_from_phase(phase: np.ndarray, fps: float, min_dist_sec: float = 0.15) -> np.ndarray:
    """Phase-wrap read-out: a beat wherever the phase mean drops 2*pi -> 0."""
    d = np.diff(np.asarray(phase))
    wraps = np.where(d < -math.pi)[0] + 1
    return _min_dist_filter(wraps, min_dist_sec * fps) / fps


def beats_from_activation(prob: np.ndarray, fps: float, thr: float = 0.5,
                          min_dist_sec: float = 0.15) -> np.ndarray:
    """Decoder read-out: local maxima of the beat probability above a threshold."""
    prob = np.asarray(prob)
    peaks = [t for t in range(1, len(prob) - 1)
             if prob[t] >= thr and prob[t] >= prob[t - 1] and prob[t] >= prob[t + 1]]
    return _min_dist_filter(np.asarray(peaks, dtype=float), min_dist_sec * fps) / fps


def metronome(n_frames: int, fps: float, bpm: float = 120.0) -> np.ndarray:
    return np.arange(0.0, n_frames / fps, 60.0 / bpm)


# --- OFFICIAL bar-pointer read-out (paper §5.2): phi is the BAR phase, [0,2pi) per bar. ---
# A 2pi wrap is a DOWNBEAT (bar boundary). Beats are the m equal subdivisions of the bar:
# phi crosses 2*pi*k/m, k=0..m-1. Equivalently beats = wraps of beat-phase psi = (m*phi) mod 2pi.
# m = meter (beats per bar). Use the deterministic phase_mu chain (clean, monotone).
_TWO_PI = 2.0 * math.pi


def downbeats_from_barphase(phase: np.ndarray, fps: float, min_dist_sec: float = 0.30) -> np.ndarray:
    """Downbeats = bar-boundary wraps (2pi -> 0) of the bar phase phi."""
    return beats_from_phase(phase, fps, min_dist_sec)


def beats_from_barphase(phase: np.ndarray, m: int, fps: float, min_dist_sec: float = 0.10) -> np.ndarray:
    """Beats = the m subdivisions of the bar (phi crossing 2*pi*k/m), via wraps of (m*phi)."""
    psi = (int(m) * np.asarray(phase, dtype=float)) % _TWO_PI
    w = np.where(np.diff(psi) < -math.pi)[0] + 1
    return _min_dist_filter(w.astype(float), min_dist_sec * fps) / fps


def bpm_from_logtempo(log_tempo: float, m: int, fps: float) -> float:
    """phi-dot is the BAR advance rate (rad/frame); beat-BPM = 60*fps*m*exp(log_tempo)/(2pi)."""
    return 60.0 * fps * int(m) * math.exp(float(log_tempo)) / _TWO_PI


def f_measure(ref_sec: np.ndarray, est_sec: np.ndarray) -> float:
    ref = np.asarray(ref_sec, dtype=float)
    est = np.asarray(est_sec, dtype=float)
    if len(ref) == 0:
        return float("nan")
    if len(est) == 0:
        return 0.0
    return float(mir_eval.beat.f_measure(ref, est))


@torch.no_grad()
def evaluate(model, logmel, songs, device, fps: float = 86.1328125,
             max_frames: int = 4000, temperature: float = 0.3):
    """``songs`` is an iterable of (key, audio[N], beats[T], downbeats[T], meta)."""
    was_training = model.training
    model.eval()
    accum = {"phase_wrap": [], "decoder": [], "metronome": []}
    per_song = []
    for key, audio, beats, _downbeats, _meta in songs:
        h = logmel(audio.to(device).unsqueeze(0))           # [1, T, n_mels]
        T = min(h.shape[1], max_frames)
        out = free_run(model, h[:, :T], temperature=temperature)
        phase_mu = out["phase_mu"][0, :T].cpu().numpy()
        dec = out["decoder_prob"][0, :T].cpu().numpy()
        ref = np.where(beats.numpy()[:T] > 0.5)[0] / fps
        row = {
            "key": key, "n_ref": int(len(ref)),
            "phase_wrap": f_measure(ref, beats_from_phase(phase_mu, fps)),
            "decoder": f_measure(ref, beats_from_activation(dec, fps)),
            "metronome": f_measure(ref, metronome(T, fps)),
        }
        for nm in accum:
            if not math.isnan(row[nm]):
                accum[nm].append(row[nm])
        per_song.append(row)
    if was_training:
        model.train()
    summary = {k: (float(np.mean(v)) if v else float("nan")) for k, v in accum.items()}
    summary["n_songs"] = len(per_song)
    return summary, per_song
