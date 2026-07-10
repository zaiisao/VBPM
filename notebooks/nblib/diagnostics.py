"""Against-the-data diagnostics: do the assumed prior families fit real annotations, and which
solution does the pure objective itself prefer?

- ``plot_tempo_increment_fit``  : real beat-to-beat log-tempo increments vs the assumed Gaussian (§1.4).
- ``plot_phase_residual_fit``   : real beat-microtiming residuals vs the assumed von Mises (§1.4).
- ``plot_meter_content_fit``    : real per-song meter + measured bar-to-bar transitions vs the assumed switching (§1.4).
- ``pure_negative_elbo_accounting`` : per-term -ELBO bookkeeping for two trained models (§13 probe).

The three fit graphs need only beat/downbeat annotations (no model); they read the cached ``.pt`` records directly.
"""
import os
import glob

import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt

from .setup import set_all_seeds
from .data import sample_training_crops

_VAL_FEATURE_DIRS = ("cache/acts/bt_val_rich", "../cache/acts/bt_val_rich")

TWO_PI = 2.0 * np.pi

def plot_tempo_increment_fit(big_jump=0.2, min_beats=5, feature_dir=None):
    """Does the assumed Gaussian tempo random walk fit the real beat-to-beat log-tempo increments?
    Delta = diff(-log(IBI)) is the fluctuation the walk must explain; beats only, no model.

    Arguments the notebook owns and feeds in (§1.4), rather than leaving them buried here:
      big_jump    : |Delta| threshold (in LOG-tempo) counted as a "real tempo jump" for the tail statistic.
                    Log-space makes it relative, so 0.2 corresponds to an exp(0.2)-1 = 22% beat-to-beat
                    tempo change; the §1.4 reading quotes this same number, so the notebook sets it once.
      min_beats   : minimum beats a song needs before it contributes increments.
      feature_dir : cached-annotation directory (default: auto-detect the val cache).

    Returns the fitted statistics so the surrounding prose can cite them instead of hard-coding numbers.
    """
    val_feature_dir = feature_dir or next(p for p in _VAL_FEATURE_DIRS if os.path.isdir(p))
    tempo_increments = []
    for cached_path in sorted(glob.glob(f"{val_feature_dir}/*.pt")):
        beat_targets = torch.load(cached_path, map_location="cpu")["beat_targets"].numpy()
        beat_frames = np.where(beat_targets > 0.5)[0].astype(float)
        if len(beat_frames) >= min_beats:
            tempo_increments.append(np.diff(-np.log(np.diff(beat_frames))))
    tempo_increments = np.concatenate(tempo_increments)
    tempo_increments = tempo_increments[np.isfinite(tempo_increments)]

    gaussian_fit = stats.norm.fit(tempo_increments)       # the light-tailed shape the lineage assumes
    laplace_fit = stats.laplace.fit(tempo_increments)     # the adopted heavier-tailed law (madmom's exp.)
    empirical_rate = np.mean(np.abs(tempo_increments) > big_jump)
    gaussian_rate = 2 * stats.norm.sf(big_jump, *gaussian_fit)
    jump_percent = (np.exp(big_jump) - 1.0) * 100.0
    excess_kurtosis = float(stats.kurtosis(tempo_increments))
    tail_ratio = float(empirical_rate / gaussian_rate)
    print(f"{len(tempo_increments)} real beat-to-beat increments   excess kurtosis "
          f"{excess_kurtosis:.1f}  (Gaussian = 0)")
    print(f"P(|Delta| > {big_jump:g}) = {empirical_rate:.3%} in the data, {gaussian_rate:.3%} under the fitted "
          f"Gaussian  ->  ~{tail_ratio:.0f}x too thin  (|Delta| > {big_jump:g} == a {jump_percent:.0f}% tempo jump)")

    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    bound = np.percentile(np.abs(tempo_increments), 99.9); low, high = -bound, bound
    axis.hist(tempo_increments, bins=140, range=(low, high), density=True, alpha=0.5, color="0.6",
              label=r"real $\Delta$log-tempo (val set)")
    grid = np.linspace(low, high, 500)
    axis.plot(grid, stats.norm.pdf(grid, *gaussian_fit), "--", color="tab:red", lw=1.8, label="Gaussian (lineage assumption)")
    axis.plot(grid, stats.laplace.pdf(grid, *laplace_fit), "-", color="tab:green", lw=1.8,
              label="Laplace (adopted; heavier tails)")
    axis.axvline(big_jump, color="0.4", ls=":", lw=1.0)                   # the fed-in jump threshold
    axis.axvline(-big_jump, color="0.4", ls=":", lw=1.0)
    axis.set_yscale("log")   # log-y exposes the tails, where the two laws part ways
    axis.set_xlabel(r"beat-to-beat log-tempo increment  $\Delta$"); axis.set_ylabel("density (log)")
    axis.set_title("Gaussian (assumed) can't fit real tempo increments:\ntoo fat in the shoulders, too thin in the tails")
    axis.legend(frameon=False); axis.spines[["top", "right"]].set_visible(False); axis.grid(alpha=0.25, lw=0.5)
    plt.tight_layout(); plt.show()
    return {"n": len(tempo_increments), "excess_kurtosis": excess_kurtosis, "big_jump": big_jump,
            "jump_percent": float(jump_percent), "empirical_rate": float(empirical_rate),
            "gaussian_rate": float(gaussian_rate), "tail_ratio": tail_ratio}


def plot_phase_residual_fit():
    """Does the assumed von Mises phase prior fit real beat microtiming? The prior advances the pointer
    at the previous tempo; a beat early/late vs that prediction is the residual rho = 2*pi*(IBI_k/IBI_{k-1}-1)."""
    val_feature_dir = next(p for p in _VAL_FEATURE_DIRS if os.path.isdir(p))
    phase_residuals, tempo_increments_for_corr = [], []
    for cached_path in sorted(glob.glob(f"{val_feature_dir}/*.pt")):
        beat_frames = np.where(torch.load(cached_path, map_location="cpu")["beat_targets"].numpy() > 0.5)[0].astype(float)
        if len(beat_frames) < 6:
            continue
        ibi = np.diff(beat_frames)
        phase_residuals.append(2 * np.pi * (ibi[1:] / ibi[:-1] - 1.0))
        tempo_increments_for_corr.append(-np.diff(np.log(ibi)))          # the Section-1.4 tempo increment
    phase_residuals = np.concatenate(phase_residuals)
    phase_residuals = np.mod(phase_residuals + np.pi, TWO_PI) - np.pi     # wrap to (-pi, pi]
    tempo_increments_for_corr = np.concatenate(tempo_increments_for_corr)

    R = np.abs(np.mean(np.exp(1j * phase_residuals)))                     # mean resultant length
    if R < 0.85:                                                         # Fisher's kappa-from-R approximation
        kappa_hat = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        kappa_hat = 1.0 / (R ** 3 - 4 * R ** 2 + 3 * R)
    far_tail = 1.0
    empirical_rate = np.mean(np.abs(phase_residuals) > far_tail)
    vonmises_rate = 2 * stats.vonmises.sf(far_tail, kappa_hat)
    tempo_correlation = np.corrcoef(phase_residuals, -TWO_PI * tempo_increments_for_corr)[0, 1]
    print(f"{len(phase_residuals)} real phase residuals   excess kurtosis "
          f"{stats.kurtosis(phase_residuals):.1f}  (von Mises ~ 0)   fitted kappa {kappa_hat:.0f}")
    print(f"P(|rho| > {far_tail}) = {empirical_rate:.3%} in the data, {vonmises_rate:.3%} under von Mises  ->  "
          f"~{empirical_rate / vonmises_rate:.0f}x too thin in the far tail")
    print(f"corr(phase residual, tempo increment) = {tempo_correlation:.2f}  ->  the SAME beat-timing "
          f"deviation the tempo latent already models")

    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    bound = np.percentile(np.abs(phase_residuals), 99.9); low, high = -bound, bound
    axis.hist(phase_residuals, bins=140, range=(low, high), density=True, alpha=0.5, color="0.6",
              label="real phase residual (val set)")
    grid = np.linspace(low, high, 500)
    axis.plot(grid, stats.vonmises.pdf(grid, kappa_hat), "--", color="tab:red", lw=1.8, label="von Mises (lineage assumption)")
    axis.plot(grid, stats.wrapcauchy.pdf(np.mod(grid, TWO_PI), R), "-", color="tab:green", lw=1.8,
              label="wrapped Cauchy (adopted; heavier tails)")
    axis.set_yscale("log")   # log-y exposes the tails
    axis.set_xlabel("beat-phase residual  (rad)"); axis.set_ylabel("density (log)")
    axis.set_title("von Mises (assumed) can't fit real beat microtiming:\ntoo round at the peak, too thin in the tails")
    axis.legend(frameon=False); axis.spines[["top", "right"]].set_visible(False); axis.grid(alpha=0.25, lw=0.5)
    plt.tight_layout(); plt.show()


def plot_meter_content_fit():
    """What 'meter' actually is in the annotations: a near-constant per-song choice, not the frame-to-frame
    switching a transition matrix suggests. Beats-per-bar = beat frames in [downbeat_k, downbeat_{k+1});
    needs downbeat annotations, no model. Model support is {2,3,4}; the two meters real songs use are shown."""
    val_feature_dir = next(p for p in _VAL_FEATURE_DIRS if os.path.isdir(p))
    meters = [3, 4]                                           # the two meters real songs actually use
    meter_index = {m: i for i, m in enumerate(meters)}
    all_bar_counts, per_song_dominant, single_meter = [], [], []
    transition_counts = np.zeros((len(meters), len(meters)))
    for cached_path in sorted(glob.glob(f"{val_feature_dir}/*.pt")):
        record = torch.load(cached_path, map_location="cpu")
        beat_frames = np.where(record["beat_targets"].numpy() > 0.5)[0]
        downbeat_frames = np.where(record["downbeat_targets"].numpy() > 0.5)[0]
        if len(downbeat_frames) < 3 or len(beat_frames) < 6:
            continue
        bars = [int(np.sum((beat_frames >= a) & (beat_frames < b)))     # beats between consecutive downbeats
                for a, b in zip(downbeat_frames[:-1], downbeat_frames[1:])]
        all_bar_counts.extend(bars)
        bars = [b for b in bars if b in meter_index]         # keep {3,4}; drop rare 2 and edge 1/6/8 (<0.2%)
        if not bars:
            continue
        per_song_dominant.append(max(set(bars), key=bars.count))
        single_meter.append(len(set(bars)) == 1)
        for previous, current in zip(bars[:-1], bars[1:]):
            transition_counts[meter_index[previous], meter_index[current]] += 1

    all_bar_counts = np.array(all_bar_counts)
    per_song_dominant = np.array(per_song_dominant)
    per_song_share = np.array([np.mean(per_song_dominant == m) for m in meters])
    switches = int(transition_counts.sum() - np.trace(transition_counts))
    steps = int(transition_counts.sum())
    row_normalized = transition_counts / np.clip(transition_counts.sum(1, keepdims=True), 1, None)
    dust = 1.0 - np.mean(np.isin(all_bar_counts, meters))
    print(f"{len(all_bar_counts)} real bars: 4->{100*np.mean(all_bar_counts==4):.1f}%  "
          f"3->{100*np.mean(all_bar_counts==3):.1f}%   (model support {{2,3,4}}; 2 and 6/8 together "
          f"{100*dust:.1f}% -- measurement dust)")
    print(f"{len(per_song_dominant)} songs: a fixed M=4 mislabels {100*(1-per_song_share[meter_index[4]]):.0f}% "
          f"of them (the 3/4 minority is real and identifiable)  ->  meter must be a latent")
    print(f"but {100*np.mean(single_meter):.1f}% of songs keep ONE meter throughout; only {switches}/{steps} "
          f"bar-to-bar steps switch ({100*switches/steps:.1f}%)  ->  the empirical transition matrix is ~identity")

    figure, (song_axis, transition_axis) = plt.subplots(1, 2, figsize=(9.6, 3.3),
                                                        gridspec_kw={"width_ratios": [1.05, 1]})
    drawn = song_axis.bar([f"{m}/4" for m in meters], 100 * per_song_share, color="0.6", width=0.6)
    for rectangle, share in zip(drawn, per_song_share):
        song_axis.text(rectangle.get_x() + rectangle.get_width() / 2, 100 * share + 1.5, f"{100*share:.0f}%",
                       ha="center", va="bottom", fontsize=9)
    song_axis.set_ylim(0, 100); song_axis.set_xlabel("the song's meter"); song_axis.set_ylabel("% of songs")
    song_axis.set_title("Meter is a between-song choice\n(fixed $M=4$ is wrong ~1 song in 7)")
    song_axis.spines[["top", "right"]].set_visible(False)

    image = transition_axis.imshow(row_normalized, vmin=0, vmax=1, cmap="viridis")
    for (row, col), value in np.ndenumerate(row_normalized):
        transition_axis.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=10,
                             color="white" if value < 0.6 else "black")
    transition_axis.set_xticks(range(len(meters)), [f"{m}/4" for m in meters])
    transition_axis.set_yticks(range(len(meters)), [f"{m}/4" for m in meters])
    transition_axis.set_xlabel("$m_t$"); transition_axis.set_ylabel("$m_{t-1}$")
    transition_axis.set_title(f"...within a song it barely moves\n(measured: {switches} switches in {steps} bar steps)")
    figure.colorbar(image, ax=transition_axis, shrink=0.85)
    plt.tight_layout(); plt.show()


@torch.no_grad()
def pure_negative_elbo_accounting(model, train_songs, negative_elbo_terms, crop_length_frames,
                                  batch_size, num_batches=12):
    # Mean per-sequence pure -ELBO terms over shared batches (identical crops AND sampling noise across
    # models: the data seed and the per-batch rollout seed are both fixed). negative_elbo_terms is the
    # notebook's inline objective (Section 8), passed in so this stays decoupled from the model code.
    accumulator = {"reconstruction": 0.0, "kl_meter": 0.0, "kl_phase": 0.0, "kl_tempo": 0.0}
    model.eval()
    for batch_index in range(num_batches):
        set_all_seeds(10_000 + batch_index)      # same crops for every model
        features, beat_targets, downbeat_targets = sample_training_crops(
            train_songs, crop_length_frames, batch_size)
        torch.manual_seed(20_000 + batch_index)  # same sampling noise for every model
        rollout = model.rollout(features, beat_targets, downbeat_targets,
                                gumbel_temperature=0.3, sample=True, compute_kl=True)
        _, term_means = negative_elbo_terms(rollout, beat_targets, downbeat_targets)
        for term_name, value in term_means.items():
            accumulator[term_name] += value / num_batches
    model.train()
    accumulator["total"] = sum(accumulator.values())
    return accumulator
