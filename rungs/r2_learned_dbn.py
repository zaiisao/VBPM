"""R2 -- the Böck 2016 DBN with LEARNED factors, trained by the exact forward algorithm.

The ladder rule made literal: R2 changes ONLY how the factors are produced. Deployment is
DBN2016 itself, constructed with the learned transition_lambda (see make_rung()) -- same state
space, same engine, same read-out, so any R1-vs-R2 difference is attributable to the factors.

What is learned here:
  * transition_lambda -- madmom's tempo-change tolerance, the one hand-set scalar of the
    transition model. Differentiable rebuild per step (exponential kernel, log-row-normalized;
    madmom's hard threshold-to-zero is NOT applied during training -- it is non-differentiable --
    and reappears at deployment through DBN2016's standard constructor).
  * (end-to-end) the emission, implicitly: the observation model keeps Böck's parametric form on
    [T, 2] activations, so training the FRONTEND through this loss is what learns the emission.

The objective is the supervised CRF negative log-likelihood of the ANNOTATED bar-pointer path:

    nll = logZ - score(annotated path),   logZ = exact forward through the structured DP

which is differentiable w.r.t. both transition_lambda and the activations (the DP was built for
exactly this -- see rungs/bar_pointer/structured_dp.py, float32 note included).

The annotated path is constructible EXACTLY in the Krebs state space: between consecutive
annotated beat frames f_i -> f_{i+1}, the pointer occupies the tempo block whose interval equals
the actual frame gap (k = f_{i+1} - f_i), advancing +1 per frame through its k states, then takes
the beat-boundary tempo transition. Segments whose gap falls outside the tempo grid, or whose
bar positions are not consecutive mod beats_per_bar, are unrepresentable -> the crop is skipped
(counted by the caller). Crops must start and end on annotated beat frames so path and logZ cover
the same frames.
"""
import numpy as np
import torch
from torch import nn

from rungs.r1_2016_dbn import DBN2016


class R2LearnedFactors(nn.Module):
    """Training-side owner of the learned factors. Deployment = make_rung() -> a plain DBN2016."""

    def __init__(self, fps: float, beats_per_bar=(3, 4), init_transition_lambda: float = 100.0,
                 device: str = "cuda", min_bpm: float = 55.0, max_bpm: float = 215.0,
                 observation_lambda: int = 16):
        super().__init__()
        # R1's chassis, bare and float32 (training regime); predict() is never called on this.
        # observation_lambda MUST match the deployment decode -- training the CRF against a
        # different beat-region width co-adapts the learned factors to the wrong observation
        # world (measured: lambda learned under 16 was decode-optimal under 16, not under 6).
        self.chassis = DBN2016(fps=fps, min_bpm=min_bpm, max_bpm=max_bpm,
                               beats_per_bar=beats_per_bar, num_tempi=None,
                               threshold=0.0, correct=False,
                               observation_lambda=observation_lambda,
                               dtype=torch.float32, device=device)
        self.device = device
        self.log_transition_lambda = nn.Parameter(
            torch.log(torch.tensor(float(init_transition_lambda))))
        self._min_interval = int(self.chassis.state_spaces[0].interval_frames[0])
        self._max_interval = int(self.chassis.state_spaces[0].interval_frames[-1])

    @property
    def transition_lambda(self) -> float:
        return float(self.log_transition_lambda.exp())

    def log_tempo_transition(self) -> torch.Tensor:
        """[V, V] log p(tempo_to | tempo_from), differentiable in transition_lambda.

        madmom's kernel is exp(-lambda * |ratio - 1|); we use |log ratio| in the exponent's
        argument only through ratio itself, so replicate exactly: -lambda * |ratio - 1|, then
        log-row-normalize. (No threshold: training needs gradients through every entry.)
        """
        intervals = torch.from_numpy(
            self.chassis.state_spaces[0].interval_frames.astype(np.float32)).to(self.device)
        ratio = intervals[None, :] / intervals[:, None]
        scores = -self.log_transition_lambda.exp() * (ratio - 1.0).abs()
        return scores - torch.logsumexp(scores, dim=1, keepdim=True)

    def log_class_densities(self, activations: torch.Tensor) -> torch.Tensor:
        """Torch mirror of DBN2016._log_class_densities (clip recipe): [T, 2] probs -> [T, 3]."""
        eps = self.chassis.eps
        beat = activations[:, 0].clamp(eps, 1 - eps)
        downbeat = activations[:, 1].clamp(eps, 1 - eps)
        beat_not_downbeat = (beat - downbeat).clamp(min=eps)
        num_non_beat_states = self.chassis.observation_lambda - 1
        no_beat = (1.0 - beat_not_downbeat - downbeat).clamp(min=1e-12)
        return torch.stack([torch.log(no_beat / num_non_beat_states),
                            torch.log(beat_not_downbeat), torch.log(downbeat)], dim=1)

    def annotated_state_path(self, beat_frames: np.ndarray, beat_in_bar: np.ndarray,
                             beats_per_bar: int):
        """(state_path [T], meter_index) for the span beat_frames[0]..beat_frames[-1], or None
        if unrepresentable (gap outside the tempo grid / non-consecutive bar positions).
        beats_per_bar is the SONG-level meter (a crop may not contain a full bar)."""
        meters = tuple(self.chassis.beats_per_bar)
        bpb = int(beats_per_bar)
        if bpb not in meters:
            return None
        meter_index = meters.index(bpb)
        space = self.chassis.state_spaces[meter_index]
        gaps = np.diff(beat_frames)
        if gaps.min() < self._min_interval or gaps.max() > self._max_interval:
            return None
        if not np.all((beat_in_bar[1:] - beat_in_bar[:-1]) % bpb == 1):
            return None
        path = np.concatenate([
            space.first_states[beat_in_bar[i], gap - self._min_interval] + np.arange(gap)
            for i, gap in enumerate(gaps)])
        return path.astype(np.int64), meter_index

    def crf_nll(self, activations: torch.Tensor, state_path: np.ndarray,
                meter_index: int) -> torch.Tensor:
        """-log p(annotated path | activations) = logZ - score(path). Differentiable w.r.t.
        activations (the e2e emission) and transition_lambda."""
        chassis, dp = self.chassis, self.chassis.dynamic_programs[meter_index]
        space = chassis.state_spaces[meter_index]
        densities = self.log_class_densities(activations)
        log_transition = self.log_tempo_transition()
        state_to_class = chassis.state_to_classes[meter_index]
        log_init = chassis.log_initial_distributions[meter_index]

        # score of the annotated path: emission at each frame + the tempo kernel at boundaries
        # (the within-beat +1 advance has probability 1 -> contributes 0)
        path = torch.from_numpy(state_path).to(self.device)
        emission_score = densities[
            torch.arange(len(state_path), device=self.device),
            state_to_class[path].long()].sum()
        intervals = space.state_interval_frames[state_path]
        is_boundary = np.where(np.isin(state_path[1:], space.first_states.reshape(-1)))[0]
        from_index = torch.from_numpy(
            intervals[is_boundary] - self._min_interval).long().to(self.device)
        to_index = torch.from_numpy(
            intervals[is_boundary + 1] - self._min_interval).long().to(self.device)
        transition_score = log_transition[from_index, to_index].sum()
        path_score = log_init[path[0]] + emission_score + transition_score

        log_z = dp.forward_log_likelihood(log_init, log_transition, densities,
                                          state_to_class=state_to_class)
        return log_z - path_score

    def make_rung(self, **kwargs) -> DBN2016:
        """Deployment: a plain DBN2016 whose transition_lambda is the LEARNED value."""
        return DBN2016(fps=self.chassis.fps, transition_lambda=self.transition_lambda,
                       beats_per_bar=tuple(self.chassis.beats_per_bar), **kwargs)
