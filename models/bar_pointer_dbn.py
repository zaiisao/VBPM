"""Bar-pointer Dynamic Bayesian Network as a (trainable) prior p(z).

This is a clean, vectorized PyTorch re-implementation of the madmom bar-pointer
DBN — the model CHART's prior is *supposed* to be (FAITHFULNESS.md: "a VAE whose
prior is the madmom bar-pointer DBN"). It is written from the papers + madmom's
reference code (which is correct but unusable as a differentiable module):

  * Krebs, Böck, Widmer, "An Efficient State Space Model for Joint Tempo and
    Meter Tracking", ISMIR 2015.            (state space + exponential tempo transition)
  * Böck, Krebs, Widmer, "Joint Beat and Downbeat Tracking with Recurrent
    Neural Networks", ISMIR 2016.           (RNN observation model)
  * madmom/features/beats_hmm.py            (reference implementation)

Why re-implement instead of calling madmom:
  1. madmom's HMM is fixed-parameter; here the tempo-transition `lambda` (and
     optionally a learned emission) are nn.Parameters, so the prior is TRAINABLE
     — the SMC-blind-spot point: the optimal lambda varies per song, so a learned
     /adaptive lambda should beat any single fixed value.
  2. It runs on GPU, in log-space, and exposes a DIFFERENTIABLE forward
     log-likelihood (for end-to-end training) alongside Viterbi (for decoding).
  3. It is deployed the way a DBN is meant to be: audio-conditioned inference
     (Viterbi over the observed activations), not prior free-running.

State: z_t = (position-in-bar phi in [0, num_beats), tempo/interval n = frames per
beat). The bar pointer advances phi deterministically by 1 frame/step within a beat
and may change tempo at beat boundaries. Beats are read off the decoded state path
where the integer beat index increments; downbeats where the bar wraps to 0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

_NEG_INF = -1e30


# --------------------------------------------------------------------------- #
#  State space  (Krebs 2015, BeatStateSpace / BarStateSpace)
# --------------------------------------------------------------------------- #

def _intervals(min_interval: float, max_interval: float, num_intervals: int | None) -> np.ndarray:
    """Modeled beat intervals (frames-per-beat). Linear by default; log-spaced and
    de-duplicated to `num_intervals` if that is smaller (madmom BeatStateSpace)."""
    lin = np.arange(round(min_interval), round(max_interval) + 1)
    if num_intervals is None or num_intervals >= len(lin):
        return lin.astype(int)
    n = num_intervals
    out = lin
    while len(out) < num_intervals:
        out = np.unique(np.round(np.logspace(
            np.log2(min_interval), np.log2(max_interval), n, base=2)))
        n += 1
    return out.astype(int)


def build_bar_state_space(num_beats: int, min_interval: float, max_interval: float,
                          num_intervals: int | None):
    """Returns the flattened bar state space (see Krebs 2015 / madmom BarStateSpace).

    For each of `num_beats` beats and each interval i, a run of i position states
    linspace(0,1,i,endpoint=False). Returns numpy arrays:
      positions  [S]  in [0, num_beats)   — fractional position in the bar
      intervals  [S]  frames-per-beat (tempo) of each state
      first      list[num_beats] of arrays — position-0 state of each interval (per beat)
      last       list[num_beats] of arrays — position-(i-1) state of each interval (per beat)
    """
    iv = _intervals(min_interval, max_interval, num_intervals)
    positions, intervals = [], []
    first_per_beat, last_per_beat = [], []
    offset = 0
    for b in range(num_beats):
        firsts = (offset + np.cumsum(np.r_[0, iv[:-1]])).astype(int)   # pos-0 of each interval
        lasts = (offset + np.cumsum(iv) - 1).astype(int)              # pos-(i-1) of each interval
        first_per_beat.append(firsts)
        last_per_beat.append(lasts)
        for i in iv:
            positions.append(np.linspace(0.0, 1.0, int(i), endpoint=False) + b)
            intervals.append(np.full(int(i), int(i)))
            offset += int(i)
    return (np.concatenate(positions), np.concatenate(intervals).astype(int),
            first_per_beat, last_per_beat)


# --------------------------------------------------------------------------- #
#  Bar-pointer DBN
# --------------------------------------------------------------------------- #

class BarPointerDBN(nn.Module):
    """Trainable bar-pointer DBN usable as CHART's prior p(z) + inference engine.

    Args:
      fps:            frames per second of the activation (default 86.133 = 22050/256).
      num_beats:      beats per bar (4 = 4/4). min/max_bpm define the tempo range.
      num_intervals:  number of (log-spaced) tempo states; smaller = coarser/faster.
      observation_lambda: split each beat into this many parts; the first part is the
                          beat region (madmom default 16).
      learnable_lambda: if True, the tempo-transition lambda is an nn.Parameter
                        (per-beat, so the downbeat boundary can differ).
      init_lambda:    initial transition lambda (madmom default 100).
    """

    def __init__(self, fps: float = 22050 / 256, num_beats: int = 4,
                 min_bpm: float = 55.0, max_bpm: float = 215.0,
                 num_intervals: int | None = 50, observation_lambda: int = 16,
                 learnable_lambda: bool = False, init_lambda: float = 100.0,
                 beats_only: bool = False):
        super().__init__()
        # beats_only: a BEAT pointer (no bar / downbeat structure) -- the right model
        # for beats-only sets like SMC (madmom DBNBeatTracker). Uses only the beat
        # activation; the bar wraps every beat.
        self.beats_only = bool(beats_only)
        if self.beats_only:
            num_beats = 1
        self.fps = float(fps)
        self.num_beats = int(num_beats)
        self.observation_lambda = int(observation_lambda)
        min_interval = 60.0 * fps / max_bpm          # frames per beat at the fastest tempo
        max_interval = 60.0 * fps / min_bpm
        pos, iv, first_pb, last_pb = build_bar_state_space(
            num_beats, min_interval, max_interval, num_intervals)
        S = len(pos)
        self.num_states = S

        # ---- transition structure (indices fixed; log-probs depend on lambda) ----
        all_first = np.concatenate(first_pb)
        # (a) deterministic within-beat advance: every non-first state s <- s-1, prob 1
        adv_to = np.setdiff1d(np.arange(S), all_first)
        adv_from = adv_to - 1
        # (b) tempo-change edges at beat boundaries: last(beat b-1) -> first(beat b)
        bnd_from, bnd_to, bnd_from_int, bnd_to_int, bnd_beat = [], [], [], [], []
        for b in range(num_beats):
            f_states = first_pb[b]                    # to   (pos 0 of beat b)
            l_states = last_pb[b - 1]                 # from (pos i-1 of beat b-1)
            fi = iv[l_states]                         # from intervals
            ti = iv[f_states]                         # to intervals
            # full cartesian boundary edges (dense over interval pairs)
            F, T = np.meshgrid(np.arange(len(l_states)), np.arange(len(f_states)), indexing="ij")
            bnd_from.append(l_states[F.ravel()]); bnd_to.append(f_states[T.ravel()])
            bnd_from_int.append(fi[F.ravel()]); bnd_to_int.append(ti[T.ravel()])
            bnd_beat.append(np.full(F.size, b))
        bnd_from = np.concatenate(bnd_from); bnd_to = np.concatenate(bnd_to)

        # combined edge list: advance edges (logp=0) + boundary edges (logp from lambda)
        from_idx = np.concatenate([adv_from, bnd_from])
        to_idx = np.concatenate([adv_to, bnd_to])
        self.n_adv = len(adv_from)
        self.register_buffer("from_idx", torch.as_tensor(from_idx, dtype=torch.long))
        self.register_buffer("to_idx", torch.as_tensor(to_idx, dtype=torch.long))
        self.register_buffer("bnd_from_int", torch.as_tensor(np.concatenate(bnd_from_int), dtype=torch.float32))
        self.register_buffer("bnd_to_int", torch.as_tensor(np.concatenate(bnd_to_int), dtype=torch.float32))
        self.register_buffer("bnd_beat", torch.as_tensor(np.concatenate(bnd_beat), dtype=torch.long))

        # ---- observation pointers: state -> {0 no-beat, 1 beat, 2 downbeat} ----
        border = 1.0 / self.observation_lambda
        within = (pos % 1.0)
        ptr = np.zeros(S, dtype=np.int64)
        ptr[within < border] = 1                      # beat region
        if not self.beats_only:
            ptr[pos < border] = 2                     # downbeat region (first beat of bar)
        self.register_buffer("obs_ptr", torch.as_tensor(ptr, dtype=torch.long))
        self.register_buffer("state_pos", torch.as_tensor(pos, dtype=torch.float32))
        self.register_buffer("state_int", torch.as_tensor(iv, dtype=torch.float32))
        self.num_classes = 2 if self.beats_only else 3   # {no-beat, beat[, downbeat]}

        # ---- learnable tempo-transition lambda (per beat boundary) ----
        log_lam = torch.full((num_beats,), float(np.log(init_lambda)))
        self.log_lambda = nn.Parameter(log_lam) if learnable_lambda else None
        if not learnable_lambda:
            self.register_buffer("_log_lambda_fixed", log_lam)

    # ------------------------------------------------------------------ #
    @property
    def _lam(self) -> Tensor:
        return (self.log_lambda if self.log_lambda is not None else self._log_lambda_fixed).exp()

    def _edge_logp(self, log_lambda: Tensor | None = None) -> Tensor:
        """Log transition prob per edge. Advance edges = 0 (prob 1). Boundary edges
        = exp(-lambda|to/from - 1|) normalized per from-state (exponential_transition).
        `log_lambda` ([num_beats]) overrides self.log_lambda for a PER-SONG transition lambda;
        gradients flow into it (end-to-end-trainable tempo stiffness)."""
        lam = (self._lam if log_lambda is None else log_lambda.exp())    # [num_beats]
        lam = lam[self.bnd_beat]                                          # [E_bnd]
        ratio = self.bnd_to_int / self.bnd_from_int
        logp = -lam * (ratio - 1.0).abs()                               # unnormalized log prob
        # normalize over to-states sharing the same from-state (per-from softmax in log-space)
        f = self.from_idx[self.n_adv:]
        m = torch.full((self.num_states,), _NEG_INF, device=logp.device).scatter_reduce(
            0, f, logp, reduce="amax", include_self=True)
        z = torch.zeros(self.num_states, device=logp.device).index_add_(
            0, f, (logp - m[f]).exp())
        logZ = m + z.clamp_min(1e-30).log()
        bnd_logp = logp - logZ[f]
        return torch.cat([torch.zeros(self.n_adv, device=logp.device), bnd_logp])

    def observation_logp(self, activations: Tensor) -> Tensor:
        """[T,2] beat/downbeat probs -> [T,S] per-state observation log-likelihood
        (madmom RNNDownBeatTrackingObservationModel.log_densities + pointers).

        madmom assumes EXCLUSIVE [beat, downbeat] (sum<=1). Beat This (and most
        sigmoid trackers) emit INDEPENDENT beat/downbeat where downbeat ⊂ beat, so we
        convert: non-downbeat-beat = beat - downbeat, no-beat = 1 - beat."""
        a = activations.clamp(0.0, 1.0)
        if self.beats_only:                                       # beat activation only (col 0)
            beat = a[:, 0].clamp_min(1e-6)
            no_beat = (1.0 - a[:, 0]).clamp_min(1e-6) / (self.observation_lambda - 1)
            dens = torch.stack([no_beat.log(), beat.log()], dim=-1)            # [T,2]
            return dens[:, self.obs_ptr]
        beat_incl, down = a[:, 0], a[:, 1]
        p_beat = (beat_incl - down).clamp_min(1e-6)                # non-downbeat beat
        p_down = down.clamp_min(1e-6)
        no_beat = (1.0 - beat_incl).clamp_min(1e-6) / (self.observation_lambda - 1)
        dens = torch.stack([no_beat.log(), p_beat.log(), p_down.log()], dim=-1)  # [T,3]
        return dens[:, self.obs_ptr]                                          # [T,S]

    def _segment_logsumexp(self, vals: Tensor, idx: Tensor) -> Tensor:
        m = torch.full((self.num_states,), _NEG_INF, device=vals.device).scatter_reduce(
            0, idx, vals, reduce="amax", include_self=True)
        z = torch.zeros(self.num_states, device=vals.device).index_add_(
            0, idx, (vals - m[idx]).exp())
        return m + z.clamp_min(1e-30).log()

    def forward_logpartition(self, obs_logp: Tensor, elp: Tensor | None = None) -> Tensor:
        """Differentiable forward-algorithm log-partition: log sum_paths exp(score(path))
        for a given per-state emission obs_logp [T,S]. The CRF loss is
        (forward_logpartition(obs) - forward_logpartition(obs masked to GT-consistent states)).
        Pass `elp` (precomputed edge log-probs, e.g. with a per-song lambda) to share it."""
        elp = self._edge_logp() if elp is None else elp
        T = obs_logp.shape[0]
        alpha = obs_logp[0] - np.log(self.num_states)                         # uniform init prior
        for t in range(1, T):
            alpha = self._segment_logsumexp(alpha[self.from_idx] + elp, self.to_idx) + obs_logp[t]
        return torch.logsumexp(alpha, dim=0)

    def forward_loglik(self, activations: Tensor) -> Tensor:
        """log p(activations) under the DBN with the FIXED madmom observation. activations [T,2]."""
        return self.forward_logpartition(self.observation_logp(activations))

    # ---- learned-emission entry point (the encoder / "posterior") ------------ #
    def class_logp_to_states(self, class_logp: Tensor) -> Tensor:
        """[T, C] per-frame per-class emission log-potentials -> [T, S] per-state
        observation (via the pointer map). C = num_classes (0 no-beat, 1 beat[, 2 down]).
        This replaces madmom's fixed log(activation): a learned head produces class_logp
        from audio features, and the DBN structure does the rest."""
        return class_logp[:, self.obs_ptr]

    def forward_backward(self, obs_logp: Tensor) -> Tensor:
        """Differentiable forward-backward. obs_logp [T,S] -> per-frame per-CLASS log
        posterior marginal [T, C]. The structured posterior given the fixed DBN prior;
        train the emission through this so q(z|x) refines p_DBN(z)."""
        elp = self._edge_logp()
        T, S = obs_logp.shape
        # forward
        a = [obs_logp[0] - np.log(S)]
        for t in range(1, T):
            a.append(self._segment_logsumexp(a[-1][self.from_idx] + elp, self.to_idx) + obs_logp[t])
        alpha = torch.stack(a)                                                  # [T,S]
        logZ = torch.logsumexp(alpha[-1], dim=0)
        # backward (group messages by FROM-state)
        b = [torch.zeros(S, device=obs_logp.device)]
        for t in range(T - 2, -1, -1):
            msg = elp + obs_logp[t + 1][self.to_idx] + b[0][self.to_idx]
            b.insert(0, self._segment_logsumexp(msg, self.from_idx))
        beta = torch.stack(b)                                                   # [T,S]
        gamma = alpha + beta - logZ                                             # [T,S] log p(s_t | x)
        marg = [torch.logsumexp(gamma[:, self.obs_ptr == c], dim=1)            # [T] per class
                for c in range(self.num_classes)]
        return torch.stack(marg, dim=1)                                         # [T, C]

    # ---- Viterbi decode (shared core) ---------------------------------------- #
    @torch.no_grad()
    def _viterbi(self, obs: Tensor, elp: Tensor | None = None) -> Tensor:
        elp = self._edge_logp() if elp is None else elp
        T = obs.shape[0]
        delta = obs[0] - np.log(self.num_states)
        bptr = torch.empty((T, self.num_states), dtype=torch.long, device=obs.device)
        bptr[0] = torch.arange(self.num_states, device=obs.device)
        for t in range(1, T):
            cand = delta[self.from_idx] + elp
            best = torch.full((self.num_states,), _NEG_INF, device=obs.device).scatter_reduce(
                0, self.to_idx, cand, reduce="amax", include_self=True)
            is_max = cand >= (best[self.to_idx] - 1e-6)
            bp = torch.zeros(self.num_states, dtype=torch.long, device=obs.device)
            bp[self.to_idx[is_max]] = self.from_idx[is_max]
            bptr[t] = bp
            delta = best + obs[t]
        path = torch.empty(T, dtype=torch.long, device=obs.device)
        path[-1] = int(delta.argmax())
        for t in range(T - 1, 0, -1):
            path[t - 1] = bptr[t, path[t]]
        return path

    def _snap(self, region: Tensor, act: Tensor) -> Tensor:
        """madmom `correct`: per contiguous run of `region` (states in a beat range), emit
        one beat at the frame with the highest activation `act` (beats.py:154-172)."""
        bp = region.to(torch.int8)
        idx = (torch.nonzero(bp[1:] != bp[:-1], as_tuple=True)[0] + 1).tolist()
        if bp[0]:
            idx = [0] + idx
        if int(bp[-1]):
            idx = idx + [bp.numel()]
        peaks = [int(act[l:r].argmax()) + l for l, r in zip(idx[::2], idx[1::2]) if r > l]
        return torch.as_tensor(peaks, dtype=torch.long, device=act.device)

    def _path_to_beats(self, path: Tensor, beat_snap=None, down_snap=None):
        """Path -> (beat_frames, downbeat_frames). With *_snap activations given, snap to the
        activation peak in each beat/downbeat region (madmom correct=True); else emit at the
        phase-wrap frame (correct=False)."""
        cls = self.obs_ptr[path]                                  # 0 no-beat, 1 beat, 2 downbeat
        if beat_snap is not None:
            beat_frames = self._snap(cls >= 1, beat_snap)
        else:
            within = self.state_pos[path] % 1.0
            beat_frames = torch.nonzero(within[1:] < within[:-1], as_tuple=True)[0] + 1
        if self.beats_only:
            down_frames = beat_frames[:0]
        elif down_snap is not None:
            down_frames = self._snap(cls == 2, down_snap)
        else:
            pos = self.state_pos[path]
            down_frames = torch.nonzero(pos[1:] < pos[:-1], as_tuple=True)[0] + 1
        return beat_frames, down_frames

    @torch.no_grad()
    def decode(self, activations: Tensor, correct: bool = True):
        """Audio-conditioned decode from the FIXED madmom observation -> (beats, downbeats).
        correct=True snaps beats to the activation peak (madmom default)."""
        path = self._viterbi(self.observation_logp(activations))
        if not correct:
            return self._path_to_beats(path)
        down = activations[:, 1] if activations.shape[1] > 1 else None
        return self._path_to_beats(path, beat_snap=activations[:, 0], down_snap=down)

    @torch.no_grad()
    def decode_emission(self, class_logp: Tensor, correct: bool = True, snap_act: Tensor | None = None,
                        elp: Tensor | None = None):
        """Decode from a LEARNED per-class emission [T,C] -> (beats, downbeats). The emission
        sets the Viterbi PATH; beats snap to `snap_act` if given (e.g. the sharp frontend
        activation) else to the emission's own (possibly mushy) beat-class probability.
        `elp` (per-song edge log-probs) lets the transition lambda vary per song."""
        path = self._viterbi(self.class_logp_to_states(class_logp), elp=elp)
        if not correct:
            return self._path_to_beats(path)
        beat_snap = snap_act if snap_act is not None else class_logp[:, 1].exp()
        down = class_logp[:, 2].exp() if class_logp.shape[1] > 2 else None
        return self._path_to_beats(path, beat_snap=beat_snap, down_snap=down)
