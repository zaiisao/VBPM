"""R1 -- Baseline A in OUR framework: the Krebs bar-pointer HMM, hand-set factors, our inference.

The same model R0 runs (madmom's DBN), rebuilt on our own code and decoded with our own DP. THE
CERTIFICATE: if R1 reproduces R0, our inference is validated on a real model and the ladder's anchor
is set -- after which R0 -> R4 is a clean "hand-set vs learned bar-pointer, same inference" axis.
R2+ change ONLY how the factors are produced.

Status: R1 and a matched madmom decode the SAME Viterbi path (100% of frames, every song tested) and
score identically to 4 decimals. "Matched" matters: madmom's DBNDownBeatTrackingProcessor defaults
wrap this model in three decode heuristics R1 does not implement -- num_tempi=60 (a coarser
log-spaced tempo grid), threshold=0.05 (crop to the main above-threshold segment) and correct=True
(report the activation peak inside a beat region rather than the region entry). Those are worth
~+0.006 beat F on our val set, essentially all of it threshold=0.05 cropping the fade-in/fade-out
that only the Ballroom subset has. They are decode heuristics, not the model.

Everything here is HAND-SET; nothing is learned:
  state space : Krebs 2015 -- a tempo of i frames-per-beat owns exactly i states, so the pointer
                advances exactly +1 state per frame (see common/state_space.py for why a uniform
                grid fails).
  transition  : deterministic advance inside a beat; madmom's exponential tempo mix at beat
                boundaries, exp(-transition_lambda |ratio - 1|), row-normalized.
  emission    : Boeck 2016's mutually-exclusive {no-beat, beat, downbeat} observation model on the
                DECORRELATED activation (max(beat - downbeat, eps), downbeat) -- the same
                decorrelation R0 uses, and required because that model assumes the columns sum to <=1.

Everything is torch (GPU + autograd-ready), which is what R2-R4 need and what madmom cannot give.
"""
import numpy as np
import torch

from common.deployment import threshold_crop
from common.readout import state_path_to_events
from common.state_space import BarPointerStateSpace
from common.structured_dp import StructuredBarPointerDP


class HandcraftedBarPointerHMM:
    def __init__(self, fps: float, min_bpm: float = 55.0, max_bpm: float = 215.0,
                 beats_per_bar: int = 4, observation_lambda: int = 16,
                 transition_lambda: float = 100.0, eps: float = 1e-5,
                 dtype: torch.dtype = torch.float64, device: str = "cuda",
                 num_tempi: int = None):
        """dtype defaults to float64 because R1's whole job is to reproduce madmom's MAP path. In
        float32 the score accumulated over ~20k frames drifts enough to lose it (measured: 0.12 nats
        worse than madmom on a Beethoven val song -- a strictly suboptimal decode, which Viterbi is
        not allowed to be). The cost is small: this DP is kernel-launch bound, not compute bound.

        num_tempi: None = every integer interval (the exact model, what the certificate runs);
        60 = madmom's shipped log-spaced grid (see common/state_space.py).
        """
        self.state_space = BarPointerStateSpace(fps, min_bpm, max_bpm, beats_per_bar,
                                                observation_lambda, num_tempi=num_tempi)
        self.dynamic_program = StructuredBarPointerDP(self.state_space, device=device, dtype=dtype)
        self.fps, self.eps, self.dtype, self.device = fps, eps, dtype, device
        self.log_initial_distribution = torch.full(
            (self.state_space.num_states,), -float(np.log(self.state_space.num_states)),
            dtype=dtype, device=device)
        self.log_tempo_transition = self.dynamic_program.build_log_tempo_transition(transition_lambda)
        # The DP gathers each state's class density on the fly from this map -- the per-state
        # [num_frames, num_states] emission (2.08 GB per 3-min song) never exists; only the
        # [num_frames, 3] class table (372 KB) does.
        self.state_to_class = torch.from_numpy(self.state_space.position_classes).to(device)

    def _log_class_densities(self, beat_activation: np.ndarray,
                             downbeat_activation: np.ndarray) -> torch.Tensor:
        """Boeck 2016's 3-class densities on the decorrelated activation.

        Returns [num_frames, 3] -- one column per observation class (NO_BEAT, BEAT, DOWNBEAT). Every
        state emits through exactly one class, so this table plus state_to_class IS the emission.
        """
        eps = self.eps
        beat_activation = np.clip(beat_activation, eps, 1 - eps)
        downbeat_activation = np.clip(downbeat_activation, eps, 1 - eps)
        # Decorrelate: the model needs mutually-exclusive classes, but the frontend's beat channel
        # fires on downbeats too, so beat + downbeat ~ 2 there and the no-beat density goes negative.
        beat_not_downbeat = np.maximum(beat_activation - downbeat_activation, eps)
        # The no-beat probability is shared out over the observation_lambda - 1 non-beat states.
        num_non_beat_states = self.state_space.observation_lambda - 1
        no_beat_probability = 1.0 - beat_not_downbeat - downbeat_activation

        # Deliberately NOT clipped, and madmom does not clip it either. A zero here means the frame
        # CANNOT be a non-beat state; flooring it to log(eps/15) = -14.2 downgrades an impossibility
        # to an improbability, and Viterbi will buy it -- measured on a Ballroom val song where this
        # fires on just 5 frames: 92.07% path agreement and a 0.0187-nat SUBOPTIMAL decode. Safe
        # because decorrelation guarantees no_beat_probability >= 0 (so the worst case is
        # log(0) = -inf, never log(negative) = nan), and the beat/downbeat classes are floored at
        # eps and therefore always finite, so no frame can be -inf in all three classes.
        with np.errstate(divide="ignore"):
            log_class_densities = np.stack(                              # [num_frames, 3]
                [np.log(no_beat_probability / num_non_beat_states),
                 np.log(beat_not_downbeat), np.log(downbeat_activation)], axis=1)
        return torch.from_numpy(log_class_densities).to(dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def decode(self, activations, threshold: float = 0.0, correct: bool = False) -> dict:
        """activations: [num_frames, 2] probabilities (beat, downbeat). Same interface as every rung.

        threshold / correct: madmom's shipped deployment lessons (crop dead air before decoding /
        report beats at activation peaks), behavior-copied, named as madmom names them. OFF by
        default: the bare model is what the certificate pins to madmom and what rung comparisons
        use. Turn BOTH on with num_tempi=60 to reproduce R0-as-shipped. See common/deployment.py.
        """
        activations = (activations.detach().cpu().numpy() if torch.is_tensor(activations)
                       else np.asarray(activations))
        # The crop must see what the decoder sees: the decorrelated pair (madmom thresholds the
        # combined activation it decodes, not the raw channels).
        eps = self.eps
        beat = np.clip(activations[:, 0], eps, 1 - eps)
        downbeat = np.clip(activations[:, 1], eps, 1 - eps)
        decorrelated = np.stack([np.maximum(beat - downbeat, eps), downbeat], axis=-1)
        decorrelated, first_frame = threshold_crop(decorrelated, threshold)
        if not len(decorrelated):
            return {"beats": np.array([]), "downbeats": np.array([])}

        activations = activations[first_frame:first_frame + len(decorrelated)]
        log_class_densities = self._log_class_densities(activations[:, 0], activations[:, 1])
        state_path = self.dynamic_program.viterbi(
            self.log_initial_distribution, self.log_tempo_transition, log_class_densities,
            state_to_class=self.state_to_class).cpu().numpy()
        return state_path_to_events(state_path, self.state_space, self.fps,
                                    snap_to_activations=decorrelated if correct else None,
                                    first_frame=first_frame)


if __name__ == "__main__":
    # Synthetic 120 BPM 4/4, 10 s. NOTE: this test passed even with the BROKEN uniform grid, because
    # at fps=50/120 BPM the fractional advance happened to be 0.96 (~1). Real data at 86 fps is what
    # caught it. Kept as a smoke test only -- the real gate is R1 vs R0.
    fps = 50.0
    num_frames = int(round(10 * fps))
    beat_period_frames = fps * 60.0 / 120.0
    activations = np.zeros((num_frames, 2))
    activations[:, 0], activations[:, 1] = 0.05, 0.02
    beat_frames = np.round(np.arange(0.0, num_frames, beat_period_frames)).astype(int)
    beat_frames = beat_frames[beat_frames < num_frames]
    activations[beat_frames, 0] = 0.9
    activations[beat_frames[::4], 1] = 0.92

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HandcraftedBarPointerHMM(fps=fps, device=device)
    print(model.state_space)
    events = model.decode(activations)
    print(f"synthetic 120 BPM 4/4, 10 s (fps={fps:.1f}), device={device}:")
    print(f"  beats {len(events['beats'])} (expect ~20) | "
          f"downbeats {len(events['downbeats'])} (expect ~5)")
    if len(events["beats"]) > 1:
        mean_inter_beat_interval = float(np.diff(events["beats"]).mean())
        print(f"  mean IBI {mean_inter_beat_interval:.3f}s -> "
              f"{60.0 / mean_inter_beat_interval:.1f} BPM (expect ~120)")
