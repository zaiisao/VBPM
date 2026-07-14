"""R0 -- Baseline A: the madmom bar-pointer DBN (Krebs/Boeck 2015).

Self-contained: it calls the official madmom DBNDownBeatTrackingProcessor directly and formats the
input with the standard decorrelation convention (Boeck et al.):

    combined = [ max(beat - downbeat, floor),  downbeat ]

Why decorrelate: madmom's downbeat observation model computes the "no-beat" log-density as roughly
    log( (1 - (beat + downbeat)) / normalizer ),
so it assumes beat + downbeat <= 1. A neural frontend whose beat channel fires on every beat
(downbeats included) gives beat + downbeat ~ 2 at each downbeat -> log(negative), which madmom would
silently clamp. Decorrelating turns the two columns into a valid [P(beat-not-downbeat), P(downbeat)]
pair that sums to <= 1, so the observation model is well-formed and nothing is clamped. (Measured on
our val cache: decorrelating is worth ~+0.06 F vs feeding raw activations -- not optional; the raw
form's log(negative) nan corrupts the whole Viterbi path.) R0 is the literal, correct DBN baseline --
the certificate R1 (our own PyTorch bar-pointer DBN) must reproduce.

This follows the Beat This / Beat Transformer paradigm -- the joint bar-pointer DBN + Boeck
decorrelation that both SOTA systems independently converged on. WaveBeat's separate-tracker approach
is a different model class, considered and NOT adopted (unproven on our data). The two knobs let R0
match each system's exact recipe; per our ablation they are near-equivalent, so the defaults are what
matter:

    input_form : "prob"  activations already in [0,1] (our cache act2)                    [DEFAULT]
                 "logit" raw pre-sigmoid logits (Beat This / Beat Transformer) -> we sigmoid them
    bounding   : "clip"    np.clip(x, eps, 1-eps)          (ours, WaveBeat)                [DEFAULT]
                 "squeeze" x*(1-eps) + eps/2               (Beat This)
                 "none"    no clip (Beat Transformer). For logit input; unsafe on probs with exact 0/1
    eps        : bounds values off 0/1 and floors the decorrelation; 1e-5 (Beat This/ours) is a hair
                 better than 1e-8 (WaveBeat), but its exact value barely matters.

  Beat This        = input_form="logit", bounding="squeeze"
  Beat Transformer = input_form="logit", bounding="none"
  our prob cache   = input_form="prob",  bounding="clip"    (the defaults)

fps: REQUIRED and frontend-specific -- a property of the activations, never a global constant, and a
wrong fps mis-scales every DBN tempo. Pass record['fps'] from the cache. Native rates differ by model
(Beat This 50, MERT ~75, WaveBeat 172.27); our cache currently interpolates every frontend up to 86.13
(= 22050/256), so record['fps'] is 86.13 today -- but reading it keeps R0 correct the moment a frontend
is cached at its own native rate.
"""
import numpy as np
from madmom.features.downbeats import DBNDownBeatTrackingProcessor


def _to_numpy(x) -> np.ndarray:
    """Accept a numpy array or a (possibly CUDA) torch tensor."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


class MadmomDBN:
    """R0: the official madmom joint bar-pointer DBN, fed the standard decorrelated activation.

    Follows the Beat This / Beat Transformer paradigm; input_form and bounding match each system's
    exact recipe (see module docstring). beats_per_bar=[3,4], min_bpm=55, max_bpm=215,
    transition_lambda=100 are the settings both SOTA trackers use -- a faithful baseline, not re-tuned.
    """

    def __init__(
        self,
        fps: float,
        input_form: str = "prob",
        bounding: str = "clip",
        eps: float = 1e-5,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        beats_per_bar=(3, 4),
        transition_lambda: int = 100
    ):
        if input_form not in ("prob", "logit"):
            raise ValueError(f"input_form must be 'prob' or 'logit', got {input_form!r}")

        if bounding not in ("clip", "squeeze", "none"):
            raise ValueError(f"bounding must be 'clip', 'squeeze' or 'none', got {bounding!r}")

        self.input_form = input_form
        self.bounding = bounding
        self.eps = eps
        self._dbn = DBNDownBeatTrackingProcessor(
            beats_per_bar=list(beats_per_bar),
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=fps,
            transition_lambda=transition_lambda
        )

    def _bound(self, beat, downbeat):
        """Bound activations off exact 0/1 and return the decorrelation floor to pair with them."""
        eps = self.eps
        
        if self.bounding == "clip":
            return np.clip(beat, eps, 1 - eps), np.clip(downbeat, eps, 1 - eps), eps
        if self.bounding == "squeeze":
            return beat * (1 - eps) + eps / 2, downbeat * (1 - eps) + eps / 2, eps / 2
        
        return beat, downbeat, 1e-12                                  # none (Beat Transformer; tiny floor for safety)

    def decode(self, observations) -> dict:
        """observations: [T, 2] activations (beat, downbeat), in the form given by input_form.

        Returns {'beats': seconds[N], 'downbeats': seconds[M]} -- the common deployment interface
        every rung exposes; R0 fulfils it via madmom's joint bar-pointer DBN.
        """
        obs = _to_numpy(observations)
        if obs.ndim != 2 or obs.shape[1] < 2:
            raise ValueError(f"expected [T, 2] activations (beat, downbeat), got {obs.shape}")

        beat, downbeat = obs[:, 0], obs[:, 1]
        if self.input_form == "logit":
            beat = 1.0 / (1.0 + np.exp(-beat))
            downbeat = 1.0 / (1.0 + np.exp(-downbeat))

        beat, downbeat, floor = self._bound(beat, downbeat)
        combined = np.stack([np.maximum(beat - downbeat, floor), downbeat], axis=-1)   # sums <= 1

        out = np.asarray(self._dbn(combined))                     # [M, 2] -> (time, position-in-bar)
        beats = out[:, 0] if out.size else np.array([])
        downbeats = out[out[:, 1] == 1, 0] if out.size else np.array([])

        return {"beats": np.asarray(beats), "downbeats": np.asarray(downbeats)}

if __name__ == "__main__":
    import warnings

    # Smoke test: synthetic 120 BPM, 4/4, 10 s of PROBABILITY activations (beat channel fires on every
    # beat, downbeats included). Proves the wrapper runs end to end.
    fps = 50.0   # Beat This's native rate; the synthetic below mimics its channel form
    T = int(round(10 * fps))
    beat_period = fps * 60.0 / 120.0
    obs = np.zeros((T, 2))
    obs[:, 0], obs[:, 1] = 0.05, 0.02                    # distinct baselines (avoid exact-equal channels)
    beat_frames = np.round(np.arange(0.0, T, beat_period)).astype(int)
    beat_frames = beat_frames[beat_frames < T]
    obs[beat_frames, 0] = 0.9
    obs[beat_frames[::4], 1] = 0.92

    out = MadmomDBN(fps=fps).decode(obs)
    print(f"synthetic 120 BPM 4/4, 10 s (fps={fps:.1f}), probability activations (defaults):")
    print(f"  detected beats:     {len(out['beats']):2d}   (expect ~20)")
    print(f"  detected downbeats: {len(out['downbeats']):2d}   (expect ~5)")
    if len(out["beats"]) > 1:
        ibi = float(np.diff(out["beats"]).mean())
        print(f"  mean inter-beat interval: {ibi:.3f}s  ->  {60.0 / ibi:.1f} BPM  (expect ~120)")

    # Verify the option matrix all runs and lands on ~the same answer (bounding is cosmetic; the logit
    # path recovers the probs via sigmoid). 'none' on prob input can hit log(0) -> harmless -inf.
    logit_obs = np.log(np.clip(obs, 1e-6, 1 - 1e-6) / (1 - np.clip(obs, 1e-6, 1 - 1e-6)))
    print("\noption matrix (beats / downbeats -- should be ~equal across cells):")
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for form, src in (("prob", obs), ("logit", logit_obs)):
            for bnd in ("clip", "squeeze", "none"):
                r = MadmomDBN(fps=fps, input_form=form, bounding=bnd).decode(src)
                print(f"  input_form={form:5s} bounding={bnd:7s} -> {len(r['beats']):2d} / {len(r['downbeats']):2d}")
