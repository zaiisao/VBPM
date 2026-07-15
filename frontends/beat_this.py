"""Beat This frontend (Foscarin, Schlueter & Widmer, ISMIR 2024; external/beat_this submodule).

Wraps the OFFICIAL beat_this.inference.Audio2Frames -- their spectrogram, their chunked
split-predict-aggregate, their checkpoint loading -- rather than re-plumbing the model. We add only
the property surface the Tracker needs (fps, activation_form) and optional resampling of the
activation grid.

Properties of what this emits:
  * native grid is EXACTLY 50 fps (22050 Hz audio, hop 441); pass target_fps to linearly
    interpolate the logits onto another grid (our activation caches historically use
    86.1328125 = 22050/256, so cached and live activations line up frame for frame).
  * activations are LOGITS (activation_form="logit"): Beat This's own DBN path feeds
    sigmoid(logits) to madmom, and the Tracker applies the same conversion where a decoder wants
    probabilities.

Checkpoints (all cached locally under ~/.cache/torch/hub/checkpoints/):
  * "final0" -- trained on everything except GTZAN. Fine for deployment and demos; NOT fold-honest
    for evaluation on the training datasets.
  * "fold0".."fold7" -- the Beat This 8-fold protocol. For any number reported on our val folds,
    use the checkpoint whose held-out fold matches; final0 numbers on those songs are leakage.
"""
import sys
from pathlib import Path
from typing import Optional

import torch

from frontends import Frontend

_BEAT_THIS_ROOT = Path(__file__).resolve().parent.parent / "external" / "beat_this"


class BeatThisFrontend(Frontend):
    name = "beat_this"
    activation_form = "logit"
    bounding = "squeeze"       # Beat This's published convention: sigmoid(x)*(1-eps) + eps/2.
                               # With this, Tracker(frontend, "madmom_dbn") is EVENT-IDENTICAL to
                               # the official Audio2Beats(dbn=True) (verified 8/8 GTZAN songs);
                               # with "clip" it diverged on 1/8 -- the convention is NOT always
                               # event-neutral, though mean F is identical to 4 decimals.
    NATIVE_FPS = 50.0

    def __init__(self, checkpoint: str = "final0", device: str = "cuda",
                 target_fps: Optional[float] = None, float16: bool = False):
        if str(_BEAT_THIS_ROOT) not in sys.path:
            sys.path.insert(0, str(_BEAT_THIS_ROOT))
        from beat_this.inference import Audio2Frames

        self._audio2frames = Audio2Frames(checkpoint_path=checkpoint, device=device,
                                          float16=float16)
        self.checkpoint = checkpoint
        self.device = device
        self.fps = target_fps if target_fps is not None else self.NATIVE_FPS

    @torch.no_grad()
    def activations(self, signal, sample_rate: int) -> torch.Tensor:
        """[num_samples] mono (any sample rate; Beat This resamples internally) -> [num_frames, 2]
        (beat, downbeat) LOGITS at self.fps."""
        beat_logits, downbeat_logits = self._audio2frames(signal, sample_rate)
        logits = torch.stack([beat_logits, downbeat_logits], dim=-1)          # [T@50fps, 2]
        if self.fps != self.NATIVE_FPS:
            num_target = int(round(logits.shape[0] * self.fps / self.NATIVE_FPS))
            logits = torch.nn.functional.interpolate(
                logits.t().unsqueeze(0), size=num_target, mode="linear", align_corners=False
            ).squeeze(0).t()
        return logits.cpu()


if __name__ == "__main__":
    # Smoke test: a synthetic 120 BPM 4/4 click track through frontend -> both decoders.
    import numpy as np

    from frontends import Tracker

    SR, SECONDS, BPM = 22050, 12, 120.0
    signal = np.zeros(SR * SECONDS, dtype=np.float32)
    click = (0.8 * np.sin(2 * np.pi * 1000 * np.arange(0.02 * SR) / SR)
             * np.hanning(int(0.02 * SR))).astype(np.float32)
    beat_samples = (np.arange(0, SECONDS * BPM / 60) * 60 / BPM * SR).astype(int)
    for i, s in enumerate(beat_samples):
        signal[s:s + len(click)] += click * (1.5 if i % 4 == 0 else 1.0)   # accent the downbeats

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frontend = BeatThisFrontend(device=device)
    print(f"frontend: {frontend.name} ckpt={frontend.checkpoint} fps={frontend.fps} "
          f"form={frontend.activation_form}")
    activations = frontend.activations(signal, SR)
    print(f"activations: {tuple(activations.shape)} (expect ~{SECONDS * 50} frames)")

    for decoder in ("madmom_dbn", "bar_pointer_hmm"):
        kwargs = {} if decoder == "madmom_dbn" else {"device": device, "beats_per_bar": 4}
        events = Tracker(frontend, decoder, **kwargs).track(signal, SR)
        n_beats, n_downbeats = len(events["beats"]), len(events["downbeats"])
        ibi = float(np.diff(events["beats"]).mean()) if n_beats > 1 else float("nan")
        print(f"  {decoder:16s}: {n_beats:3d} beats (expect ~{SECONDS * 2}), "
              f"{n_downbeats:2d} downbeats (expect ~{SECONDS // 2}), "
              f"mean IBI {ibi:.3f}s -> {60.0 / ibi:.1f} BPM (expect ~120)")
