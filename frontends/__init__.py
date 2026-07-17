"""Feature extractors (frontends): audio -> [num_frames, 2] (beat, downbeat) activations.

One script per frontend (beat_this.py, later mert.py, ...). A frontend wraps the official upstream
model behind a two-property interface -- WHAT it emits (`fps`, `activation_form`) and HOW to get it
(`activations(signal, sample_rate) -> [num_frames, 2]`). Selecting a frontend by name
(build_frontend) and pairing it with a bar-pointer model is tracker.py's job, one level up -- this
package only defines the interface and its implementations.

Deliberately simple (a resurrected, slimmed version of the archived
data/feature_extractor.py + configs/frontends/*.yaml system): properties live on the wrapper class,
not in YAML, until we have enough frontends to need config files again.
"""


class Frontend:
    """Interface. A frontend turns audio into [num_frames, 2] (beat, downbeat) activations."""

    name: str = "?"
    fps: float = None               # the activation frame rate -- bar-pointer models build on this
    activation_form: str = "prob"   # "prob" or "logit" -- what activations() returns
    bounding: str = "clip"          # the frontend's PUBLISHED bounding convention (see rungs/r0);
                                    # wired into the DBN so our pipeline == the published one

    def activations(self, signal, sample_rate: int):
        """[num_samples] mono audio -> [num_frames, 2] (beat, downbeat) in `activation_form`."""
        raise NotImplementedError
