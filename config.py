"""Central configuration for VBPM (the Variational Bar Pointer Model) and its ablations.

The DEFAULTS here ARE the VBPM model: the generative bar-pointer Dynamical VAE derived from the
ELBO in the "ELBO for DBN" paper. Every field whose name begins with ``divergence_`` turns ON a
deliberate departure (an ablation) from that model -- these exist so we can measure, in controlled
isolation, why a given VBPM design choice matters. With all ``divergence_*`` flags at their
defaults you get the VBPM model exactly.

Run ``python train.py --help`` for the command-line surface (auto-generated from this dataclass).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields


# Frame rate of the cached frontend features: 22050 Hz audio / 256-sample hop = 86.1328125 fps.
FRAMES_PER_SECOND = 22050.0 / 256.0


@dataclass
class Config:
    # ---- data ----
    train_feature_dir: str = "cache/acts/bt_train_rich"   # cached frontend features (training split)
    val_feature_dir: str = "cache/acts/bt_val_rich"       # cached frontend features (validation split)
    num_train_songs: int = 400
    num_val_songs: int = 40
    feature_dim: int = 512                                 # dimensionality of one frame of frontend features
    crop_length_frames: int = 256                          # training crop length (~3 s)
    batch_size: int = 16

    # ---- model (the VBPM bar-pointer DVAE) ----
    hidden_size: int = 64                                  # encoder/decoder hidden width
    num_meters: int = 4                                    # number of meter (beats-per-bar) hypotheses
    beats_per_bar: int = 4                                 # M: beats per bar used by the geometric read-out

    # ---- optimization ----
    num_steps: int = 1000
    learning_rate: float = 1e-3
    grad_clip_norm: float = 5.0
    gumbel_temperature_start: float = 1.0                  # meter Gumbel-softmax temperature, annealed...
    gumbel_temperature_end: float = 0.3                    # ...linearly to this by the final step
    seed: int = 0

    # ---- evaluation ----
    eval_max_frames: int = 1600                            # cap on frames per song at eval time
    eval_beat_tolerance_seconds: float = 0.07             # mir_eval F-measure tolerance window

    # =====================================================================================
    # ABLATIONS: deliberate departures from the VBPM model. Each defaults to the VBPM setting.
    # Each one corresponds to an experiment that probes why a VBPM design choice matters.
    # =====================================================================================

    # Phase posterior update rule:
    #   "free"       -> posterior phase is read directly from the audio each frame (default).
    #   "integrator" -> phase is the deterministic integral of the tempo latent (phi_t = phi_{t-1}+exp(tempo)).
    #   "filter"     -> Kalman-style: predict via tempo, then correct toward the audio with a learned gain.
    divergence_phase_update: str = "free"

    # Tempo latent source:
    #   "latent"   -> tempo is a free latent inferred by the encoder and regularized by the ELBO (default).
    #   "autocorr" -> tempo is COMPUTED by a differentiable autocorrelation head on the features.
    divergence_tempo_source: str = "latent"

    # Training-time likelihood (decoder); inference ALWAYS uses the geometric read-out from z.
    #   "mlp"       -> learned MLP Bernoulli decoder p(beat,downbeat | z) (the VBPM training likelihood).
    #   "geometric" -> geometric emission (beat ~ cos(M*phi), downbeat ~ cos(phi)).
    divergence_decoder: str = "mlp"

    # Auxiliary sawtooth phase supervision (Oyama 2021 / Chen & Su 2022). 0.0 = off (default).
    # When > 0, adds lambda * (1 - cos(phi - phi_ground_truth_sawtooth)) to the loss.
    divergence_sawtooth_weight: float = 0.0

    # Free-bits floor on each KL term, in nats-per-frame. 0.0 = strict ELBO (default).
    divergence_free_bits: float = 0.0

    # Positive-class weights for the reconstruction BCE. (1.0, 1.0) = unweighted Bernoulli (default).
    divergence_beat_pos_weight: float = 1.0
    divergence_downbeat_pos_weight: float = 1.0

    # Probability of hiding the beats from the encoder during training (a posterior regularizer).
    # 0.0 = the posterior always sees the data (default).
    divergence_beat_dropout: float = 0.0

    # Meter handling: "latent" = inferred Categorical meter (default); "fixed" = hard-set to beats_per_bar.
    divergence_meter: str = "latent"

    # Unfreeze the feature extractor and train end-to-end. False = frozen cached features (what we run).
    # (NOTE: end-to-end requires the audio pipeline and is not exercised by the cached-feature path.)
    divergence_end_to_end: bool = False

    # ---- misc ----
    save_path: str = ""                                    # if set, save the trained model here
    device: str = "cuda"

    def gumbel_temperature(self, step: int) -> float:
        """Linearly annealed Gumbel-softmax temperature for the meter relaxation."""
        progress = min(step / max(self.num_steps, 1), 1.0)
        return self.gumbel_temperature_start + (self.gumbel_temperature_end - self.gumbel_temperature_start) * progress

    @property
    def is_default_vbpm(self) -> bool:
        """True iff every divergence is at its default (used for logging / sanity)."""
        return (
            self.divergence_phase_update == "free"
            and self.divergence_tempo_source == "latent"
            and self.divergence_decoder == "mlp"
            and self.divergence_sawtooth_weight == 0.0
            and self.divergence_free_bits == 0.0
            and self.divergence_beat_pos_weight == 1.0
            and self.divergence_downbeat_pos_weight == 1.0
            and self.divergence_beat_dropout == 0.0
            and self.divergence_meter == "latent"
            and not self.divergence_end_to_end
        )


def parse_config() -> Config:
    """Build a Config from command-line arguments (one flag per dataclass field)."""
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for field in fields(Config):
        default_value = getattr(defaults, field.name)
        flag = "--" + field.name
        if field.type == "bool" or isinstance(default_value, bool):
            parser.add_argument(flag, action="store_true" if not default_value else "store_false")
        else:
            parser.add_argument(flag, type=type(default_value), default=default_value)
    namespace = parser.parse_args()
    return Config(**vars(namespace))
