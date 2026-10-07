"""Physical and event scoring shared by audio generator experiments."""

import math

import torch

from diagnostics.real_event_references import interbeat_tempo_errors
from diagnostics.synthetic_ladder import frame_score as synthetic_score


@torch.no_grad()
def score(model, data):
    # Keep the same physical/event metrics, with real-audio references used
    # only here. Real annotation interpolation does not identify the actual
    # instantaneous tempo between beats, so its correlation is diagnostic.
    """Score physical trajectories and beat/downbeat event accuracy."""
    result = synthetic_score(model, data)
    phase, velocity = model.trajectory(data["x"])
    bpm = velocity * 4 * 50 * 60 / (2 * math.pi)
    error = phase - data["phase"]
    error = torch.atan2(error.sin(), error.cos()).abs()
    advance = torch.cat((velocity, velocity[:, -1:]), -1).clamp_min(1e-6)
    time_error_ms = error / advance / 50 * 1000
    result["phase_time_MAE_ms"] = float(time_error_ms.mean())
    result["phase_time_p95_ms"] = float(torch.quantile(time_error_ms, 0.95))
    result["angular_3deg_diagnostic_passed"] = result["raw_phase_MAE_deg"] <= 3
    if "beat_times" in data:
        result.update(interbeat_tempo_errors(phase, data["beat_times"]))
    result["minimum_BPM"] = float(bpm.min())
    result["maximum_BPM"] = float(bpm.max())
    observed_tempo_error = result.get("interbeat_tempo_RMSE_BPM", result["tempo_RMSE_BPM"])
    physical = (
        result["phase_time_p95_ms"] <= 70
        and observed_tempo_error is not None
        and observed_tempo_error <= 3
        and bool((bpm >= 20).all() and (bpm <= 300).all())
    )
    result["physical_gate_passed"] = physical
    result["joint_gate_passed"] = (
        physical
        and result["beat_event_F1_70ms"] >= 0.85
        and result["downbeat_event_F1_70ms"] >= 0.85
    )
    return result
