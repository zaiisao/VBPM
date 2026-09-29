"""Scoring: the downbeat metrics themselves, and the model-facing evaluation loops."""
from __future__ import annotations

from collections import defaultdict

import mir_eval
import numpy as np
import torch

from ..data.excerpts import collate_excerpts
from ..constants import TWO_PI

TOLERANCE_S = 0.070


def peak_times(probs, fps: float, period_s: float, threshold: float = 0.5):
    """Frames whose probability is a local maximum above ``threshold`` -> times (s)."""
    probs = np.asarray(probs, dtype=np.float64)

    # RELATIVE to the curve's own maximum. An absolute 0.5 was unreachable: the emission
    # a + b cos(phi) tops out near 0.16 at every (a, b) this model has ever learned, so
    # the picker returned [] on every crop and every "emission-D F 0.000" ever reported
    # was a threshold artifact rather than a measurement.
    ceiling = float(probs.max()) if probs.size else 0.0
    if ceiling <= 0.0:
        return np.zeros(0, dtype=np.float64)
    probs = probs / ceiling

    min_gap = max(1, int(round(0.5 * period_s * fps)))
    order = np.argsort(-probs)
    taken: list[int] = []
    for i in order:
        if probs[i] < threshold:
            break
        if all(abs(i - j) >= min_gap for j in taken):
            taken.append(int(i))
    return np.sort(np.asarray(taken, dtype=np.float64)) / fps


def continuity_scores(annotated, predicted):
    """(CMLt, AMLt) from mir_eval's (CMLc, CMLt, AMLc, AMLt)."""
    annotated = np.asarray(annotated, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(predicted) <= 1 or len(annotated) <= 1:
        return 0.0, 0.0
    _cmlc, cmlt, _amlc, amlt = mir_eval.beat.continuity(annotated, predicted)
    return float(cmlt), float(amlt)


def f_measure(predicted, annotated, tolerance: float = TOLERANCE_S):
    """(f, precision, recall) with greedy one-to-one matching inside ``tolerance``."""
    predicted = np.asarray(predicted, dtype=np.float64)
    annotated = np.asarray(annotated, dtype=np.float64)
    if len(annotated) == 0:
        return (1.0, 1.0, 1.0) if len(predicted) == 0 else (0.0, 0.0, 1.0)
    if len(predicted) == 0:
        return 0.0, 1.0, 0.0

    used = np.zeros(len(annotated), dtype=bool)
    hits = 0
    for t in predicted:
        gap = np.abs(annotated - t)
        gap[used] = np.inf
        j = int(np.argmin(gap))
        if gap[j] <= tolerance:
            used[j] = True
            hits += 1

    precision = hits / len(predicted)
    recall = hits / len(annotated)
    f = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
    return f, precision, recall


def trajectory_period(mu, mask, fps):
    """[B, T] phase -> [B] bar period in seconds, read off the model's OWN trajectory."""
    inc = mu[:, 1:] - mu[:, :-1]
    inc = torch.atan2(torch.sin(inc), torch.cos(inc))
    weight = mask[:, 1:] * mask[:, :-1]
    tempo = (inc * weight).sum(1) / weight.sum(1).clamp(min=1.0)
    # a non-advancing (or backward) trajectory has no period; fall back to the window
    # length so the grid degenerates to a single time rather than dividing by zero
    span = mask.sum(1).clamp(min=1.0) / fps
    period = torch.where(tempo > 1e-6, TWO_PI / (tempo.clamp(min=1e-6) * fps), span)
    return period.cpu().numpy()


def trajectory_health(mu, kappa, mask, crops):
    """(advance, kappa, phase_err, coverage): what F cannot see about the trajectory."""
    from ..data.dataset import true_phase

    inc = mu[:, 1:] - mu[:, :-1]
    inc = torch.atan2(torch.sin(inc), torch.cos(inc))
    w = mask[:, 1:] * mask[:, :-1]
    advance = float((inc * w).sum() / w.sum().clamp(min=1.0))
    kappa_mean = float((kappa * mask).sum() / mask.sum().clamp(min=1.0))

    errs, visited = [], np.zeros(16, dtype=bool)
    for i, crop in enumerate(crops):
        t = len(crop["y"])
        phi_true, ok = true_phase(crop)
        if not ok.any():
            continue
        m = mu[i, :t].cpu().numpy()[ok]
        diff = np.angle(np.exp(1j * (m - phi_true[ok])))
        errs.append(np.abs(diff))
        visited |= np.bincount(((m % TWO_PI) / TWO_PI * 16).astype(int) % 16,
                               minlength=16).astype(bool)

    phase_err = float(np.mean(np.concatenate(errs))) if errs else float("nan")
    return advance, kappa_mean, phase_err, float(visited.mean())


def null_times(crop, kind: str, rng):
    """A baseline downbeat sequence with the right RATE but no learned phase."""
    period = crop["bar_period"]
    span = len(crop["y"]) / crop["fps"] if "fps" in crop else None
    duration = span if span is not None else (crop["downbeat_times"][-1] - crop["t0"])
    offset = rng.uniform(0.0, period) if kind == "random" else 0.0
    return crop["t0"] + offset + np.arange(0.0, max(duration, 0.0), period)


def event_times(flags, crossing, raw):
    """Event TIMES per crop from the path's own crossings, interpolated within the frame."""
    return [crossing[i][flags[i]].cpu().numpy() for i in range(len(raw))]


def crop_times(frames, raw):
    """Frame positions -> seconds, dropping events past each crop's last valid frame."""
    return [f[f < len(c["y"]) - 1] / c["fps"] + c["t0"] for f, c in zip(frames, raw)]


def scoring_records(raw) -> list:
    """Collated excerpt batch -> per-item scoring records, trimmed to valid frames."""
    records = []
    for i in range(len(raw["y"])):
        valid = int(raw["mask"][i].sum())
        if valid == 0:
            records.append(None)
            continue
        records.append({"y": raw["y"][i, :valid].numpy(),
                        "fps": float(raw["fps"][i]), "t0": float(raw["t0"][i]),
                        "downbeat_times": np.asarray(raw["downbeat_times"][i]),
                        "beat_times": np.asarray(raw.get("beat_times", [[]] * len(raw["y"]))[i]),
                        "anchors": np.asarray(raw["anchors"][i]),
                        "dataset": raw["dataset"][i], "song_id": raw["song_id"][i]})
    return records


def evaluate(model, dataset, frontend, device, batch_size: int, seed: int = 0):
    """Per-dataset downbeat metrics for both read-outs, beside the nulls."""
    assert dataset.deterministic, "evaluation scores FIXED windows"
    model.eval()
    rows: dict = defaultdict(lambda: defaultdict(list))
    rng = np.random.default_rng(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         collate_fn=collate_excerpts)

    with torch.no_grad():
        for raw in loader:
            records = scoring_records(raw)
            keep = [i for i, c in enumerate(records) if c is not None]
            if not keep:
                continue
            crops = [records[i] for i in keep]
            # the same frontend call training makes; features never touch disk
            h = frontend.forward_features(raw["input"]).clone()
            mask = raw["mask"].to(device, non_blocking=True)

            path = model.infer_path(h, mask)
            mu = path["phi_path"][keep]
            crossing = path["crossing"][keep]
            times = crop_times(event_times(path["is_downbeat"][keep], crossing, crops), crops)
            beats = crop_times(event_times(path["is_beat"][keep], crossing, crops), crops)
            probs = model.emission_probs(h, mask, path)[keep].cpu().numpy()
            beats_per_bar = path["meter_path"] @ model.meter_values
            meter = (beats_per_bar * mask).sum(1) / mask.sum(1).clamp(min=1.0)

            # the peak picker and the nulls need a bar period; take the model's OWN
            # inferred tempo, the only period left in the pipeline now that delta is gone
            period = trajectory_period(mu, mask[keep], float(raw["fps"][0]))
            for i, crop in enumerate(crops):
                crop["bar_period"] = float(period[i])

            for i, crop in enumerate(crops):
                t = len(crop["y"])
                truth = np.asarray(crop["downbeat_times"])
                rule_g = times[i]
                per = rows[crop["dataset"]]
                if len(truth) == 0:
                    if beats is not None and len(crop["beat_times"]):
                        bt = np.asarray(crop["beat_times"])
                        per["beat F"].append(f_measure(beats[i], bt)[0])
                        bc, ba = continuity_scores(bt, beats[i])
                        per["beat CMLt"].append(bc)
                        per["beat AMLt"].append(ba)
                        per["beat est/ref"].append(len(beats[i]) / max(len(bt), 1))
                        per["meter"].append(float(meter[keep][i]))
                    continue
                per["rule-g"].append(f_measure(rule_g, truth)[0])
                cmlt, amlt = continuity_scores(truth, rule_g)
                per["rule-g CMLt"].append(cmlt)
                per["rule-g AMLt"].append(amlt)
                per["est/ref"].append(len(rule_g) / max(len(truth), 1))

                alt_d = peak_times(probs[i, :t], crop["fps"], crop["bar_period"]) + crop["t0"]
                per["emission-D"].append(f_measure(alt_d, truth)[0])

                if beats is not None and len(crop["beat_times"]):
                    bt = np.asarray(crop["beat_times"])
                    per["beat F"].append(f_measure(beats[i], bt)[0])
                    bc, ba = continuity_scores(bt, beats[i])
                    per["beat CMLt"].append(bc)
                    per["beat AMLt"].append(ba)
                    per["beat est/ref"].append(len(beats[i]) / max(len(bt), 1))
                    per["meter"].append(float(meter[keep][i]))

                for kind in ("random", "zero"):
                    per[f"null-{kind}"].append(
                        f_measure(null_times(crop, kind, rng), truth)[0])

    return {ds: {k: (float(np.mean(v)), len(v)) for k, v in per_mode.items()}
            for ds, per_mode in rows.items()}


def print_table(results):
    """One row per (split, dataset, mode). One run = one seed; sweeps aggregate outside."""
    print("\n==== downbeat F (+-70 ms) ====")
    rows = sorted((split, dataset, mode, value, count)
                  for split, per_dataset in results.items()
                  for dataset, modes in per_dataset.items()
                  for mode, (value, count) in modes.items())
    units = {"est/ref": "ratio", "rule-g CMLt": "CMLt", "rule-g AMLt": "AMLt"}
    for split, dataset, mode, value, count in rows:
        print(f"  {split:6s} {mode:15s} {dataset:11s} "
              f"{units.get(mode, 'F'):5s} {value:.3f}  (n={count})")
