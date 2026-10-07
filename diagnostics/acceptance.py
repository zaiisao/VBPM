"""Apply the existing 70ms physical/prior/peak timing review separately.

Original argmax classifications and reports remain unchanged. This review
does not claim calibrated frame classification or unseen-audio performance.
"""

import argparse
import json
from pathlib import Path


def main():
    """Run the command-line tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    target = args.model_dir / "acceptance_70ms.json"
    if target.exists():
        parser.error("Preserve the existing acceptance review")
    training = json.loads((args.model_dir / "report.json").read_text())
    readouts = json.loads((args.model_dir / "readout_review.json").read_text())
    assert training["status"] == "complete"
    assert Path(training["batch_cache"]).resolve() == Path(readouts["batch_cache"]).resolve()
    assert training["songs"] == readouts["songs"]
    assert len(training["results"]) == len(readouts["results"]) == 2
    results = []
    for endpoint, audit in zip(training["results"], readouts["results"]):
        assert endpoint["seed"] == audit["seed"]
        scores = dict(endpoint["scores"])
        original = scores["joint_gate_passed"]
        timing = all(
            audit[key][event] >= 0.85
            for key in ("latent_mean", "decoder_relative_peaks", "latent_samples")
            for event in ("beat_F1_70ms", "downbeat_F1_70ms")
        )
        assert audit["latent_samples"]["draws"] >= 64
        passed = bool(
            scores["physical_gate_passed"]
            and timing
            and audit["latent_samples"]["nonpositive_velocity_fraction"] == 0
        )
        scores["original_argmax_joint_gate_passed"] = original
        scores["joint_gate_passed"] = passed
        results.append(
            dict(
                seed=endpoint["seed"],
                scores=scores,
                latent_mean=audit["latent_mean"],
                decoder_relative_peaks=audit["decoder_relative_peaks"],
                latent_samples=audit["latent_samples"],
            )
        )
    report = dict(
        status="complete",
        scope=training["scope"],
        policy=(
            "Existing separate timing review: phase p95 <=70ms, observed inter"
            "beat tempo RMSE <=3 BPM, means 20..300 BPM; physical mean, 64 pri"
            "or draws and prediction-only relative decoder peaks each beat/dow"
            "nbeat F1 >=0.85; no nonpositive sampled velocity"
        ),
        limitation=(
            "Original argmax classifications unchanged; this is event timing, "
            "not frame confidence calibration or generalization"
        ),
        source_report=str(args.model_dir / "report.json"),
        source_readout=str(args.model_dir / "readout_review.json"),
        results=results,
    )
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=[r["scores"]["joint_gate_passed"] for r in results])))


if __name__ == "__main__":
    main()
