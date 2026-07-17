"""CLI inference: python track.py song.wav [--frontend beat_this] [--bar-pointer 2016_dbn].

Prints one event per line ("<seconds>  beat" / "<seconds>  DOWNBEAT"). This is deliberately just
argument parsing around tracker.py's definitions -- training and evaluation build their Trackers
from the same registries; this script is only the run-it-on-a-file entry point.
"""
import argparse

import numpy as np
import soundfile

from tracker import BAR_POINTERS, Tracker, build_frontend


def main():
    parser = argparse.ArgumentParser(description="Track beats/downbeats: frontend x bar-pointer model.")
    parser.add_argument("audio", help="path to an audio file (anything soundfile reads)")
    parser.add_argument("--frontend", default="beat_this", help="frontend name (build_frontend)")
    parser.add_argument("--checkpoint", default="final0",
                        help="frontend checkpoint (final0 is NOT fold-honest on training datasets)")
    parser.add_argument("--bar-pointer", default="2016_dbn", choices=sorted(BAR_POINTERS))
    parser.add_argument("--shipped", action="store_true",
                        help="madmom's shipped decode options (num_tempi=60, threshold=0.05, "
                             "correct=True) instead of the bare model (2016_dbn only; "
                             "madmom_dbn always runs as shipped)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_kwargs = {}
    if args.bar_pointer != "madmom_dbn":
        model_kwargs["device"] = args.device
        if args.shipped:
            model_kwargs.update(num_tempi=60, threshold=0.05, correct=True)

    frontend = build_frontend(args.frontend, checkpoint=args.checkpoint, device=args.device)
    tracker = Tracker(frontend, args.bar_pointer, **model_kwargs)

    signal, sample_rate = soundfile.read(args.audio, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    events = tracker.track(signal, sample_rate)

    downbeats = set(np.round(events["downbeats"], 6))
    for t in events["beats"]:
        print(f"{t:10.3f}  {'DOWNBEAT' if round(float(t), 6) in downbeats else 'beat'}")


if __name__ == "__main__":
    main()
