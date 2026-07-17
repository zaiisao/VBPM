"""MAP state path -> musical events. Shared by R1-R4 (the read-off must be identical across rungs,
or the ladder stops being a controlled comparison).
"""
import numpy as np

from rungs.bar_pointer.state_space import BEAT, DOWNBEAT


def state_path_to_events(state_path, state_space, fps: float, snap_to_activations=None,
                         first_frame: int = 0) -> dict:
    """state_path: [num_frames] state indices from Viterbi. Returns {'beats': seconds,
    'downbeats': seconds}.

    A beat is the ENTRY into any beat region (position class BEAT or DOWNBEAT); a downbeat is the
    entry into the downbeat region. The downbeat is also a beat, so it appears in both lists --
    matching R0/madmom, whose joint DBN likewise emits the downbeat as a beat with position 1.

    Because bar position only ever increases, entering a beat region is exactly the frame where the
    integer beat counter ticks over -- which is how madmom reads its own path off (its `correct=False`
    branch, `np.diff(positions.astype(int))`). Verified equal: R1 and a matched madmom score
    identically to 4 decimals.

    snap_to_activations: pass the [num_frames, 2] activations the model saw to instead report each
    beat at the strongest activation frame WITHIN its beat region (madmom's `correct=True`,
    behavior-copied down to the flat argmax over both columns). A deployment lesson, not the model:
    the Viterbi region entry can sit a frame or two off the perceptual onset, and on soft-onset
    material (classical piano) snapping to the evidence peak is worth ~+0.014 beat F; on tight
    pop/electronic it does nothing or slightly hurts.

    first_frame: offset added to every frame index before conversion to seconds -- pass the crop
    offset when the activations were threshold-cropped (rungs/deployment.py).
    """
    position_classes = state_space.position_classes[np.asarray(state_path)]
    is_in_beat_region = position_classes >= BEAT

    if snap_to_activations is None:
        is_in_downbeat_region = position_classes == DOWNBEAT
        entered_beat_region = is_in_beat_region & ~np.concatenate([[False], is_in_beat_region[:-1]])
        entered_downbeat_region = (is_in_downbeat_region
                                   & ~np.concatenate([[False], is_in_downbeat_region[:-1]]))
        beat_frames = np.where(entered_beat_region)[0]
        downbeat_frames = np.where(entered_downbeat_region)[0]
    else:
        # madmom's correct=True: for each contiguous beat region of the path, report the frame with
        # the strongest activation (flat argmax over both columns, hence the // 2).
        edges = np.nonzero(np.diff(is_in_beat_region.astype(int)))[0] + 1
        if is_in_beat_region[0]:
            edges = np.r_[0, edges]
        if is_in_beat_region[-1]:
            edges = np.r_[edges, len(is_in_beat_region)]
        beat_frames = np.array([
            int(np.argmax(snap_to_activations[left:right]) // 2 + left)
            for left, right in edges.reshape(-1, 2)], dtype=np.int64)
        beat_positions = state_space.state_positions[np.asarray(state_path)[beat_frames]]
        downbeat_frames = beat_frames[beat_positions.astype(int) == 0]

    return {"beats": (beat_frames + first_frame) / fps,
            "downbeats": (downbeat_frames + first_frame) / fps}
