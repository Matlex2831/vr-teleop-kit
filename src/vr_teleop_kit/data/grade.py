"""Per-episode quality heuristics for a recorded LeRobotDataset.

Extracted from `tools/so101_dataset_report.py` so the CLI report and the
web review page grade episodes with exactly one implementation — a
verdict that disagreed with itself depending on where you read it would
be worse than no verdict at all.

The checks all run off the recorded arrays; no video decoding is needed:

  grasp        the gripper must CLOSE and then REOPEN. A pick-and-place
               that never closes the jaws did not pick anything up; a
               run of several close/open cycles is a struggle with
               retries.
  motion       total joint sweep. Near zero means the operator never
               engaged the clutch and the arm sat at rest the whole
               episode.
  tracking     mean |action - state|. This separates OPERATOR failures
               from HARDWARE failures: if commands were issued and the
               arm did not follow, the servos are the problem, not the
               demonstration.
  share        fraction of the whole dataset this episode occupies. One
               long fumbling episode can dominate a small dataset and
               drag the policy toward imitating the fumbling.

Verdicts are heuristics tuned for a single pick-and-place task; they
flag episodes worth LOOKING at, they do not overrule your eyes.
"""

from __future__ import annotations

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_IDX = 5

# Gripper is 0..100 with 100 = fully open. "Closed enough to be holding
# something" and "open enough to have let go" — deliberately far apart so
# jitter around a threshold cannot manufacture a grasp.
GRIP_CLOSED_BELOW = 40.0
GRIP_OPEN_ABOVE = 70.0

# A joint sweep below this (degrees, summed over the arm) means the arm
# effectively never moved.
STILL_TOTAL_DEG = 5.0

# Mean |action - state| above this means the arm did not follow commands.
TRACKING_FAIL_DEG = 5.0

# An episode occupying more than this fraction of the dataset is called
# out — not wrong, but it dominates training.
DOMINANT_SHARE = 0.30

VERDICTS = ("PASS", "SUSPECT", "FAIL")


def analyse(sub, fps: int, total_frames: int) -> dict:
    """Grade one episode.

    `sub` is the slice of the data parquet belonging to a single
    episode (a DataFrame with `observation.state` and `action` columns
    of per-frame arrays). Returns the measurements plus a `verdict` /
    `why` pair.
    """
    state = np.stack(sub["observation.state"].to_numpy())
    action = np.stack(sub["action"].to_numpy())
    grip = state[:, GRIPPER_IDX]

    closed = grip < GRIP_CLOSED_BELOW
    # Rising edges of "closed" = number of distinct grasp attempts.
    closes = int(np.sum(np.diff(closed.astype(int)) == 1)) + int(bool(closed[0]))
    reopened = bool(closed.any() and grip[int(np.argmax(closed)):].max() > GRIP_OPEN_ABOVE)

    sweep = state[:, :5].max(0) - state[:, :5].min(0)
    tracking = float(np.abs(action[:, :5] - state[:, :5]).mean())

    n = len(sub)
    r = {
        "frames": n,
        "seconds": n / fps,
        "share": n / total_frames if total_frames else 0.0,
        "grip_min": float(grip.min()),
        "grip_max": float(grip.max()),
        "closes": closes,
        "reopened": reopened,
        "sweep": sweep,
        "sweep_total": float(sweep.sum()),
        "tracking": tracking,
    }

    # Order matters: report the most fundamental failure first.
    if r["sweep_total"] < STILL_TOTAL_DEG:
        r["verdict"], r["why"] = "FAIL", "arm never moved — clutch not engaged"
    elif tracking > TRACKING_FAIL_DEG:
        r["verdict"], r["why"] = "FAIL", f"arm did not follow commands ({tracking:.1f}° error) — check the servos"
    elif closes == 0:
        r["verdict"], r["why"] = "FAIL", "gripper never closed — nothing was picked up"
    elif not reopened:
        r["verdict"], r["why"] = "SUSPECT", "gripper closed but never reopened — object not released"
    elif closes > 1:
        r["verdict"], r["why"] = "SUSPECT", f"{closes} grasp attempts — retries, likely a struggle"
    else:
        r["verdict"], r["why"] = "PASS", "single clean grasp and release"

    if r["verdict"] == "PASS" and r["share"] > DOMINANT_SHARE:
        r["why"] += f" (but {r['share']*100:.0f}% of the dataset — long)"
    return r
