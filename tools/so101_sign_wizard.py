"""SO-101 joint-sign wizard — let the hardware answer, one joint at a time.

For each joint, slowly moves the servo +8° IN THE FOLLOWER'S OWN
CONVENTION (no signs applied), tells you what the URDF model expects
that motion to look like, and asks whether that's what physically
happened. y = signs agree (+1), n = mirrored (-1). At the end it prints
the exact `--joint-signs` string for examples/teleop_so101.py.

The wiggle is ±8° at ~4°/s around the current pose. Start with the arm
roughly in the calibration L-pose (upper arm up, forearm forward) so no
joint is near a hard stop, and keep the workspace clear.

Run:
    python tools/so101_sign_wizard.py --port /dev/ttyACM0 --id so101
"""

from __future__ import annotations

import argparse
import time

# lerobot >=0.6 merged the SO followers; SO101Follower is the alias.
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vr_teleop_kit.lerobot.so101_utils import presync_goal_positions  # noqa: E402

# (joint, what MODEL-POSITIVE motion looks like physically)
CHECKS = [
    ("shoulder_pan", "the whole arm swings to the arm's RIGHT (your right when standing behind it)"),
    ("shoulder_lift", "the whole arm tilts DOWN toward the table"),
    ("elbow_flex", "the forearm (and gripper) dips DOWN"),
    ("wrist_flex", "the gripper nose tips DOWN"),
    ("wrist_roll", "the gripper's top (moving-jaw side) rolls toward the arm's LEFT"),
]

STEP_DEG = 8.0
RATE_DEG_S = 4.0
TICK_S = 0.05
# Per-command increment and hard envelope around the start pose. Every
# command is clamped into [start - GUARD, start + GUARD] no matter what
# any position read claims — a glitched read can then never turn into a
# large motion.
CMD_STEP_DEG = RATE_DEG_S * TICK_S   # 0.2°/tick
GUARD_DEG = 12.0


def _read_stable(follower, joint: str, n: int = 5) -> tuple[float, float]:
    """Median of n reads plus their spread. Feetech buses occasionally
    return corrupted positions, and a joint whose encoder sits near the
    12-bit wrap boundary can teleport by ±360° between reads — a single
    read must never be trusted before commanding relative motion."""
    vals = sorted(float(follower.get_observation()[f"{joint}.pos"]) for _ in range(n))
    return vals[n // 2], vals[-1] - vals[0]


def _move(follower, joint: str, start: float, delta: float) -> bool:
    """Slowly ramp one joint start → start+delta → start, commanding an
    absolute trajectory from the vetted start (never re-anchoring on live
    reads). Aborts (returns False) if the joint stops tracking the
    command — the wrap-glitch signature."""
    cmd = start
    for target in (start + delta, start):
        while abs(cmd - target) > 1e-6:
            step = max(-CMD_STEP_DEG, min(CMD_STEP_DEG, target - cmd))
            cmd = max(start - GUARD_DEG, min(start + GUARD_DEG, cmd + step))
            follower.send_action({f"{joint}.pos": cmd})
            time.sleep(TICK_S)
        time.sleep(0.4)
        pos, spread = _read_stable(follower, joint, 3)
        if spread > 5.0 or abs(pos - cmd) > 20.0:
            print(f"  !! {joint} not tracking (cmd={cmd:.1f}°, reads {pos:.1f}° "
                  f"spread {spread:.1f}°) — aborting this joint. Its encoder is "
                  "likely near the wrap boundary here; move the arm to the "
                  "L-pose and retry.")
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="so101")
    args = ap.parse_args()

    # NOTE: max_relative_target is deliberately None. The wizard's own
    # trajectory (0.2°/tick, hard ±12° envelope) is the safety layer. A
    # follower-side relative clamp is actively harmful here: it re-anchors
    # every goal to the current (gravity-sagged) position, so on a loaded
    # joint the goal ratchets downward after the sag and the arm "falls"
    # slowly to its stop while commands trail it — observed on the elbow.
    # Sync stale RAM goals to present positions before torque-enable —
    # otherwise connect() snaps the arm to the previous session's last
    # command at full speed.
    presync_goal_positions(args.port)
    follower = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.id, use_degrees=True, max_relative_target=None,
    ))
    follower.connect()
    print("\nConnected — torque is ON, the arm holds its pose.")
    print("Put the arm roughly in the L-pose first if it isn't (Ctrl-C, use "
          "tools/so101_readout.py to go limp, then rerun this).\n")

    signs: list[str] = []
    try:
        for joint, expect in CHECKS:
            input(f"[{joint}] ENTER to wiggle +8° ... ")
            start, spread = _read_stable(follower, joint)
            if spread > 2.0:
                print(f"  !! {joint} reads are unstable (spread {spread:.1f}°) — "
                      "encoder near its wrap boundary at this pose. Skipping; "
                      "reposition and rerun.\n")
                signs.append("?")
                continue
            if abs(start) > 70.0:
                print(f"  !! {joint} reads {start:.1f}° — that is far from the "
                      "L-pose (expected ≈0°). Not moving it from here; put the "
                      "arm in the L-pose and rerun.\n")
                signs.append("?")
                continue
            if not _move(follower, joint, start, STEP_DEG):
                signs.append("?")
                continue
            while True:
                ans = input(f"  Did {expect}? [y/n] ").strip().lower()
                if ans in ("y", "n"):
                    break
            signs.append("1" if ans == "y" else "-1")
            print(f"  → {joint}: {'+1' if ans == 'y' else '-1'}\n")
    except KeyboardInterrupt:
        print("\naborted")
        follower.disconnect()
        return

    flag = ",".join(signs)
    print("=" * 60)
    if "?" in signs:
        print(f"PARTIAL RESULT (rerun for the '?' joints):  {flag}")
    else:
        print(f"RESULT:  --joint-signs {flag}")
    print("=" * 60)
    print("(disconnecting — torque drops, support the arm)")
    follower.disconnect()


if __name__ == "__main__":
    main()
