"""SO-101 live joint readout — verify calibration before teleoperating.

Connects to the Feetech bus, DISABLES torque (the arm goes limp — hold
it!), and streams normalized joint positions (degrees, gripper 0..100)
at ~10 Hz. Use it to check that the calibration matches the URDF
convention the IK uses:

  1. Hold the arm in the middle pose — an L-shape: shoulder_pan
     centered (arm straight ahead), upper arm VERTICAL, elbow bent ~90°
     so the forearm and gripper point straight FORWARD horizontally,
     wrist straight, gripper not rolled (jaws opening vertically,
     moving jaw on top). Every joint should read ≈ 0°.
  2. Move one joint at a time and check the SIGN:
       shoulder_pan   +  →  arm swings to its RIGHT (seen from behind)
       shoulder_lift  +  →  arm pitches DOWN
       elbow_flex     +  →  forearm folds DOWN
       wrist_flex     +  →  gripper pitches DOWN
       wrist_roll     +  →  gripper rolls (either way; note which)
       gripper           →  0 = closed, 100 = open
  3. Values should stay within the URDF limits printed in the header.
     A joint that reads far outside them (e.g. ±150°) means the
     calibration's zero is off — recalibrate holding the middle pose
     at the first ENTER prompt.

Run:
    python tools/so101_readout.py --port /dev/ttyACM0 --id so101
Stop with Ctrl-C. Torque stays off on exit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# URDF joint limits (rad→deg), from so101_new_calib.urdf. Printed for
# reference; a reading far outside these means a broken calibration zero.
URDF_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="so101",
                    help="robot id whose calibration file to load")
    args = ap.parse_args()

    calib_path = (
        Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
        / f"{args.id}.json"
    )
    if not calib_path.is_file():
        raise SystemExit(f"no calibration at {calib_path} — run lerobot-calibrate first")
    raw = json.loads(calib_path.read_text())
    calibration = {k: MotorCalibration(**v) for k, v in raw.items()}

    motors = {name: Motor(i + 1, "sts3215", MotorNormMode.DEGREES)
              for i, name in enumerate(JOINT_NAMES)}
    motors["gripper"] = Motor(6, "sts3215", MotorNormMode.RANGE_0_100)

    bus = FeetechMotorsBus(args.port, motors, calibration)
    bus.connect()
    print("connected. DISABLING TORQUE — hold the arm, it will go limp. Ctrl-C to stop.\n")
    bus.disable_torque()

    hdr = "  ".join(f"{n[:9]:>9s}" for n in JOINT_NAMES) + f"  {'gripper':>9s}"
    lim = "  ".join(f"{f'{lo:.0f}..{hi:.0f}':>9s}" for lo, hi in
                    (URDF_LIMITS_DEG[n] for n in JOINT_NAMES)) + f"  {'0..100':>9s}"
    print(hdr)
    print(lim)
    print("-" * len(hdr))
    try:
        while True:
            pos = bus.sync_read("Present_Position")
            row = "  ".join(f"{pos[n]:9.1f}" for n in JOINT_NAMES)
            flags = [n for n in JOINT_NAMES
                     if not (URDF_LIMITS_DEG[n][0] - 15 <= pos[n] <= URDF_LIMITS_DEG[n][1] + 15)]
            warn = ("   ⚠ outside URDF range: " + ",".join(flags)) if flags else ""
            print(f"{row}  {pos['gripper']:9.1f}{warn}", end="\r" if not flags else "\n")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped (torque left off).")
    finally:
        try:
            bus.disconnect()  # disable_torque again is a no-op
        except Exception:
            pass


if __name__ == "__main__":
    main()
