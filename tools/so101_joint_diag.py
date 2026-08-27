"""SO-101 single-joint diagnostic — is the servo commanded wrong, or failing?

Three phases on one joint (default elbow_flex):

  1. Register dump — torque state, operating mode, PID gains, torque and
     protection limits, temperature, voltage, current, position limits.
     A servo that "falls with torque on" usually confesses here
     (Operating_Mode != 0, Torque_Enable 0, Max_Torque_Limit tiny,
     temperature at the protection threshold, ...).
  2. Hold test — goal := current position, you gently let go. If the
     joint holds (droop < ~3°) the servo is fine and any dive during
     teleop is a COMMANDED motion (pipeline side). If it sinks or
     free-falls, the servo itself can't hold (overload cutoff, stripped
     gears, mode).
  3. Step test (only if the hold passed) — ±12° tracking against and
     with gravity, sampled.

Support the arm in the L-pose before starting: torque locks on connect.

Run:
    python tools/so101_joint_diag.py --port /dev/ttyACM0 --id so101 [--joint elbow_flex]
"""

from __future__ import annotations

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vr_teleop_kit.lerobot.so101_utils import presync_goal_positions  # noqa: E402

DUMP_REGS = [
    ("Torque_Enable", "1 = on"),
    ("Operating_Mode", "0 = position servo; anything else is wrong here"),
    ("P_Coefficient", "expect 32"),
    ("I_Coefficient", "expect 0"),
    ("D_Coefficient", "expect 32"),
    ("Max_Torque_Limit", "counts of full-scale torque, expect ~1000"),
    ("Protection_Current", "overcurrent protection threshold"),
    ("Protective_Torque", "torque fraction after protection trips"),
    ("Protection_Time", "ms of overload before protection trips"),
    ("Overload_Torque", "load threshold that counts as overload"),
    ("Max_Temperature_Limit", "°C protection threshold"),
    ("Present_Temperature", "°C now"),
    ("Present_Voltage", "0.1 V units"),
    ("Present_Current", "6.5 mA units"),
    ("Present_Load", "signed load, 0.1% units"),
    ("Min_Position_Limit", "servo-side clamp (counts)"),
    ("Max_Position_Limit", "servo-side clamp (counts)"),
    ("Homing_Offset", "counts"),
    ("Goal_Position", "counts (raw)"),
    ("Present_Position", "counts (raw)"),
]


def read_deg(follower, joint: str) -> float:
    return float(follower.bus.read("Present_Position", joint))


def sample(follower, joint: str, seconds: float, label: str) -> tuple[float, float]:
    print(f"  t(s)   {joint}°    load    current   temp   ({label})")
    t0 = time.perf_counter()
    first = last = read_deg(follower, joint)
    while True:
        t = time.perf_counter() - t0
        pos = read_deg(follower, joint)
        load = follower.bus.read("Present_Load", joint, normalize=False)
        cur = follower.bus.read("Present_Current", joint, normalize=False)
        temp = follower.bus.read("Present_Temperature", joint, normalize=False)
        print(f"  {t:4.1f}  {pos:7.1f}  {load:6.0f}  {cur:8.0f}  {temp:4.0f}")
        last = pos
        if t >= seconds:
            return first, last
        time.sleep(0.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="so101")
    ap.add_argument("--joint", default="elbow_flex")
    args = ap.parse_args()
    joint = args.joint

    input(f"Support the arm in the L-pose (torque locks on connect). ENTER to connect... ")
    # Sync each servo's stale RAM Goal_Position to its present position
    # BEFORE torque-enable, so connect() locks the arm where it is
    # instead of snapping to the previous session's last command.
    presync_goal_positions(args.port)
    follower = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.id, use_degrees=True,
        max_relative_target=None, position_p_coefficient=32,
    ))
    follower.connect()

    print(f"\n── phase 1: {joint} register dump ─────────────────────────")
    for reg, note in DUMP_REGS:
        try:
            v = follower.bus.read(reg, joint, normalize=False)
            print(f"  {reg:24s} = {v:<8}  ({note})")
        except Exception as e:
            print(f"  {reg:24s} read failed: {type(e).__name__}")
    print(f"  Present_Position (deg)   = {read_deg(follower, joint):.1f}")

    print(f"\n── phase 2: hold test ─────────────────────────────────────")
    start = read_deg(follower, joint)
    follower.send_action({f"{joint}.pos": start})
    input(f"Goal pinned at {start:.1f}°. GENTLY let go of the arm, then ENTER... ")
    first, last = sample(follower, joint, 6.0, "hands off — watching droop")
    droop = last - first
    print(f"\n  droop over 6 s: {droop:+.1f}°")
    if abs(droop) < 3.0:
        print("  → HOLDS. The servo is fine; a dive during teleop is a COMMANDED motion.")
    elif abs(droop) < 15.0:
        print("  → SAGS. Weak but working — check temperature/protection values above.")
    else:
        print("  → FALLS with torque on. Servo-side problem: protection tripped "
              "(temp/current above), Operating_Mode wrong, or stripped gears.")
        print("(skipping step test; disconnecting — support the arm)")
        follower.disconnect()
        return

    print(f"\n── phase 3: step test ±12° ────────────────────────────────")
    for delta, label in ((-12.0, "UP against gravity"), (+12.0, "DOWN with gravity"), (0.0, "back to start")):
        target = start + delta
        follower.send_action({f"{joint}.pos": target})
        time.sleep(1.2)
        pos = read_deg(follower, joint)
        print(f"  goal {target:7.1f}°  reached {pos:7.1f}°  (err {pos - target:+5.1f}°)  [{label}]")

    print("\ndone (disconnecting — torque drops, support the arm)")
    follower.disconnect()


if __name__ == "__main__":
    main()
