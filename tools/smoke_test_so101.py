"""SO101QuestTeleoperator smoke test — fake Quest, no hardware.

Spins up a fake Quest as a WS client that sends synthetic xr_frame
messages through the relay server, while the SO101QuestTeleoperator
subscribes from the other side. Verifies:

  - teleop.connect() blocks until the WS is open
  - get_action() returns the SO-101 action schema (degrees + 0..100
    gripper) with the rest pose first
  - clutch ENGAGE on the right grip captures origin, moving the
    controller drives the joint values
  - trigger pull closes the gripper (gripper.pos drops from 100)
  - no clutch → no motion

Prereqs: relay running locally (vr-teleop-relay), lerobot installed,
SO101_URDF pointing at the URDF.

Run:
    python tools/smoke_test_so101.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import threading
import time

import websockets

from vr_teleop_kit.lerobot.so101_quest_teleop import (
    SO101QuestTeleoperator,
    SO101QuestTeleoperatorConfig,
)

# Optional override: python tools/smoke_test_so101.py ws://127.0.0.1:8443/ws
WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8443/ws"

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _frame(right_pos, right_grip, right_trigger, viewer_yaw_deg=0.0):
    def buttons(grip, trig):
        b = [{"p": False, "v": 0.0}] * 7
        b[0] = {"p": trig > 0.5, "v": float(trig)}
        b[1] = {"p": bool(grip), "v": 1.0 if grip else 0.0}
        return b

    yaw = math.radians(viewer_yaw_deg)
    return {
        "type": "xr_frame",
        "t_client": time.time(),
        "controllers": {
            "right": {
                "position": list(right_pos),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "buttons": buttons(right_grip, right_trigger),
                "axes": [],
            },
            "left": {
                "position": [-0.2, 1.4, -0.3],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "buttons": buttons(False, 0.0),
                "axes": [],
            },
        },
        "viewer": {
            "position": [0.0, 1.5, 0.0],
            "orientation": [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)],
        },
    }


async def fake_quest_send(scenario):
    async with websockets.connect(WS_URL) as ws:
        for delay, frame in scenario:
            await asyncio.sleep(delay)
            await ws.send(json.dumps(frame))


def run_scenario_in_thread(scenario):
    def worker():
        asyncio.run(fake_quest_send(scenario))
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def fmt_action(a):
    return [round(a[f"{n}.pos"], 2) for n in JOINT_NAMES]


def main() -> None:
    print(f"connecting SO101QuestTeleoperator to {WS_URL}")
    teleop = SO101QuestTeleoperator(
        SO101QuestTeleoperatorConfig(id="smoke", ws_url=WS_URL, publish_ik_state=False)
    )
    teleop.connect()
    assert teleop.is_connected, "teleop failed to connect"
    print("connected ✓")

    a0 = teleop.get_action()
    expected_keys = {f"{n}.pos" for n in JOINT_NAMES} | {"gripper.pos"}
    assert set(a0.keys()) == expected_keys, f"unexpected action keys: {sorted(a0)}"
    print(f"\n[before xr_frame] joints (deg): {fmt_action(a0)}  gripper={a0['gripper.pos']:.1f}")
    assert a0["gripper.pos"] == 100.0, "gripper should start open"

    # Scenario: idle → engage → move +X then up → close gripper → release.
    scenario = [
        (0.30, _frame(right_pos=[0.0, 1.4, -0.3], right_grip=False, right_trigger=0.0)),
        (0.30, _frame(right_pos=[0.0, 1.4, -0.3], right_grip=True,  right_trigger=0.0)),  # engage
        (0.30, _frame(right_pos=[0.04, 1.42, -0.3], right_grip=True, right_trigger=0.0)),
        (0.30, _frame(right_pos=[0.08, 1.45, -0.3], right_grip=True, right_trigger=0.5)),  # half-close
        (0.30, _frame(right_pos=[0.08, 1.45, -0.3], right_grip=True, right_trigger=1.0)),  # full-close
        (0.30, _frame(right_pos=[0.08, 1.45, -0.3], right_grip=False, right_trigger=0.0)), # release
    ]
    run_scenario_in_thread(scenario)

    print("\nsampling get_action() at 50 Hz for 2.5 s while fake Quest plays:")
    samples = []
    t0 = time.time()
    while time.time() - t0 < 2.5:
        a = teleop.get_action()
        samples.append((time.time() - t0, a))
        time.sleep(0.02)

    print()
    print(f"{'t(s)':>6}  joints[pan,lift,elbow,wflex,wroll] (deg)   gripper   hint")
    last_joints = None
    for t, a in samples[::10]:
        joints = fmt_action(a)
        hint = "← qpos changed" if joints != last_joints else ""
        last_joints = joints
        print(f"{t:6.2f}  {joints}  {a['gripper.pos']:6.1f}  {hint}")

    final = samples[-1][1]
    print()
    assert any(abs(final[f"{n}.pos"] - a0[f"{n}.pos"]) > 0.05 for n in JOINT_NAMES), \
        "arm did not move from rest — IK pipeline broken"
    print("✓ arm joints moved from rest pose during engagement")
    grippers = [a["gripper.pos"] for _, a in samples]
    assert min(grippers) < 60.0, f"gripper did not close (min={min(grippers):.1f})"
    print(f"✓ gripper responded to trigger (min value: {min(grippers):.1f})")
    # Joint values must stay inside the URDF limits (in degrees).
    solver = teleop._arm["solver"]
    import numpy as np
    for j, n in enumerate(JOINT_NAMES):
        lo, hi = np.degrees(solver.joint_limits[j])
        vals = [a[f"{n}.pos"] for _, a in samples]
        assert min(vals) >= lo - 1e-6 and max(vals) <= hi + 1e-6, \
            f"{n} left its limits: [{min(vals):.1f}, {max(vals):.1f}] vs [{lo:.1f}, {hi:.1f}]"
    print("✓ all joint commands within URDF limits")

    teleop.disconnect()
    print("\ndone, teleop disconnected.")


if __name__ == "__main__":
    main()
