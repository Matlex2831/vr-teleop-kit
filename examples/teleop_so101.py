"""VR single-arm teleop — SO-101 live hardware example.

Runs the SO101QuestTeleoperator against a real SO-101 follower,
alongside the relay server (`vr-teleop-relay`) and the WebXR client on
the Quest.

At startup, the teleop's internal qpos is initialised to the rest pose
(SO101_REST_POSE env var, five comma-separated radians; default from
ik/so101_model.py if unset), and `ramp_to_rest()` drives the physical
arm there in linearly-interpolated steps. The operator then squeezes
grip to engage and teleoperates from the same anchor pose.

Pre-requisites:
  - Relay server + transport up (`vr-teleop-relay`; see the README for
    the USB `adb reverse` and LAN HTTPS transports)
  - Quest browser at http://localhost:8443/ (USB) or
    https://<workstation-lan-ip>:8443/ (LAN), in passthrough VR
  - lerobot installed with Feetech support:
    pip install "lerobot[feetech]"
  - The follower calibrated once (interactive, moves through joint
    ranges):
    lerobot-calibrate --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 --robot.id=<your-arm-id>
  - SO101_URDF pointing at Simulation/SO101/so101_new_calib.urdf from
    https://github.com/TheRobotStudio/SO-ARM100 (or pass --urdf-path)

Run:
    python examples/teleop_so101.py --port /dev/ttyACM0 --id my_so101

Camera (optional): export CAM_TOP=/dev/video<realsense-rgb-node> before
starting `vr-teleop-relay` to stream an overview camera into the VR
panel (any v4l2 device works — the RealSense RGB stream included).
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np

from vr_teleop_kit.ik.so101_model import DEFAULT_Q_REST
from vr_teleop_kit.lerobot.so101_quest_teleop import (
    SO101QuestTeleoperator,
    SO101QuestTeleoperatorConfig,
)
from vr_teleop_kit.lerobot.cli import add_ik_cli_args, ik_kwargs_from_args

from lerobot.utils.utils import init_logging
# lerobot ≥0.6 merged the SO-100/101 followers into `so_follower`;
# SO101Follower/SO101FollowerConfig are the compatible aliases.
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from vr_teleop_kit.lerobot.so101_utils import (
    JOINT_NAMES,
    build_rest_action,
    check_calibration_plausible,
    get_observation_median,
    parse_rest_pose_env,
    presync_goal_positions,
    ramp_to_rest,
)

# Re-exported above so `record_so101.py`'s `from teleop_so101 import ...`
# keeps working; the implementations moved into the installed package so
# the web UI's rollout launcher can share them.
__all__ = [
    "JOINT_NAMES", "build_rest_action", "check_calibration_plausible",
    "get_observation_median", "parse_rest_pose_env", "presync_goal_positions",
    "ramp_to_rest",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="follower serial device (prefer stable /dev/serial/by-id/... names)")
    ap.add_argument("--id", default="so101",
                    help="robot id — must match the id used with lerobot-calibrate")
    ap.add_argument("--hand", choices=("left", "right"), default="right",
                    help="which Quest controller drives the arm")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws", help="relay server WS URL")
    ap.add_argument("--freq", type=int, default=60, help="teleop loop rate (Hz)")
    ap.add_argument("--urdf-path", default="",
                    help="SO-101 URDF path (default: SO101_URDF env var)")
    ap.add_argument("--max-relative-target", type=float, default=None,
                    help="follower-side per-command joint delta clamp (degrees). "
                         "Leave unset: it anchors goals to the current (sagging) "
                         "position, starving gravity-loaded joints of P authority "
                         "until they stall or trip overload protection. The "
                         "teleop's per-joint Δq caps and the velocity-limited "
                         "ramp are the real safety layers.")
    ap.add_argument("--rest-duration-s", type=float, default=3.0,
                    help="seconds to ramp the arm to rest pose at startup")
    ap.add_argument("--rest-steps", type=int, default=90,
                    help="number of interpolation steps in the startup ramp")
    ap.add_argument("--skip-calibration-check", action="store_true",
                    help="proceed even if the calibration file fails the URDF "
                         "plausibility check (see check_calibration_plausible)")
    ap.add_argument("--joint-signs", default="1,1,1,1,1",
                    help="per-joint sign between the model (URDF) convention and "
                         "this unit's servo directions, comma-separated +1/-1 for "
                         "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll. "
                         "Determine with tools/so101_readout.py (a limp joint hangs "
                         "DOWN = model-positive for lift/elbow/wrist_flex).")
    add_ik_cli_args(ap)
    args = ap.parse_args()

    init_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    rest = parse_rest_pose_env("SO101_REST_POSE", DEFAULT_Q_REST.tolist())
    signs = [float(s) for s in args.joint_signs.split(",")]
    if len(signs) != 5 or any(s not in (1.0, -1.0) for s in signs):
        raise SystemExit(f"--joint-signs must be 5 comma-separated +1/-1, got {args.joint_signs!r}")
    ik_overrides = ik_kwargs_from_args(args)
    logger.info("======== vr-teleop config (SO-101) ========")
    logger.info("port   : %s  (id=%s)", args.port, args.id)
    logger.info("hand   : %s", args.hand)
    logger.info("rest   : %s rad", [round(x, 3) for x in rest])
    logger.info("signs  : %s", [int(s) for s in signs])
    if ik_overrides:
        logger.info("CLI IK overrides: %s", ik_overrides)
    logger.info("===========================================")

    # position_p_coefficient=32 restores the STS3215 factory gain.
    # lerobot's default of 16 is tuned for leader-follower streaming and
    # is too soft to lift this arm through gravity-heavy configurations:
    # combined with a relative-target clamp (≤N° of position error → ≤N°
    # of P authority) the shoulder/elbow stall, draw sustained near-stall
    # current, and eventually trip the servo's overload protection — at
    # which point torque collapses and the arm free-falls.
    follower = SO101Follower(SO101FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        max_relative_target=args.max_relative_target,
        position_p_coefficient=32,
    ))

    teleop = SO101QuestTeleoperator(SO101QuestTeleoperatorConfig(
        id="vr-teleop-so101",
        ws_url=args.ws_url,
        hand=args.hand,
        urdf_path=args.urdf_path,
        rest_qpos=rest,
        joint_signs=signs,
        **ik_overrides,
    ))

    problems = check_calibration_plausible(follower, logger)
    if problems and not args.skip_calibration_check:
        raise SystemExit(
            "Calibration failed the URDF plausibility check (see errors above).\n"
            "Recalibrate with the arm held in the MIDDLE pose at the first ENTER\n"
            "prompt — an L-shape: upper arm VERTICAL, elbow bent ~90°, forearm +\n"
            "gripper pointing straight FORWARD horizontally, wrist straight,\n"
            "gripper not rolled — then sweep every joint to both physical stops:\n"
            f"    lerobot-calibrate --robot.type=so101_follower "
            f"--robot.port={args.port} --robot.id={args.id}\n"
            "Verify with: python tools/so101_readout.py --port "
            f"{args.port} --id {args.id}\n"
            "(--skip-calibration-check overrides, at your own risk.)"
        )

    teleop.connect()
    # Sync each servo's stale RAM Goal_Position to its present position
    # BEFORE torque-enable: Feetech goals persist across host processes
    # while the bus is powered, and connect() would otherwise snap the
    # arm to the previous session's last command at full speed.
    presync_goal_positions(args.port)
    follower.connect()

    # Second gate: the first observation must be inside the URDF limits
    # (±15° slack) before the ramp runs. Out-of-range readings mean EITHER
    # the arm is resting folded past the model's range (fine — the operator
    # repositions it, torque off, and we re-check) OR the calibration zero
    # is broken (readings stay wrong no matter how the arm is posed —
    # abort and recalibrate). Ctrl-C aborts.
    def _out_of_range(obs) -> list[str]:
        solver = teleop._arm["solver"]
        bad = []
        for j, name in enumerate(JOINT_NAMES):
            # Follower degrees → model convention before comparing to limits.
            v = signs[j] * float(obs.get(f"{name}.pos", 0.0))
            lo, hi = np.degrees(solver.joint_limits[j])
            if not (lo - 15.0 <= v <= hi + 15.0):
                bad.append(f"{name}={v:.1f}° (model frame) outside URDF [{lo:.0f}, {hi:.0f}]")
        return bad

    if not args.skip_calibration_check:
        try:
            torque_dropped = False
            while True:
                bad = _out_of_range(get_observation_median(follower))
                if not bad:
                    break
                if not torque_dropped:
                    follower.bus.disable_torque()
                    torque_dropped = True
                print(
                    "\nJoints outside the model's range:\n  " + "\n  ".join(bad) +
                    "\nIf the arm is just resting folded up: torque is now OFF — "
                    "move it roughly toward the middle pose (upper arm up, forearm "
                    "forward) and press ENTER to re-check.\nIf it reads wrong in "
                    "every pose, the calibration zero is broken: Ctrl-C and "
                    "recalibrate (verify with tools/so101_readout.py)."
                )
                input("ENTER to re-check... ")
            if torque_dropped:
                follower.bus.enable_torque()
        except KeyboardInterrupt:
            follower.disconnect()
            teleop.disconnect()
            raise SystemExit("aborted at the calibration gate")

    ramp_to_rest(follower, build_rest_action(rest, signs), args.rest_duration_s, args.rest_steps, logger)

    period = 1.0 / args.freq
    logger.info("vr-teleop running at %d Hz (hand=%s); Ctrl-C to stop", args.freq, args.hand)
    get_torques = getattr(follower, "get_joint_torques", None)
    tick_count = 0
    try:
        next_tick = time.perf_counter()
        while True:
            action = teleop.get_action()
            # Log the first few seconds of commanded actions (every 30th
            # tick) so a joint that "dives on its own" can be told apart
            # from a joint that is being commanded to dive.
            if tick_count < 600 and tick_count % 30 == 0:
                logger.info("action[t=%d]: %s", tick_count,
                            {k: round(v, 1) for k, v in action.items()})
            tick_count += 1
            follower.send_action(action)
            # Per-tick force haptic: best-effort; silent if the follower
            # doesn't expose gripper torque.
            if get_torques is not None:
                try:
                    torques = get_torques()
                except Exception:
                    torques = None
                if torques:
                    teleop.send_feedback({"torques": torques})
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        try:
            teleop.disconnect()
        finally:
            follower.disconnect()


if __name__ == "__main__":
    main()
