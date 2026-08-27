"""Subprocess entry point for running a trained policy on the real arm.

This is the only part of the /data page that moves hardware, so the
order of operations matters more than the code does:

  1. Refuse if something else holds the servo bus. The `haller_hmi`
     service sits on /dev/ttyACM0 with torque enabled; concurrent access
     corrupts Feetech packets rather than failing cleanly.
  2. `presync_goal_positions()` BEFORE anything enables torque. Feetech
     Goal_Position is a RAM register that survives across processes: the
     previous session's last command is still in there, and torque-enable
     makes the arm snap to it at full speed. Syncing goals to present
     positions makes torque-enable a no-op.
  3. Calibration plausibility check and a first-observation gate against
     the URDF limits, so a bad calibration is caught while the arm is
     still limp.
  4. Velocity-limited ramp to the rest pose, leaving torque ON, so the
     rollout's own `connect()` is a no-op for the servos.

Only then does `lerobot-rollout` take over. The duration is always
bounded — a policy loop started from a browser button should not be able
to run until someone notices.

`--dry-run` (or `"dry_run": true` in the spec) performs the checks and
prints the exact command without connecting to the arm at all.

Usage:
    python -m vr_teleop_kit.data.rollout_runner /path/to/spec.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

logger = logging.getLogger("rollout_runner")


def port_holders(device: str) -> list[str]:
    """Processes with the serial device open (best effort, this user's)."""
    out: list[str] = []
    try:
        target = os.path.realpath(device)
    except OSError:
        return out
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or proc.name == str(os.getpid()):
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.path.realpath(fd) == target:
                        cmd = (proc / "cmdline").read_bytes().decode("utf-8", "replace")
                        out.append(f"pid {proc.name}: {cmd.replace(chr(0), ' ').strip()[:160]}")
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return out


def build_argv(spec: dict, camera_json: str) -> list[str]:
    argv = [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={spec['policy_path']}",
        f"--policy.device={spec.get('device', 'cuda')}",
        "--robot.type=so101_follower",
        f"--robot.port={spec['port']}",
        f"--robot.id={spec.get('robot_id', 'so101')}",
        "--robot.use_degrees=true",
        f"--fps={float(spec.get('fps', 30))}",
        f"--duration={float(spec['duration_s'])}",
        # Put the arm back where it started rather than leaving it
        # wherever the policy's last action pointed.
        "--return_to_initial_position=true",
    ]
    if camera_json:
        argv.append(f"--robot.cameras={camera_json}")
    if spec.get("task"):
        argv.append(f"--task={spec['task']}")
    for extra in spec.get("extra_args") or []:
        argv.append(str(extra))
    return argv


def camera_config_json(spec: dict, touch_hardware: bool = True) -> str:
    """Build lerobot's `--robot.cameras` mapping from `NAME=SOURCE` specs.

    Reuses `examples/record_so101.py`'s resolver so a RealSense goes
    through librealsense here exactly as it did during recording. Its
    v4l2/UYVY path decodes dark with a heavy magenta cast, and a policy
    shown magenta frames it was never trained on will simply not work.
    """
    specs = spec.get("cameras") or []
    if not specs:
        return ""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
    from record_so101 import build_camera_configs, disable_realsense_emitter

    # The emitter setting persists in the camera, so writing it is a real
    # device change — a dry run resolves the config and stops there.
    if touch_hardware:
        disable_realsense_emitter()
    configs = build_camera_configs(
        specs, int(spec.get("cam_width", 640)), int(spec.get("cam_height", 480)),
        int(spec.get("cam_fps", 30)),
    )

    from lerobot.cameras.realsense import RealSenseCameraConfig

    parts = []
    for name, cfg in configs.items():
        if isinstance(cfg, RealSenseCameraConfig):
            parts.append(
                f"{name}: {{type: intelrealsense, serial_number_or_name: "
                f"{cfg.serial_number_or_name}, width: {cfg.width}, height: {cfg.height}, "
                f"fps: {cfg.fps}}}"
            )
        else:
            parts.append(
                f"{name}: {{type: opencv, index_or_path: {cfg.index_or_path}, "
                f"width: {cfg.width}, height: {cfg.height}, fps: {cfg.fps}}}"
            )
    return "{" + ", ".join(parts) + "}"


def safety_prep(spec: dict) -> None:
    """Steps 1-4 above. Leaves the arm holding the rest pose, torque ON."""
    import numpy as np

    from vr_teleop_kit.ik.so101_ik import SO101IKSolver
    from vr_teleop_kit.ik.so101_model import DEFAULT_Q_REST
    from vr_teleop_kit.lerobot.so101_utils import (
        JOINT_NAMES,
        build_rest_action,
        check_calibration_plausible,
        get_observation_median,
        parse_rest_pose_env,
        presync_goal_positions,
        ramp_to_rest,
    )

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    port = spec["port"]
    signs = [float(s) for s in str(spec.get("joint_signs", "1,1,1,1,1")).split(",")]
    rest = parse_rest_pose_env("SO101_REST_POSE", DEFAULT_Q_REST.tolist())

    follower = SO101Follower(SO101FollowerConfig(
        port=port,
        id=spec.get("robot_id", "so101"),
        use_degrees=True,
        position_p_coefficient=32,
        disable_torque_on_disconnect=False,
    ))

    problems = check_calibration_plausible(follower, logger)
    if problems and not spec.get("skip_calibration_check"):
        raise SystemExit(
            "Calibration failed the URDF plausibility check (see above). Recalibrate "
            "(middle pose = the L-shape) before running a policy on this arm."
        )

    presync_goal_positions(port)
    follower.connect()
    try:
        if not spec.get("skip_calibration_check"):
            solver = SO101IKSolver(urdf_path=os.environ.get("SO101_URDF") or None)
            obs = get_observation_median(follower)
            bad = []
            for j, name in enumerate(JOINT_NAMES):
                v = signs[j] * float(obs.get(f"{name}.pos", 0.0))
                lo, hi = np.degrees(solver.joint_limits[j])
                if not (lo - 15.0 <= v <= hi + 15.0):
                    bad.append(f"{name}={v:.1f}° outside URDF [{lo:.0f}, {hi:.0f}]")
            if bad:
                # No interactive re-check here: nobody is at the terminal
                # when this is launched from a browser.
                follower.bus.disable_torque()
                raise SystemExit(
                    "Joints are outside the model's range:\n  " + "\n  ".join(bad) +
                    "\nTorque is OFF. Pose the arm toward the middle pose and start again."
                )
        ramp_to_rest(follower, build_rest_action(rest, signs), 3.0, 90, logger)
    finally:
        follower.disconnect()  # torque stays ON: the arm holds the rest pose


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]
    if not args:
        print("usage: python -m vr_teleop_kit.data.rollout_runner SPEC.json [--dry-run]")
        return 2

    spec = json.loads(Path(args[0]).read_text())
    dry_run = dry_run or bool(spec.get("dry_run"))
    run_dir = Path(spec["run_dir"])

    from . import runs

    status, exit_code, error = "done", 0, ""
    try:
        port = spec["port"]
        if not Path(port).exists():
            raise SystemExit(f"no serial device at {port}")
        holders = port_holders(port)
        if holders:
            raise SystemExit(
                f"{port} is already open by:\n  " + "\n  ".join(holders) +
                "\nStop it first — two processes on one Feetech bus corrupt each other's packets."
            )
        if float(spec.get("duration_s") or 0) <= 0:
            raise SystemExit("duration_s must be > 0")

        camera_json = camera_config_json(spec, touch_hardware=not dry_run)
        argv = build_argv(spec, camera_json)
        print("rollout argv:", " ".join(argv), flush=True)

        if dry_run:
            print("\ndry run: preflight passed, the arm was not touched.", flush=True)
            return 0

        print("\npre-flight: presync goals, calibration gates, ramp to rest", flush=True)
        safety_prep(spec)

        print("\nstarting policy — the arm is moving on its own now", flush=True)
        import lerobot.scripts.lerobot_rollout as lerobot_rollout
        sys.argv = argv
        lerobot_rollout.main()
    except KeyboardInterrupt:
        status, exit_code, error = "stopped", 130, "interrupted"
        print("\ninterrupted", flush=True)
    except SystemExit as e:
        code = e.code
        if isinstance(code, str):
            print(code, flush=True)
            status, exit_code, error = "failed", 1, code.splitlines()[0]
        else:
            exit_code = int(code or 0)
            status = "done" if exit_code == 0 else "failed"
            error = "" if exit_code == 0 else f"exited with {exit_code}"
    except Exception as e:
        status, exit_code, error = "failed", 1, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        runs.write_result(run_dir, status, exit_code, error)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
