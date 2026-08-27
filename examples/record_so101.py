"""VR demonstration recording — SO-101 + Quest teleop → LeRobotDataset.

Records teleoperated episodes (joint observations + commanded actions +
camera frames) into a LeRobotDataset ready for imitation-learning
training (ACT, Diffusion Policy, ...). Built on LeRobot's own
`lerobot-record` machinery (`lerobot.scripts.lerobot_record.record`),
wrapped with the same hardware-safety sequence as
`examples/teleop_so101.py`:

  1. calibration plausibility check against the URDF ranges,
  2. `presync_goal_positions()` BEFORE torque-enable (a stale RAM
     Goal_Position otherwise snaps the arm at full speed on connect),
  3. first-observation gate (readings must land inside URDF limits),
  4. velocity-limited ramp to the rest pose. The prep connection leaves
     torque ON at rest, so the follower re-connect inside `record()` is
     a no-op for the servos.

Episode control (the operator wears the headset, so keyboard-only
control is clumsy):

  Quest controller:
    B (right controller)  end the current episode / end the reset phase
    Y (left controller)   discard + re-record the current episode
  Keyboard (pynput; same as stock lerobot-record):
    Right arrow           end episode / end reset phase
    Left arrow            re-record episode
    Escape                stop the whole session (current episode saves)

The dataset lands under `$HF_LEROBOT_HOME` (default
`~/.cache/huggingface/lerobot/<repo_id>`). `--push-to-hub` uploads it
afterwards; the default keeps everything local. Record more episodes
into the SAME dataset later with `--resume`.

Pre-requisites: everything from examples/teleop_so101.py (relay up,
Quest in the WebXR page, calibrated follower, SO101_URDF), plus the
dataset extras: uv pip install 'lerobot[core_scripts]'.

The RealSense RGB node cannot be shared: if the relay was started with
CAM_TOP pointing at it, restart the relay without CAM_TOP while
recording (in passthrough you see the real scene anyway).

Run:
    python examples/record_so101.py --repo-id local/so101_pick_cube \
        --task "Pick up the cube and place it in the box"

Train afterwards (see the README's "Training ACT" section):
    lerobot-train --dataset.repo_id=local/so101_pick_cube \
        --policy.type=act --output_dir=outputs/train/act_so101 \
        --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import websockets

# Same-directory import (sys.path[0] is examples/ when run as a script).
from teleop_so101 import (
    JOINT_NAMES,
    build_rest_action,
    check_calibration_plausible,
    get_observation_median,
    parse_rest_pose_env,
    ramp_to_rest,
)

from vr_teleop_kit.ik.so101_ik import SO101IKSolver
from vr_teleop_kit.ik.so101_model import DEFAULT_Q_REST
from vr_teleop_kit.lerobot.cli import add_ik_cli_args, ik_kwargs_from_args
from vr_teleop_kit.lerobot.so101_quest_teleop import SO101QuestTeleoperatorConfig
from vr_teleop_kit.lerobot.so101_utils import presync_goal_positions

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.keyboard_input import apply_recording_control
from lerobot.utils.utils import init_logging, log_say

# The module (not just the symbols): record() looks up
# init_keyboard_listener through the module namespace, which is what
# lets us wrap it to add the VR episode buttons.
import lerobot.scripts.lerobot_record as lerobot_record

logger = logging.getLogger(__name__)

# Stable v4l path of the RealSense D455 RGB stream (index2 is the only
# node cv2 can open on this camera).
REALSENSE_RGB_GLOB = "/dev/v4l/by-id/usb-Intel_R__RealSense*video-index2"

# Button index 5 = B (right controller) / Y (left controller) — the
# "handoff" buttons in so101_quest_teleop.py, unused during recording.
EPISODE_BUTTON_INDEX = 5

# Upper bound on the "get ready" gate (seconds). It normally ends on a
# button press; this only stops a forgotten session from running forever.
START_GATE_MAX_S = 3600


def dataset_root(repo_id: str, root: str | None) -> Path:
    """Where the dataset lives on disk.

    Mirrors LeRobotDataset's own default ($HF_LEROBOT_HOME/<repo_id>),
    but resolved eagerly: `resume()` rejects a None root outright, so the
    recorder always passes a concrete path.
    """
    if root:
        return Path(root).expanduser()
    base = os.environ.get("HF_LEROBOT_HOME") or (Path.home() / ".cache/huggingface/lerobot")
    return Path(base).expanduser() / repo_id


def find_realsense() -> str | None:
    """Serial of the first RealSense librealsense can see, or None."""
    try:
        from lerobot.cameras.realsense import RealSenseCamera
    except Exception:  # pragma: no cover - pyrealsense2 not installed
        return None
    try:
        found = RealSenseCamera.find_cameras()
    except Exception as e:
        logger.debug("RealSense discovery failed: %s", e)
        return None
    return str(found[0]["id"]) if found else None


def disable_realsense_emitter() -> None:
    """Turn the IR structured-light projector off.

    The D455's depth projector paints a dense dot pattern on the scene,
    and on this unit it bleeds through the colour sensor's IR filter —
    every surface ends up speckled. Depth is not recorded here, so the
    emitter is pure contamination of the training images. The setting
    lives in the camera and persists until something re-enables it or the
    device is power-cycled, so this is a no-op on an already-clean rig.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return
    try:
        devices = list(rs.context().query_devices())
        for dev in devices:
            for sensor in dev.query_sensors():
                if sensor.supports(rs.option.emitter_enabled):
                    if sensor.get_option(rs.option.emitter_enabled) != 0:
                        sensor.set_option(rs.option.emitter_enabled, 0)
                        logger.info("RealSense IR emitter disabled (was speckling the RGB frames)")
    except Exception as e:
        logger.warning("could not disable the RealSense IR emitter: %s", e)


def build_camera_configs(specs: list[str], width: int, height: int, fps: int) -> dict:
    """Turn repeated `--camera NAME=SOURCE` args into LeRobot camera configs.

    SOURCE forms:
      auto            the RealSense, through its native librealsense
                      driver; falls back to its v4l2 node if
                      pyrealsense2 is unavailable
      rs:<serial>     a specific RealSense
      /dev/videoN     any plain UVC webcam (a wrist camera, say)
      <int>           the same, by OpenCV index

    RealSense devices deliberately do NOT go through OpenCV. Their colour
    node streams UYVY, which the v4l2 path decodes wrong: the image comes
    out dark with a heavy magenta cast (measured on this rig: mean
    brightness 56/255 and a +26 magenta bias, versus 93/255 and −3
    through librealsense). Those are the pixels a policy would be trained
    on, so the native driver is not a nicety here.
    """
    configs: dict = {}
    for spec in specs:
        name, sep, source = spec.partition("=")
        if not sep or not name or not source:
            raise SystemExit(f"--camera must be NAME=SOURCE (e.g. top=auto), got {spec!r}")

        serial = None
        if source == "auto":
            serial = find_realsense()
            if serial is None:
                matches = sorted(glob.glob(REALSENSE_RGB_GLOB))
                fallback = matches[0] if matches else "/dev/video2"
                logger.warning(
                    "no RealSense via librealsense (is pyrealsense2 installed?) — "
                    "falling back to the v4l2 node %s, which renders dark and "
                    "magenta on this camera", fallback,
                )
                configs[name] = OpenCVCameraConfig(
                    index_or_path=fallback, width=width, height=height, fps=fps)
                continue
        elif source.startswith("rs:"):
            serial = source[3:]

        if serial is not None:
            configs[name] = RealSenseCameraConfig(
                serial_number_or_name=serial, width=width, height=height, fps=fps)
            logger.info("camera %r: RealSense %s (native driver)", name, serial)
        else:
            index_or_path = int(source) if source.isdigit() else source
            configs[name] = OpenCVCameraConfig(
                index_or_path=index_or_path, width=width, height=height, fps=fps)
            logger.info("camera %r: OpenCV %s", name, index_or_path)
    return configs


def start_vr_episode_buttons(ws_url: str, events: dict) -> None:
    """Drive lerobot-record's episode events from the relay.

    Two sources, one daemon thread on a second relay connection:

      xr_frame        right B rising edge → end episode / end reset
                      left  Y rising edge → re-record the episode
      record_control  {"action": "end_episode" | "rerecord" | "stop"},
                      sent by the web UI's session controls

    Rising-edge detection means a held button fires once, not once per
    phase. The web controls exist because a session started from the
    /data page has no terminal: the pynput shortcuts are out of reach,
    and stopping it by killing the process would lose the episode in
    progress (an episode is only written on a clean end).

    Independent of the teleoperator instance, so it needs no hooks inside
    record().
    """

    async def watch() -> None:
        last = {"left": False, "right": False}
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        if msg.get("type") == "record_control":
                            # Same mapping the keyboard uses, so a click and
                            # a key press cannot diverge.
                            action = msg.get("action")
                            control = {
                                "end_episode": "right",
                                "rerecord": "left",
                                "stop": "esc",
                            }.get(action)
                            if control:
                                logger.info("web control %r -> %s", action, control)
                                apply_recording_control(control, events)
                            continue
                        if msg.get("type") != "xr_frame":
                            continue
                        ctrls = msg.get("controllers") or {}
                        for hand in ("left", "right"):
                            buttons = (ctrls.get(hand) or {}).get("buttons") or []
                            pressed = (
                                bool(buttons[EPISODE_BUTTON_INDEX].get("p"))
                                if len(buttons) > EPISODE_BUTTON_INDEX
                                else False
                            )
                            if pressed and not last[hand]:
                                if hand == "right":
                                    logger.info("VR button B → end of episode/reset phase")
                                else:
                                    logger.info("VR button Y → re-record episode")
                                    events["rerecord_episode"] = True
                                events["exit_early"] = True
                            last[hand] = pressed
            except Exception:
                await asyncio.sleep(1.0)

    threading.Thread(
        target=lambda: asyncio.run(watch()), name="vr-episode-buttons", daemon=True
    ).start()


def install_start_gate(play_sounds: bool) -> None:
    """Make every episode wait for the operator instead of recording the
    instant the process boots.

    Wraps `record_loop`: before each call that carries a dataset (a real
    recording episode), run the SAME loop with no dataset — full teleop,
    nothing written to disk — until the operator presses B. That is
    LeRobot's own reset-phase machinery reused as a "get ready" phase, so
    you can pose the arm, set up the scene, and start recording
    deliberately rather than racing a countdown that began while you were
    still putting the headset on.

    `record()` calls `record_loop` with keyword arguments only, and omits
    `dataset` entirely for the reset phase — that is what distinguishes a
    recording episode from a reset here.
    """
    orig_record_loop = lerobot_record.record_loop

    def gated_record_loop(*args, **kwargs):
        dataset = kwargs.get("dataset")
        events = kwargs.get("events") or {}
        if not args and dataset is not None and not events.get("stop_recording"):
            logger.info("READY — press B (right controller) to start recording")
            log_say("Ready. Press B to start recording", play_sounds)

            gate_kwargs = {k: v for k, v in kwargs.items() if k != "dataset"}
            gate_kwargs["control_time_s"] = START_GATE_MAX_S
            orig_record_loop(**gate_kwargs)

            # Y during the gate is meaningless — nothing has been recorded
            # yet — and would otherwise discard the episode about to start.
            events["rerecord_episode"] = False
            if events.get("stop_recording"):
                return None
            logger.info("RECORDING — press B again to end the episode")
            log_say("Recording", play_sounds)
        return orig_record_loop(*args, **kwargs)

    lerobot_record.record_loop = gated_record_loop


def safety_prep(args, rest: list[float], signs: list[float]) -> None:
    """Pre-flight the follower exactly like examples/teleop_so101.py, then
    leave it holding the rest pose with torque ON (disable_torque_on_
    disconnect=False) so record()'s own connect cannot snap the arm."""
    follower = SO101Follower(SO101FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        position_p_coefficient=32,
        disable_torque_on_disconnect=False,
    ))

    problems = check_calibration_plausible(follower, logger)
    if problems and not args.skip_calibration_check:
        raise SystemExit(
            "Calibration failed the URDF plausibility check (see errors above); "
            "recalibrate (middle pose = the L-shape) or pass --skip-calibration-check."
        )

    presync_goal_positions(args.port)
    follower.connect()
    try:
        if not args.skip_calibration_check:
            # First-observation gate: solver limits come from the URDF.
            solver = SO101IKSolver(urdf_path=args.urdf_path or None)
            while True:
                obs = get_observation_median(follower)
                bad = []
                for j, name in enumerate(JOINT_NAMES):
                    v = signs[j] * float(obs.get(f"{name}.pos", 0.0))
                    lo, hi = np.degrees(solver.joint_limits[j])
                    if not (lo - 15.0 <= v <= hi + 15.0):
                        bad.append(f"{name}={v:.1f}° (model frame) outside URDF [{lo:.0f}, {hi:.0f}]")
                if not bad:
                    break
                follower.bus.disable_torque()
                print(
                    "\nJoints outside the model's range:\n  " + "\n  ".join(bad) +
                    "\nTorque is OFF — pose the arm roughly toward the middle "
                    "pose and press ENTER to re-check (Ctrl-C to abort)."
                )
                input("ENTER to re-check... ")
                follower.bus.enable_torque()
        ramp_to_rest(follower, build_rest_action(rest, signs),
                     args.rest_duration_s, args.rest_steps, logger)
    finally:
        follower.disconnect()  # torque stays ON: arm holds the rest pose


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # ── hardware / teleop (same flags as examples/teleop_so101.py) ──
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="so101",
                    help="robot id — must match the id used with lerobot-calibrate")
    ap.add_argument("--hand", choices=("left", "right"), default="right")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws")
    ap.add_argument("--urdf-path", default="",
                    help="SO-101 URDF path (default: SO101_URDF env var)")
    ap.add_argument("--joint-signs", default="1,1,1,1,1")
    ap.add_argument("--skip-calibration-check", action="store_true")
    ap.add_argument("--rest-duration-s", type=float, default=3.0)
    ap.add_argument("--rest-steps", type=int, default=90)
    add_ik_cli_args(ap)
    # ── dataset ──
    ap.add_argument("--repo-id", required=True,
                    help="dataset name, hf_user/name convention (e.g. local/so101_pick_cube)")
    ap.add_argument("--task", required=True,
                    help='short task description, e.g. "Pick up the cube and place it in the box"')
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--fps", type=int, default=30,
                    help="recording rate; also the teleop tick rate while recording")
    ap.add_argument("--episode-time-s", type=float, default=60,
                    help="hard cap per episode — normally you end earlier with B")
    ap.add_argument("--reset-time-s", type=float, default=None,
                    help="timed scene-reset window between episodes (end early with B). "
                         "Defaults to 0 with the start gate on (the gate already gives "
                         "you unlimited time between episodes) and to 15 with "
                         "--no-start-gate.")
    ap.add_argument("--no-start-gate", action="store_true",
                    help="start each episode immediately instead of waiting for B. "
                         "Off by default: without the gate, recording begins the moment "
                         "the arm finishes its ramp, while you are still getting ready.")
    ap.add_argument("--resume", action="store_true",
                    help="append episodes to an existing dataset with this repo-id")
    ap.add_argument("--root", default=None,
                    help="dataset directory (default: $HF_LEROBOT_HOME/<repo-id>, i.e. "
                         "~/.cache/huggingface/lerobot/<repo-id>)")
    ap.add_argument("--push-to-hub", action="store_true",
                    help="upload to the HuggingFace Hub after recording (needs `hf auth login`)")
    ap.add_argument("--display-data", action="store_true",
                    help="live rerun visualization of frames + joints")
    ap.add_argument("--no-sounds", action="store_true",
                    help="disable spoken episode announcements")
    ap.add_argument("--no-vr-episode-buttons", action="store_true",
                    help="disable episode control from the Quest controllers and the "
                         "web UI (both arrive over the relay)")
    # ── cameras ──
    ap.add_argument("--camera", action="append", default=None, metavar="NAME=SOURCE",
                    help="camera stream(s) to record; repeatable. SOURCE is "
                         "'auto' (the RealSense via its native driver — the default, "
                         "as top=auto), 'rs:<serial>' for a specific RealSense, or a "
                         "/dev/videoN path / OpenCV index for a plain webcam. "
                         "e.g. --camera top=auto --camera wrist=/dev/video4")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    args = ap.parse_args()

    init_logging()
    logger.setLevel(logging.INFO)

    rest = parse_rest_pose_env("SO101_REST_POSE", DEFAULT_Q_REST.tolist())
    signs = [float(s) for s in args.joint_signs.split(",")]
    if len(signs) != 5 or any(s not in (1.0, -1.0) for s in signs):
        raise SystemExit(f"--joint-signs must be 5 comma-separated +1/-1, got {args.joint_signs!r}")

    # The start gate gives unlimited time between episodes, so the timed
    # reset window is redundant with it (it would just be a second button
    # press per episode).
    start_gate = not args.no_start_gate
    if args.reset_time_s is None:
        args.reset_time_s = 0.0 if start_gate else 15.0

    disable_realsense_emitter()
    camera_configs = build_camera_configs(
        args.camera if args.camera is not None else ["top=auto"],
        args.cam_width, args.cam_height, args.cam_fps,
    )

    # ── pre-flight: gates + presync + ramp to rest (torque stays on) ──
    safety_prep(args, rest, signs)

    # ── stock lerobot-record from here on ──
    cfg = lerobot_record.RecordConfig(
        robot=SO101FollowerConfig(
            port=args.port,
            id=args.id,
            use_degrees=True,
            position_p_coefficient=32,
            cameras=camera_configs,
        ),
        teleop=SO101QuestTeleoperatorConfig(
            id="vr-teleop-so101",
            ws_url=args.ws_url,
            hand=args.hand,
            urdf_path=args.urdf_path,
            rest_qpos=rest,
            joint_signs=signs,
            **ik_kwargs_from_args(args),
        ),
        dataset=DatasetRecordConfig(
            repo_id=args.repo_id,
            # Always explicit: LeRobotDataset.resume() REFUSES root=None
            # (it would point the writer at the revision-safe Hub snapshot
            # cache and corrupt it), so --resume fails outright without
            # this. The path below is the same one create() picks by
            # default, so passing it changes nothing for a fresh dataset.
            root=dataset_root(args.repo_id, args.root),
            single_task=args.task,
            fps=args.fps,
            episode_time_s=args.episode_time_s,
            reset_time_s=args.reset_time_s,
            num_episodes=args.num_episodes,
            push_to_hub=args.push_to_hub,
            no_stamp=True,  # stable name → --resume and lerobot-train just work
        ),
        display_data=args.display_data,
        play_sounds=not args.no_sounds,
        resume=args.resume,
    )

    if not args.no_vr_episode_buttons:
        orig_init_listener = lerobot_record.init_keyboard_listener

        def init_listener_with_vr_buttons():
            listener, events = orig_init_listener()
            start_vr_episode_buttons(args.ws_url, events)
            return listener, events

        lerobot_record.init_keyboard_listener = init_listener_with_vr_buttons

    if start_gate:
        install_start_gate(play_sounds=not args.no_sounds)

    # The precision button is A on the right controller, X on the left —
    # and it is read only on the hand that drives the arm.
    precision_btn = "A" if args.hand == "right" else "X"
    phases = (
        "  Each episode waits for you before it records anything.\n"
        "\n"
        "    READY   → teleop is live, nothing is being saved.\n"
        "              Pose the arm, set up the scene, then press B.\n"
        "    RECORD  → press B again to end and save the episode.\n"
        "              Press Y to throw the episode away and redo it.\n"
        "\n"
        if start_gate
        else "  --no-start-gate: episode 0 records IMMEDIATELY.\n"
    )
    b_label = "start / end episode" if start_gate else "end episode / end reset"
    print(
        "\n──────────────── recording controls ────────────────\n"
        f"{phases}"
        f"  grip (hold)      engage teleop        B (right)  {b_label}\n"
        f"  trigger          close gripper        Y (left)   discard + re-record\n"
        f"  {precision_btn} (hold)         precision gains      Esc (kbd)  stop session\n"
        f"  thumbstick click park at rest\n"
        "  (B/Y work on either controller; the others only on the "
        f"{args.hand} one)\n"
        "─────────────────────────────────────────────────────\n"
    )

    dataset = lerobot_record.record(cfg)
    logger.info(
        "Recorded %d episodes into %s (repo_id=%s)",
        dataset.num_episodes, dataset.root, cfg.dataset.repo_id,
    )
    print(
        f"\nReview the episodes and launch training in the browser:\n"
        f"  http://localhost:8443/data\n"
        f"\nOr straight to ACT from here:\n"
        f"  lerobot-train --dataset.repo_id={cfg.dataset.repo_id} --policy.type=act \\\n"
        f"      --output_dir=outputs/train/act_so101 --job_name=act_so101 \\\n"
        f"      --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false\n"
    )


if __name__ == "__main__":
    main()
