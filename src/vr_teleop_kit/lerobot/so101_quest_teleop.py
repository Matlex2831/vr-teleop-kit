"""SO101QuestTeleoperator — single-arm VR teleoperation for the SO-101.

Subscribes to a relay-mode FastAPI server's `/ws` and reads `xr_frame`
broadcasts (controller poses + buttons + headset pose). For the chosen
controller hand, maintains a `ClutchPoseMapper` and an `SO101IKSolver`;
on the rising edge of the grip button, captures the engage frame and
runs differential IK each tick. `get_action()` returns the joint action
dict matching `SO101Follower.action_features`:

    shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
    wrist_flex.pos, wrist_roll.pos          (degrees)
    gripper.pos                             (0..100, 0 = closed)

IMPORTANT: joint values are emitted in DEGREES — configure the follower
with `use_degrees=True` so both sides speak the same units. The gripper
stays 0..100 normalized (LeRobot convention) regardless.

Standard LeRobot-style loop:

    teleop = SO101QuestTeleoperator(SO101QuestTeleoperatorConfig(...))
    follower = SO101Follower(SO101FollowerConfig(..., use_degrees=True))
    teleop.connect(); follower.connect()
    while True:
        follower.send_action(teleop.get_action())
        time.sleep(1/freq)

Per-tick pipeline inside `_update_arm` (same architecture as the
bimanual DK1 teleop — see bi_quest_teleop.py for the full rationale):

  1. Staleness gate — no `xr_frame` for > XR_FRAME_STALE_TIMEOUT_S →
     skip the tick; silently re-anchor on recovery.
  2. EMA pose filter on the controller pose.
  3. Precision button (A/X) — re-anchor on transition, then scale gains.
  4. Clutch edges — rising → anchor (controller pose, EE pose, wrist
     anchor as rotation pivot, yaw-corrected R); falling → disengage.
  5. Yaw correction — R_engage = R_CALIB · R_y(-yaw_now).
  6. IK step — FK at the last commanded qpos → mapper.target() with the
     absorbing reach limits → SO101IKSolver.solve().
  7. Haptic mix — max of (limit_pressure, pos_err_norm,
     singularity_proximity, wrist_gimbal_proximity); EMA-smoothed and
     broadcast in ik_state.

The SO-101 wrist has 2 DoF (flex + roll), so orientation tracking is
partial by construction: the demanded rotation is projected onto the
reachable pitch×roll subspace and the yaw component follows the arm's
position instead of the operator's wrist. See ik/so101_ik.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lerobot is required to use SO101QuestTeleoperator. "
        "Run from a project environment with lerobot installed."
    ) from e

try:
    import websockets
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "websockets is required to use SO101QuestTeleoperator. "
        "Install with: uv add websockets   (or pip install websockets)"
    ) from e

import mujoco

from ..core.pose_mapping import ClutchPoseMapper
from ..ik.so101_model import DEFAULT_Q_REST
from ..ik.so101_ik import ARM_DOFS, SO101IKSolver

logger = logging.getLogger(__name__)


# Default rotation taking Quest `local-floor` world axes into the arm base
# frame. The SO-101 new-calib base frame has +x pointing INTO the
# workspace (the direction the extended arm reaches at qpos=0), +y left,
# +z up. With the operator standing behind the robot facing the workspace
# (the natural tabletop teleop stance): arm_x = operator-forward
# (-quest_z), arm_y = operator-left (-quest_x), arm_z = up (+quest_y) —
# the same mapping as the DK1 default. Override via `r_calib` if your
# mounting differs (the per-engage yaw correction handles the operator
# turning in the room, so only the axis convention matters).
DEFAULT_R_CALIB = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

GRIP_BUTTON_INDEX = 1  # boolean: clutch
TRIGGER_BUTTON_INDEX = 0  # analog 0..1: gripper closure
PRECISION_BUTTON_INDEX = 4  # boolean: A (right) / X (left) — hold for precision scale
HANDOFF_BUTTON_INDEX = 5  # boolean: B (right) / Y (left) — intervention handoff signal
REST_RAMP_BUTTON_INDEX = 3  # thumbstick click — "go home" ramp

NQ = ARM_DOFS + 1  # 5 arm joints + 1 gripper jaw (viewer mirror)

# If no xr_frame has been received in this many seconds, treat the
# controller stream as stale (see bi_quest_teleop.py).
XR_FRAME_STALE_TIMEOUT_S = 0.2

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _yaw_from_quat_xyzw(q_xyzw) -> float:
    """Yaw about world-up (+Y), in radians, from a (x, y, z, w) WebXR quat."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def _R_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@TeleoperatorConfig.register_subclass("so101_quest_teleop")
@dataclass
class SO101QuestTeleoperatorConfig(TeleoperatorConfig):
    """Config for SO101QuestTeleoperator.

    `hand` picks which Quest controller drives the arm ("left"/"right").
    `ws_url` points at the FastAPI relay server hosting the WebXR page.
    `rest_qpos` is five joint angles (rad): where the teleop initialises
    its qpos, where the thumbstick-click ramp parks the arm, and what the
    IK's Tikhonov posture bias pulls toward.

    IK / mapping knobs follow `BiQuestTeleoperatorConfig`; see that class
    and `SO101IKSolver` for the semantics. Differences: `w0` defaults far
    smaller (the SO-101's links are ~3× shorter, so det(J) runs ~30×
    smaller), and the Δq cap is 5 long (3 position + 2 wrist joints).
    """

    hand: str = "right"
    ws_url: str = "ws://127.0.0.1:8443/ws"
    publish_ik_state: bool = True
    connect_timeout_s: float = 5.0
    # Path to the SO-101 URDF (deliberately not vendored — clone
    # github.com/TheRobotStudio/SO-ARM100, use Simulation/SO101/
    # so101_new_calib.urdf). Empty string falls back to the SO101_URDF
    # environment variable.
    urdf_path: str = ""
    # 3x3 rotation (row-major) taking Quest world vectors into the arm
    # base frame. See DEFAULT_R_CALIB above for the convention.
    r_calib: list[list[float]] = field(default_factory=lambda: [row[:] for row in DEFAULT_R_CALIB])
    rest_qpos: list[float] = field(default_factory=lambda: DEFAULT_Q_REST.tolist())

    # IK / mapping knobs.
    lam: float = 0.05
    lam0: float = 0.15
    w0: float = 0.002
    mu: float = 0.02
    lam_rot: float = 0.05
    lam0_rot: float = 0.4
    w0_rot: float = 0.5
    rot_reach_limit: float = 0.6  # rad (~34°)
    pos_reach_limit: float = 0.15  # m — smaller arm, smaller wall
    rot_err_hold: float = 2.2
    scale_translation: float = 1.0  # ~1:1 — the SO-101 workspace is arm-sized
    scale_rotation: float = 1.0
    precision_factor: float = 0.5
    pose_filter_alpha: float = 0.8
    # Per-joint Δq cap (rad/tick), length 5: 3 position joints + 2 wrist.
    max_dq_per_joint: list[float] | None = field(
        default_factory=lambda: [0.06, 0.06, 0.06, 0.24, 0.24]
    )
    max_dq_per_joint_scalar_pos: float = 0.06
    max_dq_per_joint_scalar_rot: float = 0.24
    # ── Force haptic (gripper torque → controller vibration) ──
    # Optional: fed by send_feedback({"torques": {...}}) from a follower
    # that exposes gripper torque (see the README). Both per-arm threshold
    # fields are kept so the stock web Settings panel drives them
    # unchanged; only the field matching `hand` is consumed.
    force_haptic_threshold_nm_left: float = 0.35
    force_haptic_threshold_nm_right: float = 0.35
    force_haptic_max_nm: float = 1.0
    force_haptic_velocity_comp_nm: float = 0.5
    force_haptic_enabled: bool = True
    rest_ramp_duration_s: float = 2.0
    # Per-joint sign (+1/-1, length 5) between the MODEL convention (the
    # new-calib URDF) and the FOLLOWER's reported/commanded degrees.
    # follower_deg = sign * model_deg. All +1 for a standard SO-101
    # follower build; a unit with mirrored servo mounts (e.g. a
    # leader-style build) needs -1 on the affected joints. Determine
    # empirically with tools/so101_readout.py (limp joints hang DOWN =
    # model-positive for lift/elbow/wrist_flex).
    joint_signs: list[float] = field(default_factory=lambda: [1.0] * 5)

    def __post_init__(self) -> None:
        if self.hand not in ("left", "right"):
            raise ValueError(
                f"SO101QuestTeleoperatorConfig.hand must be 'left' or 'right', "
                f"got {self.hand!r}"
            )
        if len(self.joint_signs) != 5 or any(s not in (1.0, -1.0, 1, -1) for s in self.joint_signs):
            raise ValueError(
                f"joint_signs must be 5 values of +1/-1, got {self.joint_signs!r}"
            )


class SO101QuestTeleoperator(Teleoperator):
    config_class = SO101QuestTeleoperatorConfig
    name = "so101_quest_teleop"

    def __init__(self, config: SO101QuestTeleoperatorConfig) -> None:
        super().__init__(config)
        self.config = config

        self._r_calib = np.asarray(config.r_calib, dtype=float)
        if self._r_calib.shape != (3, 3):
            raise ValueError(
                f"r_calib must be a 3x3 rotation matrix, got shape {self._r_calib.shape}"
            )

        self._hand = config.hand

        # Lock guards reads/writes from the LeRobot thread vs the WS
        # receiver thread.
        self._lock = threading.Lock()
        self._latest_xr_frame: dict | None = None
        self._last_xr_frame_time: float = 0.0

        q_rest = np.asarray(config.rest_qpos, dtype=float)
        if q_rest.shape != (ARM_DOFS,):
            raise ValueError(f"rest_qpos must have {ARM_DOFS} values, got shape {q_rest.shape}")

        solver = SO101IKSolver(
            urdf_path=config.urdf_path or None,
            lam_pos=config.lam,
            lam0=config.lam0,
            w0=config.w0,
            mu=config.mu,
            lam_rot=config.lam_rot,
            lam0_rot=config.lam0_rot,
            w0_rot=config.w0_rot,
            rot_err_hold=config.rot_err_hold,
            q_rest=q_rest.copy(),
            max_dq_per_joint=config.max_dq_per_joint,
        )
        # Gripper jaw qpos range from the model — used only for the sim
        # viewer mirror in qpos[5]. Jaw upper limit = fully open.
        jaw_id = mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        if jaw_id >= 0:
            self._jaw_range = tuple(float(v) for v in solver.model.jnt_range[jaw_id])
        else:  # pragma: no cover - URDF always has the jaw
            self._jaw_range = (0.0, 1.7)

        qpos_init = np.zeros(NQ)
        qpos_init[:ARM_DOFS] = q_rest
        qpos_init[ARM_DOFS] = self._jaw_range[1]  # open
        self._arm: dict = {
            "solver": solver,
            "mapper": ClutchPoseMapper(
                R=self._r_calib.copy(),
                scale=config.scale_translation,
                scale_rotation=config.scale_rotation,
                rot_reach_limit=config.rot_reach_limit,
                pos_reach_limit=config.pos_reach_limit,
            ),
            "qpos": qpos_init,
            "last_grip": False,
            "last_precision": False,
            "trigger": 0.0,
            "engaged": False,
            "haptic": 0.0,
            "force_haptic": 0.0,
            "prev_gripper_pos": None,
            "prev_gripper_t": None,
            "gripper_vel_filt": 0.0,
            "needs_reanchor": False,
            "pos_filt": None,
            "quat_filt": None,
            "last_rest_button": False,
            "ramp_active": False,
            "ramp_start_q": np.zeros(ARM_DOFS),
            "ramp_target_q": q_rest.copy(),
            "ramp_start_t": 0.0,
        }
        # Raw per-hand button state, updated synchronously from every
        # xr_frame in the WS reader thread (both hands — the operator may
        # press B/Y on either controller for handoff signals even though
        # only one hand drives the arm).
        self._buttons: dict[str, dict[str, bool]] = {
            "left": {"grip": False, "handoff": False},
            "right": {"grip": False, "handoff": False},
        }

        # WS plumbing.
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_stop: threading.Event | None = None
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._ws_connected = threading.Event()

        # Idle-torque calibration window (web UI "Calibrate" button).
        self._haptic_calib: dict | None = None

        # Measured rate of `get_action()` calls (published in ik_state).
        self._last_get_action_t: float | None = None
        self._loop_hz: float | None = None

    # ---------- Teleoperator interface ----------

    @property
    def action_features(self) -> dict[str, type]:
        feats: dict[str, type] = {f"{j}.pos": float for j in JOINT_NAMES}
        feats["gripper.pos"] = float
        return feats

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._ws_connected.is_set()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        self._ws_stop = threading.Event()
        self._ws_thread = threading.Thread(
            target=self._ws_thread_main, name="so101-quest-teleop-ws", daemon=True
        )
        self._ws_thread.start()
        if not self._ws_connected.wait(timeout=self.config.connect_timeout_s):
            raise RuntimeError(
                f"SO101QuestTeleoperator: timed out connecting to {self.config.ws_url}"
            )
        logger.info("SO101QuestTeleoperator connected to %s (hand=%s)",
                    self.config.ws_url, self._hand)

    def disconnect(self) -> None:
        if self._ws_stop is not None:
            self._ws_stop.set()
        if self._ws_loop is not None and self._ws is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._ws.close(), self._ws_loop)
                fut.result(timeout=1.0)
            except Exception:
                pass
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=2.0)
        self._ws_connected.clear()
        logger.debug("SO101QuestTeleoperator disconnected")

    def send_feedback(self, feedback: dict) -> None:
        """Push runtime feedback (gripper torque → haptics) into the
        teleop. Accepts `{"torques": {"gripper.torque": Nm,
        "gripper.pos": 0..1}}`; `{hand}_`-prefixed keys also work. The
        follower side is optional — see the README's grasp-force haptics
        section."""
        if not isinstance(feedback, dict):
            return
        torques = feedback.get("torques")
        if isinstance(torques, dict):
            self._apply_torque_feedback(torques)

    def publish_state(self) -> None:
        """Force one ``ik_state`` broadcast on the WS without waiting for
        the next ``get_action()`` call."""
        if self.config.publish_ik_state and self._ws is not None and self._ws_loop is not None:
            self._publish_ik_state_async()

    def _apply_torque_feedback(self, torques: dict) -> None:
        """Map raw gripper torque (Nm) to 0..1 haptic intensity via a
        velocity-aware dead-zone linear scaling (same model as the DK1
        teleop — see bi_quest_teleop.py for the rationale)."""
        # Accept unprefixed keys or keys prefixed with our hand.
        tau_key = pos_key = None
        for prefix in ("", f"{self._hand}_"):
            if f"{prefix}gripper.torque" in torques:
                tau_key = f"{prefix}gripper.torque"
                pos_key = f"{prefix}gripper.pos"
                break
        if tau_key is None:
            return
        ceiling = max(1e-6, float(self.config.force_haptic_max_nm))
        kv = max(0.0, float(self.config.force_haptic_velocity_comp_nm))
        vel_alpha = 0.4
        now = time.time()
        with self._lock:
            arm = self._arm
            tau = abs(float(torques[tau_key]))
            if pos_key in torques:
                pos = float(torques[pos_key])
                prev_pos = arm["prev_gripper_pos"]
                prev_t = arm["prev_gripper_t"]
                if prev_pos is not None and prev_t is not None:
                    dt = max(now - prev_t, 1e-3)
                    v_raw = (pos - prev_pos) / dt
                    arm["gripper_vel_filt"] = (1.0 - vel_alpha) * arm[
                        "gripper_vel_filt"
                    ] + vel_alpha * v_raw
                arm["prev_gripper_pos"] = pos
                arm["prev_gripper_t"] = now
            threshold_base = max(
                0.0, float(getattr(self.config, f"force_haptic_threshold_nm_{self._hand}"))
            )
            threshold_eff = threshold_base + kv * abs(arm["gripper_vel_filt"])
            if threshold_eff >= ceiling:
                intensity = 0.0
            else:
                span = ceiling - threshold_eff
                intensity = max(0.0, min(1.0, (tau - threshold_eff) / span))
            if not self.config.force_haptic_enabled:
                intensity = 0.0
            arm["force_haptic"] = intensity
            if self._haptic_calib is not None:
                calib = self._haptic_calib
                calib["peak"] = max(calib["peak"], tau)
                if now - calib["start_t"] >= calib["duration_s"]:
                    self._finalize_haptic_calibration()

    # ---------- Intervention (handoff) hooks ----------

    def is_engaged(self) -> bool:
        with self._lock:
            return self._buttons[self._hand]["grip"]

    def is_handoff_pressed(self) -> bool:
        with self._lock:
            return self._buttons["left"]["handoff"] or self._buttons["right"]["handoff"]

    def is_pause_pressed(self) -> bool:
        with self._lock:
            return self._buttons["right"]["handoff"]

    def is_reverse_pressed(self) -> bool:
        with self._lock:
            return self._buttons["left"]["handoff"]

    def seed_qpos_from_obs(self, obs: dict[str, float]) -> None:
        """Reset internal qpos + gripper + engagement state from a
        `.pos`-keyed dict in FOLLOWER units (joints in degrees,
        gripper 0..100) — either an observation or a commanded action.
        Call before handing control to the operator so the IK anchors to
        the robot's actual pose and the next grip press re-anchors the
        mapper cleanly (see BiQuestTeleoperator.seed_qpos_from_obs for
        the full rationale). Missing keys are silently skipped."""
        with self._lock:
            arm = self._arm
            signs = self.config.joint_signs
            for j, name in enumerate(JOINT_NAMES):
                key = f"{name}.pos"
                if key in obs:
                    arm["qpos"][j] = float(np.radians(signs[j] * obs[key]))
            if "gripper.pos" in obs:
                g_open = float(np.clip(obs["gripper.pos"], 0.0, 100.0)) / 100.0
                arm["trigger"] = 1.0 - g_open  # trigger 1 = closed
                lo, hi = self._jaw_range
                arm["qpos"][ARM_DOFS] = lo + g_open * (hi - lo)
            arm["ramp_active"] = False
            arm["mapper"].disengage()
            arm["engaged"] = False
            arm["last_grip"] = False
            arm["pos_filt"] = None
            arm["quat_filt"] = None
            arm["needs_reanchor"] = False

    def get_action(self) -> dict[str, float]:
        now = time.perf_counter()
        last = self._last_get_action_t
        self._last_get_action_t = now
        if last is not None:
            dt = max(now - last, 1e-3)
            inst_hz = 1.0 / dt
            self._loop_hz = 0.1 * inst_hz + 0.9 * (self._loop_hz or inst_hz)

        with self._lock:
            xr = self._latest_xr_frame
            last_frame_time = self._last_xr_frame_time

        # No xr_frame yet — caller gets the rest pose.
        if xr is None:
            return self._build_action()

        gap_s = time.time() - last_frame_time
        viewer_orient = (xr.get("viewer") or {}).get("orientation")
        yaw_now = _yaw_from_quat_xyzw(viewer_orient) if viewer_orient is not None else None

        ctrls = xr.get("controllers") or {}
        self._update_arm(ctrls.get(self._hand), yaw_now, gap_s)

        action = self._build_action()

        if self.config.publish_ik_state and self._ws is not None and self._ws_loop is not None:
            self._publish_ik_state_async()

        return action

    # ---------- internals ----------

    def _build_action(self) -> dict[str, float]:
        """Solver qpos (radians, model convention) → follower action
        (degrees + 0..100 gripper), applying the per-joint signs.

        Also includes body velocity commands from joystick axes if available.
        """
        arm = self._arm
        signs = self.config.joint_signs
        out: dict[str, float] = {
            f"{name}.pos": float(signs[j] * np.degrees(arm["qpos"][j]))
            for j, name in enumerate(JOINT_NAMES)
        }
        # LeRobot gripper convention: 0..100, 0 = closed. Trigger is 0..1
        # with 1 = squeezed = closed.
        out["gripper.pos"] = float(np.clip((1.0 - arm["trigger"]) * 100.0, 0.0, 100.0))
        
        # Add body velocity commands from joystick (x.y, y.y, theta.vel)
        if hasattr(self, "_last_body_vels") and self._last_body_vels:
            out.update(self._last_body_vels)
        
        return out

    def _anchor_mapper(
        self, arm: dict, pos: np.ndarray, quat_wxyz: np.ndarray, yaw_now: float | None, label: str
    ) -> None:
        """Capture the mapper's engage frame at the current robot+controller
        state. Used by every re-anchor path — rising-edge engage,
        stale-recovery, precision-scale toggle, rest-ramp completion."""
        ee_pos, ee_quat = arm["solver"].fk(arm["qpos"])
        anchor_pos = arm["solver"].wrist_anchor_xpos()
        if yaw_now is not None:
            R_engage = self._r_calib @ _R_y(-yaw_now)
            arm["mapper"].set_R(R_engage)
        arm["mapper"].engage(pos, quat_wxyz, ee_pos, ee_quat, pivot_armbase=anchor_pos)
        arm["engaged"] = True
        logger.debug(
            "%s clutch %s  ee_pos=%s  anchor=%s  yaw_now=%s",
            self._hand,
            label,
            ee_pos.round(3),
            anchor_pos.round(3),
            f"{np.degrees(yaw_now):.1f}°" if yaw_now is not None else "—",
        )

    def _update_arm(self, ctrl: dict | None, yaw_now: float | None, gap_s: float) -> None:
        if ctrl is None:
            return
        arm = self._arm

        # Staleness gate (see module docstring).
        if gap_s > XR_FRAME_STALE_TIMEOUT_S:
            if arm["engaged"] and not arm["needs_reanchor"]:
                arm["needs_reanchor"] = True
                logger.warning("%s xr_frame stale (%.2fs gap) — pausing", self._hand, gap_s)
            arm["haptic"] *= 0.6
            return

        pos_raw = np.asarray(ctrl["position"], dtype=float)
        ox, oy, oz, ow = ctrl["orientation"]  # WebXR sends xyzw
        quat_raw = np.array([ow, ox, oy, oz])

        # EMA smoothing on the controller pose (nlerp with hemisphere check).
        alpha = float(self.config.pose_filter_alpha)
        if arm["pos_filt"] is None or arm["quat_filt"] is None:
            arm["pos_filt"] = pos_raw.copy()
            arm["quat_filt"] = quat_raw.copy()
        else:
            arm["pos_filt"] = (1.0 - alpha) * arm["pos_filt"] + alpha * pos_raw
            q_in = quat_raw if np.dot(arm["quat_filt"], quat_raw) >= 0.0 else -quat_raw
            qf = (1.0 - alpha) * arm["quat_filt"] + alpha * q_in
            arm["quat_filt"] = qf / np.linalg.norm(qf)
            pos = arm["pos_filt"]
            quat_wxyz = arm["quat_filt"]

        buttons = ctrl.get("buttons") or []
        grip = bool(buttons[GRIP_BUTTON_INDEX]["p"]) if len(buttons) > GRIP_BUTTON_INDEX else False
        trigger = float(buttons[TRIGGER_BUTTON_INDEX]["v"]) if len(buttons) > TRIGGER_BUTTON_INDEX else 0.0
        precision = (
            bool(buttons[PRECISION_BUTTON_INDEX]["p"]) if len(buttons) > PRECISION_BUTTON_INDEX else False
        )
        rest_btn = (
            bool(buttons[REST_RAMP_BUTTON_INDEX]["p"]) if len(buttons) > REST_RAMP_BUTTON_INDEX else False
        )
        
        # Read joystick axes for body velocity commands
        axes = ctrl.get("axes")
        self._last_body_vels: dict[str, float] = {}
        if axes:
            from ..joystick_drive import axes_to_body_vel, JoystickDriveConfig
            config = JoystickDriveConfig()
            # Apply precision scaling to axes
            precision_scale = self.config.precision_factor if precision else 1.0
            axes["x"] = axes.get("x", 0.0) * precision_scale
            axes["y"] = axes.get("y", 0.0) * precision_scale
            self._last_body_vels = axes_to_body_vel(axes, config)
        if rest_btn and not arm["last_rest_button"] and not arm["ramp_active"]:
            arm["ramp_start_q"] = arm["qpos"][:ARM_DOFS].copy()
            arm["ramp_target_q"] = np.asarray(self.config.rest_qpos, dtype=float)
            arm["ramp_start_t"] = time.perf_counter()
            arm["ramp_active"] = True
            logger.debug(
                "%s rest-ramp START (duration=%.2fs, target=%s, engaged=%s)",
                self._hand,
                self.config.rest_ramp_duration_s,
                arm["ramp_target_q"].round(3),
                arm["engaged"],
            )
        arm["last_rest_button"] = rest_btn

        # Precision toggle: re-anchor on transition, then scale gains.
        if precision != arm["last_precision"]:
            logger.info(
                "%s precision %s (factor=%.2f, engaged=%s)",
                self._hand, "ON" if precision else "OFF",
                self.config.precision_factor, arm["engaged"],
            )
            if arm["engaged"]:
                self._anchor_mapper(
                    arm, pos, quat_wxyz, yaw_now, "PRECISION" if precision else "FULL-SCALE"
                )
        arm["last_precision"] = precision
        scale_factor = self.config.precision_factor if precision else 1.0
        arm["mapper"].scale = self.config.scale_translation * scale_factor
        arm["mapper"].scale_rotation = self.config.scale_rotation * scale_factor
        arm["mapper"].rot_reach_limit = self.config.rot_reach_limit
        arm["mapper"].pos_reach_limit = self.config.pos_reach_limit

        # Stale-recovery re-anchor.
        if arm["needs_reanchor"] and arm["engaged"] and grip:
            self._anchor_mapper(arm, pos, quat_wxyz, yaw_now, "RE-ANCHOR")
        arm["needs_reanchor"] = False

        # Edge-detect clutch.
        if grip and not arm["last_grip"]:
            self._anchor_mapper(arm, pos, quat_wxyz, yaw_now, "ENGAGE")
        elif not grip and arm["last_grip"]:
            arm["mapper"].disengage()
            arm["engaged"] = False
            arm["pos_filt"] = None
            arm["quat_filt"] = None
            logger.debug("%s clutch RELEASE", self._hand)
        arm["last_grip"] = grip

        # Gripper tracks the controller trigger ONLY while engaged (holds
        # its last value otherwise — no stray closes between corrections).
        if arm["engaged"]:
            arm["trigger"] = trigger
            lo, hi = self._jaw_range
            arm["qpos"][ARM_DOFS] = lo + (1.0 - trigger) * (hi - lo)

        # "Go home" ramp owns qpos while active.
        if arm["ramp_active"]:
            duration = max(1e-3, float(self.config.rest_ramp_duration_s))
            elapsed = time.perf_counter() - arm["ramp_start_t"]
            t = min(1.0, elapsed / duration)
            arm["qpos"][:ARM_DOFS] = arm["ramp_start_q"] + t * (arm["ramp_target_q"] - arm["ramp_start_q"])
            if t >= 1.0:
                arm["ramp_active"] = False
                if arm["engaged"]:
                    self._anchor_mapper(arm, pos, quat_wxyz, yaw_now, "RAMP-DONE")
                else:
                    logger.debug("%s rest-ramp DONE (disengaged)", self._hand)
            arm["haptic"] *= 0.6
            return

        # If engaged, drive arm IK toward the controller-relative target.
        ee_pos_now, ee_quat_now = arm["solver"].fk(arm["qpos"])
        out = arm["mapper"].target(pos, quat_wxyz, ee_pos_now, ee_quat_now)
        if out is not None:
            tgt_pos, tgt_quat = out
            arm["qpos"][:ARM_DOFS] = arm["solver"].solve(tgt_pos, tgt_quat, arm["qpos"])

            # Haptic mix — same four signals as the DK1 teleop. On the
            # SO-101 the gimbal channel is essentially inert (2-DoF wrist,
            # no gimbal lock) but kept for uniformity.
            pressure = float(getattr(arm["solver"], "last_limit_pressure", 0.0))
            pos_err = float(getattr(arm["solver"], "last_pos_err_norm", 0.0))
            singular = float(getattr(arm["solver"], "last_singularity_proximity", 0.0))
            gimbal = float(getattr(arm["solver"], "last_wrist_gimbal_proximity", 0.0))
            i_limit = min(1.0, max(0.0, (pressure - 0.05) / 0.25))
            i_reach = min(1.0, max(0.0, (pos_err - 0.02) / 0.06))
            i_singular = max(0.0, (singular - 0.95) / 0.2)
            i_gimbal = min(1.0, max(0.0, (gimbal - 0.5) / 0.5))
            raw_intensity = max(i_limit, i_reach, i_singular, i_gimbal)
            arm["haptic"] = 0.6 * arm["haptic"] + 0.4 * raw_intensity
        else:
            arm["haptic"] *= 0.6

    def _publish_ik_state_async(self) -> None:
        """Schedule an ik_state send on the WS asyncio loop. The qpos is
        published under `{hand}_qpos` (and the legacy `qpos` compat field)
        so the stock viewer_client.py picks it up with `--arm {hand}`."""
        arm = self._arm
        qpos_list = [float(v) for v in arm["qpos"][:NQ]]
        payload = {
            "type": "ik_state",
            f"{self._hand}_qpos": qpos_list,
            f"{self._hand}_engaged": bool(arm["engaged"]),
            f"{self._hand}_haptic": float(arm["haptic"]),
            f"{self._hand}_force_haptic": float(arm["force_haptic"]),
            "qpos": qpos_list,
            "engaged": bool(arm["engaged"]),
            "teleop_id": str(self.config.id) if self.config.id is not None else None,
            "server_time": time.time(),
            "loop_hz": float(self._loop_hz or 0.0),
        }
        # Idle values for the unused hand so the web client's haptic mixer
        # never sees missing keys.
        other = "left" if self._hand == "right" else "right"
        payload[f"{other}_engaged"] = False
        payload[f"{other}_haptic"] = 0.0
        payload[f"{other}_force_haptic"] = 0.0
        text = json.dumps(payload)

        async def _send():
            try:
                if self._ws is not None:
                    await self._ws.send(text)
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._ws_loop)
        except Exception:
            pass

    # ---------- Live config (from the web UI) ----------

    # Web-tunable knobs and their hard safety bounds — same keys as the
    # bimanual teleop so the stock web Settings panel works unchanged.
    _LIVE_CONFIG_BOUNDS: dict[str, tuple[float, float]] = {
        "scale_translation": (0.1, 10.0),
        "scale_rotation": (0.1, 10.0),
        "pose_filter_alpha": (0.05, 1.0),
        "precision_factor": (0.05, 1.0),
        "force_haptic_threshold_nm_left": (0.0, 1.0),
        "force_haptic_threshold_nm_right": (0.0, 1.0),
        "max_dq_per_joint_scalar_pos": (0.005, 0.50),
        "max_dq_per_joint_scalar_rot": (0.005, 0.50),
    }

    _LIVE_CONFIG_BOOLS: tuple[str, ...] = ("force_haptic_enabled",)

    def _apply_config_update(self, cfg: dict) -> None:
        """Apply a `config_update` payload from the web UI (see
        BiQuestTeleoperator._apply_config_update)."""
        if not isinstance(cfg, dict):
            return
        applied: dict[str, float | bool] = {}
        with self._lock:
            for key, (lo, hi) in self._LIVE_CONFIG_BOUNDS.items():
                if key not in cfg:
                    continue
                try:
                    v = float(cfg[key])
                except (TypeError, ValueError):
                    logger.warning("config_update: %s=%r not a number; ignoring", key, cfg[key])
                    continue
                if not (lo <= v <= hi):
                    logger.warning(
                        "config_update: %s=%.3f out of range [%.2f, %.2f]; clamping", key, v, lo, hi
                    )
                    v = max(lo, min(hi, v))
                setattr(self.config, key, v)
                applied[key] = v
            if (
                "max_dq_per_joint_scalar_pos" in applied
                or "max_dq_per_joint_scalar_rot" in applied
            ):
                pos = float(self.config.max_dq_per_joint_scalar_pos)
                rot = float(self.config.max_dq_per_joint_scalar_rot)
                arr = [pos] * 3 + [rot] * 2
                self.config.max_dq_per_joint = arr
                self._arm["solver"].max_dq_per_joint = np.asarray(arr, dtype=float).copy()
            for key in self._LIVE_CONFIG_BOOLS:
                if key not in cfg:
                    continue
                v_bool = bool(cfg[key])
                setattr(self.config, key, v_bool)
                applied[key] = v_bool
        if applied:
            logger.info(
                "config_update applied: %s",
                ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in applied.items()),
            )

    # ---------- Haptic-threshold calibration (web UI button) ----------

    _HAPTIC_CALIB_MARGIN_NM: float = 0.20
    _HAPTIC_CALIB_MAX_THRESHOLD_NM: float = 1.0
    _HAPTIC_CALIB_MIN_SPAN_NM: float = 0.50
    _HAPTIC_CALIB_SUSPICIOUS_PEAK_NM: float = 0.50

    def _start_haptic_calibration(self, duration_s: float = 5.0) -> None:
        duration_s = max(0.5, min(10.0, float(duration_s)))
        with self._lock:
            if self._haptic_calib is not None:
                logger.info("haptic_calibrate: already running; ignoring duplicate start")
                return
            self._haptic_calib = {
                "start_t": time.time(),
                "duration_s": duration_s,
                "peak": 0.0,
            }
        logger.info("haptic_calibrate: starting %.1fs idle-torque sampling", duration_s)

    def _finalize_haptic_calibration(self) -> None:
        """MUST be called with self._lock held (see _apply_torque_feedback)."""
        calib = self._haptic_calib
        if calib is None:
            return
        peak = float(calib["peak"])
        new_threshold = min(self._HAPTIC_CALIB_MAX_THRESHOLD_NM, peak + self._HAPTIC_CALIB_MARGIN_NM)
        setattr(self.config, f"force_haptic_threshold_nm_{self._hand}", new_threshold)

        required_max = new_threshold + self._HAPTIC_CALIB_MIN_SPAN_NM
        old_max = float(self.config.force_haptic_max_nm)
        new_max = max(old_max, required_max)
        if new_max != old_max:
            self.config.force_haptic_max_nm = new_max

        suspicious = []
        if peak > self._HAPTIC_CALIB_SUSPICIOUS_PEAK_NM:
            suspicious.append(f"{self._hand} peak={peak:.3f} Nm")
            logger.warning(
                "haptic_calibrate: suspicious idle peak — %.3f Nm. Expected "
                "<%.2f Nm at rest with empty jaws; check the gripper isn't "
                "jammed or holding closed against a stop.",
                peak, self._HAPTIC_CALIB_SUSPICIOUS_PEAK_NM,
            )

        self._haptic_calib = None
        logger.info(
            "haptic_calibrate done: peak=%.3f → threshold=%.3f Nm "
            "(margin=%.2f, force_haptic_max_nm=%.2f)",
            peak, new_threshold, self._HAPTIC_CALIB_MARGIN_NM, new_max,
        )
        # Result payload keeps the bimanual schema (the web UI renders
        # both slots); the unused hand mirrors 0.
        payload = {
            "type": "haptic_calibrate_result",
            "left_peak_nm": peak if self._hand == "left" else 0.0,
            "right_peak_nm": peak if self._hand == "right" else 0.0,
            "left_threshold_nm": float(self.config.force_haptic_threshold_nm_left),
            "right_threshold_nm": float(self.config.force_haptic_threshold_nm_right),
            "margin_nm": float(self._HAPTIC_CALIB_MARGIN_NM),
            "max_nm": float(new_max),
            "suspicious": suspicious,
        }
        text = json.dumps(payload)

        async def _send() -> None:
            try:
                if self._ws is not None:
                    await self._ws.send(text)
            except Exception:
                pass

        if self._ws_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(_send(), self._ws_loop)
            except Exception:
                pass

    # ---------- WS thread ----------

    def _ws_thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._ws_loop = loop
        try:
            loop.run_until_complete(self._ws_runner())
        finally:
            loop.close()
            self._ws_loop = None

    def _ssl_context_for(self, url: str):
        if not url.startswith("wss://"):
            return None
        import ssl

        ctx = ssl.create_default_context()
        if "://localhost" in url or "://127.0.0.1" in url:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def _ws_runner(self) -> None:
        backoff = 1.0
        ssl_ctx = self._ssl_context_for(self.config.ws_url)
        while not self._ws_stop.is_set():
            try:
                async with websockets.connect(self.config.ws_url, ssl=ssl_ctx) as ws:
                    self._ws = ws
                    self._ws_connected.set()
                    backoff = 1.0
                    try:
                        await ws.send(json.dumps({"type": "request_settings"}))
                    except Exception as e:
                        logger.debug("request_settings send failed: %s", e)
                    async for raw in ws:
                        if self._ws_stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        mtype = msg.get("type")
                        if mtype == "xr_frame":
                            btn_snapshot = {
                                "left": {"grip": False, "handoff": False},
                                "right": {"grip": False, "handoff": False},
                            }
                            ctrls = msg.get("controllers") or {}
                            for hand in ("left", "right"):
                                ctrl = ctrls.get(hand) or {}
                                buttons = ctrl.get("buttons") or []
                                if len(buttons) > GRIP_BUTTON_INDEX:
                                    btn_snapshot[hand]["grip"] = bool(buttons[GRIP_BUTTON_INDEX].get("p"))
                                if len(buttons) > HANDOFF_BUTTON_INDEX:
                                    btn_snapshot[hand]["handoff"] = bool(
                                        buttons[HANDOFF_BUTTON_INDEX].get("p")
                                    )
                            with self._lock:
                                self._latest_xr_frame = msg
                                self._last_xr_frame_time = time.time()
                                self._buttons = btn_snapshot
                        elif mtype == "config_update":
                            self._apply_config_update(msg.get("config") or {})
                        elif mtype == "haptic_calibrate":
                            self._start_haptic_calibration(
                                float(msg.get("duration_s") or 5.0)
                            )
                        # else: ignore (our own echoed ik_state, pings, etc.)
            except Exception as e:
                logger.warning(
                    "SO101QuestTeleoperator WS error (%s); reconnecting in %.1fs",
                    type(e).__name__, backoff,
                )
            finally:
                self._ws = None
                self._ws_connected.clear()
            if self._ws_stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 10.0)
