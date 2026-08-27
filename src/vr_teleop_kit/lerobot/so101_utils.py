"""Follower-side SO-101 helpers shared by the examples, the tools and
the web UI's rollout launcher.

The pre-flight sequence lives here rather than in `examples/` because it
is not example code: it is the sequence that keeps the arm from snapping
to a stale goal or driving into a limit, and every path that enables
torque needs it — including the rollout the /data page can start, which
cannot import from `examples/`.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTOR_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)


def presync_goal_positions(port: str) -> None:
    """Write Goal_Position := Present_Position (raw counts) for every motor
    while torque is still OFF — call this BEFORE `SOFollower.connect()`.

    Why: Feetech Goal_Position is a RAM register that persists across
    host processes for as long as the servo bus is powered. Whatever the
    previous session last commanded is still sitting there, and the
    instant lerobot's configure() enables torque, each servo snaps to
    that stale goal at full speed — the "arm dives violently the moment
    I press ENTER" failure. Syncing the goals to the present positions
    first makes torque-enable a no-op: the arm simply locks where it is.

    Uses a calibration-free raw bus (counts in, counts out), median of 3
    reads per motor to ride out the occasional corrupted status packet.
    """
    motors = {
        name: Motor(i + 1, "sts3215", MotorNormMode.DEGREES)
        for i, name in enumerate(MOTOR_NAMES)
    }
    bus = FeetechMotorsBus(port, motors)  # no calibration: raw counts only
    bus.connect()
    try:
        reads = [bus.sync_read("Present_Position", normalize=False) for _ in range(3)]
        goal = {n: sorted(r[n] for r in reads)[1] for n in MOTOR_NAMES}
        bus.sync_write("Goal_Position", goal, normalize=False)
    finally:
        bus.disconnect()  # torque was already off; leaves it off


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def parse_rest_pose_env(env_var: str, fallback: list[float]) -> list[float]:
    """Parse SO101_REST_POSE: five comma-separated joint angles in
    radians (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll). Unset or empty → fallback."""
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return list(fallback)
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != 5:
        raise ValueError(
            f"{env_var}={raw!r} must contain 5 comma-separated radians "
            f"(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll); "
            f"got {len(parts)}"
        )
    return parts


def build_rest_action(rest_rad: list[float], signs: list[float]) -> dict[str, float]:
    """Follower action dict for the rest pose: joints in DEGREES
    (`use_degrees=True`, model→follower signs applied), gripper open (100)."""
    action = {f"{n}.pos": float(s * np.degrees(v))
              for n, v, s in zip(JOINT_NAMES, rest_rad, signs)}
    action["gripper.pos"] = 100.0
    return action


def check_calibration_plausible(follower, logger: logging.Logger) -> list[str]:
    """Cross-check the follower's calibration file against the URDF joint
    ranges before anything moves.

    Hard failures (returned, abort): a joint missing or barely swept.

    A wrapped range (span ≈ 360° on a joint that travels ~200°) is only a
    WARNING: it happens whenever the sweep crosses the 12-bit encoder wrap,
    which a fully-folding joint (shoulder_lift / elbow_flex reach past
    ±180° from center at their hard stops) does even with a perfect middle
    pose. A wrap with a correct middle pose still yields correct zeros
    (mid ≈ homing pose); a wrap with a WRONG middle pose yields garbage —
    and the two are indistinguishable in the file, so the decisive test is
    the first-observation gate after connect (readings must land inside
    the URDF limits), plus tools/so101_readout.py in the operator's hands."""
    URDF_SPAN_DEG = {
        "shoulder_pan": 220.0, "shoulder_lift": 200.0, "elbow_flex": 193.6,
        "wrist_flex": 190.0, "wrist_roll": 320.0,
    }
    problems = []
    calib = getattr(follower, "calibration", None) or {}
    for name, span_urdf in URDF_SPAN_DEG.items():
        c = calib.get(name)
        if c is None:
            problems.append(f"{name}: missing from calibration")
            continue
        span_deg = (c.range_max - c.range_min) * 360.0 / 4096.0
        if span_deg > span_urdf + 30.0:
            logger.warning(
                "calibration check: %s recorded range %.0f° exceeds the physical "
                "~%.0f° (encoder wrap during the sweep). Zeros are still fine IF "
                "the middle pose was held at the first ENTER — the "
                "first-observation gate below decides; verify with "
                "tools/so101_readout.py.", name, span_deg, span_urdf,
            )
        elif span_deg < 60.0:
            problems.append(f"{name}: recorded range only {span_deg:.0f}° — joint barely swept")
    for p in problems:
        logger.error("calibration check: %s", p)
    return problems


def get_observation_median(follower, n: int = 3) -> dict[str, float]:
    """Median of n observations per numeric key. Feetech buses can return
    a corrupted Present_Position, and a joint near the 12-bit encoder
    wrap can teleport ±360° between reads — never anchor a relative move
    on a single read."""
    obses = [follower.get_observation() for _ in range(n)]
    out = dict(obses[-1])
    for k in obses[0]:
        vals = [o[k] for o in obses if isinstance(o.get(k), (int, float))]
        if len(vals) == n:
            out[k] = sorted(vals)[n // 2]
    return out


def ramp_to_rest(
    follower,
    target: dict[str, float],
    duration_s: float,
    steps: int,
    logger: logging.Logger,
) -> None:
    """Smoothly drive the follower from its current pose to `target` in
    linearly-interpolated increments. `duration_s` is a MINIMUM: the
    actual duration stretches so no joint is scheduled faster than
    MAX_RAMP_DEG_S — a fixed-time schedule from a collapsed pose would
    otherwise sweep 90°+ at whatever speed 3 s implies."""
    MAX_RAMP_DEG_S = 25.0
    obs = get_observation_median(follower)
    start = {k: float(obs.get(k, target[k])) for k in target}

    worst_delta = max(abs(target[k] - start[k]) for k in target)
    duration_s = max(duration_s, worst_delta / MAX_RAMP_DEG_S)
    steps = max(steps, int(duration_s * 30))
    logger.info("ramp start : %s", {k: round(v, 1) for k, v in start.items()})
    logger.info("ramp target: %s", {k: round(v, 1) for k, v in target.items()})
    logger.info("ramping to rest pose over %.1fs in %d steps (worst joint Δ %.1f°)",
                duration_s, steps, worst_delta)
    dt = duration_s / max(1, steps)
    for i in range(1, steps + 1):
        alpha = i / steps
        command = {k: start[k] + alpha * (target[k] - start[k]) for k in target}
        follower.send_action(command)
        time.sleep(dt)
    logger.info("rest pose reached")
