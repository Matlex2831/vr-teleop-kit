"""Pure-simulation SO-101 VR teleop — no hardware follower.

Runs `SO101QuestTeleoperator` against the Quest pose stream and
publishes `ik_state` back to the relay so `tools/viewer_client.py
--robot so101` can render the resulting qpos in mujoco. Same IK pipeline
as `teleop_so101.py` minus the SO101Follower / serial handling — useful
for previewing the rest pose and testing IK behavior with the physical
arm powered off.

Quick test workflow (no robot needed; SO101_URDF must point at the URDF):
  1. Relay server up:        vr-teleop-relay
  2. Mujoco viewer up:       python tools/viewer_client.py --robot so101
  3. This pure-sim loop:     python examples/pure_sim_so101.py
  4. Quest browser → http://localhost:8443/ (USB) → Start Teleop →
     squeeze the right grip, watch the arm move in the mujoco viewer.
"""

from __future__ import annotations

import argparse
import logging
import time

from vr_teleop_kit.ik.so101_model import DEFAULT_Q_REST
from vr_teleop_kit.lerobot.so101_quest_teleop import (
    SO101QuestTeleoperator,
    SO101QuestTeleoperatorConfig,
)
from vr_teleop_kit.lerobot.cli import add_ik_cli_args, ik_kwargs_from_args


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws")
    ap.add_argument("--hand", choices=("left", "right"), default="right")
    ap.add_argument("--freq", type=int, default=60)
    ap.add_argument("--urdf-path", default="")
    add_ik_cli_args(ap)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    teleop = SO101QuestTeleoperator(SO101QuestTeleoperatorConfig(
        id="pure-sim-so101",
        ws_url=args.ws_url,
        hand=args.hand,
        urdf_path=args.urdf_path,
        rest_qpos=DEFAULT_Q_REST.tolist(),
        **ik_kwargs_from_args(args),
    ))
    teleop.connect()
    logger.info("pure-sim SO-101 loop at %d Hz (hand=%s); Ctrl-C to stop",
                args.freq, args.hand)
    period = 1.0 / args.freq
    try:
        next_tick = time.perf_counter()
        while True:
            teleop.get_action()  # runs IK; publishes ik_state to the relay
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()
