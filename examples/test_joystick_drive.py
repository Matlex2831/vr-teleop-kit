"""Test script for JoystickDrive velocity output."""

from vr_teleop_kit.lerobot.so101_quest_teleop import SO101QuestTeleoperator, SO101QuestTeleoperatorConfig
from lerobot.utils.robot_utils import precise_sleep
import time
FPS = 30

teleop_arm_config = SO101QuestTeleoperatorConfig(id="vr-teleop", ws_url="ws://127.0.0.1:8443/ws", urdf_path=r"C:\Users\matth\OneDrive\Documents\SO-ARM100\Simulation\SO101\so101_new_calib.urdf")
leader_arm = SO101QuestTeleoperator(teleop_arm_config)
while True:
    t0 = time.perf_counter()
    print(t0)

    arm_action = leader_arm.get_action()
    print(arm_action)

    precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
