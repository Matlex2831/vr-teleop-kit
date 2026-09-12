# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import sys
sys.path.append('../../')
from vr_teleop_kit.lerobot import SO101QuestTeleoperator, SO101QuestTeleoperatorConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip="matthew.lan", id="lekiwi")
    # teleop_arm_config = SO100LeaderConfig(port="/dev/ttyACM0", id="arm")

    teleop_arm_config = SO101QuestTeleoperatorConfig(id="vr-teleop", ws_url="ws://127.0.0.1:8443/ws", urdf_path=r"C:\Users\matth\OneDrive\Documents\SO-ARM100\Simulation\SO101\so101_new_calib.urdf")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    leader_arm = SO101QuestTeleoperator(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Connect to the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    leader_arm.connect()
    keyboard.connect()
    # arm_action = keyboard.get_action()
    # Init rerun viewer
    init_rerun(session_name="lekiwi_teleop")

    # if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    #     raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

        # Get teleop action
        # Arm
        # arm_action = {shoulder_pan: 45.0, shoulder_lift: -20.0, elbow: -30.0, wrist_pitch: 10.0, wrist_roll: 0.0, gripper: 0.8}
        arm_action = leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
        # Keyboard
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        # Send action to robot
        _ = robot.send_action(action)

        # Visualize
        log_rerun_data(observation=observation, action=action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
