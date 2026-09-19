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
from lerobot.cameras.ros2_camera import ROS2Camera, ROS2CameraConfig
from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig

FPS = 30


def main():
    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip="0.0.0.0", id="lekiwi", cameras={
        "front": ROS2CameraConfig(
            color_topic="/camera/color/image_raw",
            depth_topic="/camera/depth/image_raw",
            use_rgb=True,
            use_depth=True,
            fps=30,
            width=480,
            height=640,
            rotation=Cv2Rotation.ROTATE_180
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            fps=30,
            width=480,
            height=640,
            fourcc="MJPG",
            rotation=Cv2Rotation.ROTATE_90,
        ),
    })
    # teleop_arm_config = SO100LeaderConfig(port="/dev/ttyACM0", id="arm")

    teleop_arm_config = SO101QuestTeleoperatorConfig(id="vr-teleop", ws_url="ws://127.0.0.1:8443/ws")
    # keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    leader_arm = SO101QuestTeleoperator(teleop_arm_config)
    # keyboard = KeyboardTeleop(keyboard_config)

    # Connect to the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    leader_arm.connect()
    # keyboard.connect()
    # Init rerun viewer
    init_rerun(session_name="lekiwi_teleop")

    action = {}
    # arm_action = {}
    # if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    #     raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

        # Get teleop action
        # Arm
        action_in = leader_arm.get_action()
        # arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
        
        # Keyboard
        # keyboard_keys = keyboard.get_action()
        # keyboard_action = robot._from_keyboard_to_base_action(keyboard_keys)

        for i in action_in:
            v = action_in[i]
            if '.pos' in i:
                action[f"arm_{i}"] = v
            else:
                # action[i] = max(v,keyboard_action[i])
                action[i] = v
        
        # action = base_action
        # action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        # Send action to robot
        # print(action)
        _ = robot.send_action(action)

        # Visualize
        log_rerun_data(observation=observation, action=action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
