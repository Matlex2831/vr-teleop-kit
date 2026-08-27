"""WebXR (Meta Quest) teleoperation stack for robot manipulators
(TRLC-DK1 bimanual, SO-101 single arm).

Layers (see the README for the architecture):
  vr_teleop_kit.core    — robot-agnostic clutch-relative pose mapping
                          with absorbing reach limits
  vr_teleop_kit.ik      — per-robot decoupled IK (DK1: model.py +
                          decoupled_ik.py; SO-101: so101_model.py +
                          so101_ik.py)
  vr_teleop_kit.relay   — FastAPI WebSocket relay + WebRTC cameras +
                          the WebXR page served to the Quest
  vr_teleop_kit.lerobot — LeRobot ``Teleoperator`` adapters (import
                          this subpackage to register the plugin types;
                          requires ``lerobot`` installed)
"""

__version__ = "0.1.0"
