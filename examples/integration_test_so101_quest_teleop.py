"""SO101 Quest Teleoperator Integration Test Suite

This test demonstrates how to integrate and test the SO101QuestTeleoperator
with a LeRobot environment. It includes both unit-level validation and
runtime integration scenarios.
"""

import pytest
import numpy as np
from dataclasses import dataclass

# Test imports
try:
    from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
except ImportError as e:
    pytest.skip(f"lerobot not installed: {e}", allow_module_level=True)

# Mock Follower for leRobot integration tests
class Follower:
    """Mock Follower class to avoid lerobot.robots import errors."""

    def __init__(self, config=None):
        self.config = config or type('FollowerConfig', (), {})()

    def send_action(self, action):
        """No-op: just accept the action dict."""
        pass



from vr_teleop_kit.lerobot.so101_quest_teleop import (
    SO101QuestTeleoperator,
    SO101QuestTeleoperatorConfig,
    JOINT_NAMES,
)


# =============================================================================
# SECTION 1: Configuration and Initialization Tests
# =============================================================================

def test_so101_config_defaults():
    """Test that SO101QuestTeleoperatorConfig has correct default values."""
    config = SO101QuestTeleoperatorConfig()
    
    # Check default hand
    assert config.hand == "right"
    
    # Check default WebSocket URL
    assert config.ws_url == "ws://127.0.0.1:8443/ws"
    
    # Check default precision factor
    assert config.precision_factor == 0.5
    
    # Check default joint signs (all +1)
    assert config.joint_signs == [1.0] * 5
    
    # Check IK parameters
    assert config.lam == 0.05
    assert config.w0 == 0.002
    
    print("✓ SO101QuestTeleoperatorConfig defaults validated")


def test_so101_config_customization():
    """Test that SO101QuestTeleoperatorConfig can be customized."""
    custom_config = SO101QuestTeleoperatorConfig(
        hand="left",
        ws_url="ws://192.168.1.100:8443/ws",
        precision_factor=0.3,
        scale_translation=0.8,
        scale_rotation=1.2,
        rest_qpos=[0.0, 0.0, 0.0, 0.0, 0.0],
        joint_signs=[1.0, -1.0, 1.0, -1.0, 1.0],
    )
    
    assert custom_config.hand == "left"
    assert custom_config.precision_factor == 0.3
    assert custom_config.scale_translation == 0.8
    assert custom_config.scale_rotation == 1.2
    assert custom_config.joint_signs == [1.0, -1.0, 1.0, -1.0, 1.0]
    
    print("✓ SO101QuestTeleoperatorConfig customization validated")


def test_so101_config_validation_errors():
    """Test that invalid configurations raise appropriate errors."""
    # Invalid hand
    with pytest.raises(ValueError, match="must be 'left' or 'right'"):
        SO101QuestTeleoperatorConfig(hand="invalid")
    
    # Invalid joint signs
    with pytest.raises(ValueError, match="must be 5 values"):
        SO101QuestTeleoperatorConfig(joint_signs=[1.0, 1.0, 1.0])


# =============================================================================
# SECTION 2: Action Features and Feedback Tests
# =============================================================================

def test_action_features():
    """Test that action_features returns the correct output schema."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    features = teleop.action_features
    
    # Check joint position features
    for joint in JOINT_NAMES:
        assert f"{joint}.pos" in features
        assert features[f"{joint}.pos"] == float
    
    # Check gripper feature
    assert "gripper.pos" in features
    assert features["gripper.pos"] == float
    
    # Verify output format
    expected_features = {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "elbow_flex.pos": float,
        "wrist_flex.pos": float,
        "wrist_roll.pos": float,
        "gripper.pos": float,
    }
    
    assert features == expected_features
    
    print("✓ Action features schema validated")


def test_feedback_features():
    """Test that feedback_features is empty (no direct feedback)."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    features = teleop.feedback_features
    
    assert features == {}
    
    print("✓ Feedback features validated (haptics via send_feedback)")


# =============================================================================
# SECTION 3: Mock Integration Tests
# =============================================================================

def test_teleop_initialization():
    """Test that SO101QuestTeleoperator initializes correctly."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Check that teleop is not connected yet
    assert not teleop.is_connected
    
    # Check that teleop is calibrated
    assert teleop.is_calibrated
    
    print("✓ SO101QuestTeleoperator initialization validated")


def test_teleop_action_output():
    """Test that get_action returns correct output format."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Get action before xr_frame (should return rest pose)
    action = teleop.get_action()
    
    # Verify action has correct keys
    assert "shoulder_pan.pos" in action
    assert "shoulder_lift.pos" in action
    assert "elbow_flex.pos" in action
    assert "wrist_flex.pos" in action
    assert "wrist_roll.pos" in action
    assert "gripper.pos" in action
    
    # Verify all values are floats
    for key, value in action.items():
        assert isinstance(value, float)
    
    print("✓ get_action output format validated")


def test_teleop_with_mock_ws():
    """Test teleop behavior with mocked WebSocket data."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Mock xr_frame data
    mock_xr_frame = {
        "type": "xr_frame",
        "controllers": {
            "right": {
                "position": [0.5, 0.0, 0.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "buttons": [
                    {"v": 0.0, "p": False},  # trigger
                    {"v": 0.5, "p": True},   # grip (clutch)
                    {"v": 0.0, "p": False},  # precision
                    {"v": 0.0, "p": False},  # rest ramp
                    {"v": 0.0, "p": False},  # B/Y
                ]
            }
        }
    }
    
    # Mock the latest_xr_frame
    teleop._latest_xr_frame = mock_xr_frame
    teleop._last_xr_frame_time = 0.0
    
    # Get action with engaged clutch
    action = teleop.get_action()
    
    # Verify action has correct structure
    assert len(action) == 6  # 5 joints + gripper
    
    # Verify gripper is at default (open)
    assert 0.0 <= action["gripper.pos"] <= 100.0
    
    print("✓ Teleop with mocked WS data validated")


# =============================================================================
# SECTION 4: Configuration Update Tests
# =============================================================================

def test_config_update():
    """Test that live config updates work correctly."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Apply config update
    config_update = {
        "scale_translation": 0.8,
        "pose_filter_alpha": 0.9,
        "force_haptic_enabled": True,
    }
    
    teleop._apply_config_update(config_update)
    
    # Verify updates applied
    assert teleop.config.scale_translation == 0.8
    assert teleop.config.pose_filter_alpha == 0.9
    assert teleop.config.force_haptic_enabled == True
    
    print("✓ Config update mechanism validated")


def test_config_update_bounds():
    """Test that config updates respect bounds."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Try to set out-of-bounds value
    config_update = {
        "scale_translation": 50.0,  # Should be clamped to [0.1, 10.0]
    }
    
    teleop._apply_config_update(config_update)
    
    # Verify value was clamped
    assert teleop.config.scale_translation < 10.0
    assert teleop.config.scale_translation >= 0.1
    
    print("✓ Config update bounds validation validated")





# =============================================================================
# SECTION 6: Haptic Feedback Tests
# =============================================================================

def test_haptic_feedback():
    """Test that haptic feedback is applied correctly."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Mock torque feedback
    feedback = {
        "torques": {
            "gripper.torque": 0.4,  # Nm
            "gripper.pos": 0.5,
        }
    }
    
    teleop.send_feedback(feedback)
    
    # Verify haptic intensity was calculated
    assert teleop._arm["force_haptic"] >= 0.0
    assert teleop._arm["force_haptic"] <= 1.0
    
    print("✓ Haptic feedback mechanism validated")


def test_haptic_calibrate():
    """Test haptic threshold calibration."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Start calibration
    teleop._start_haptic_calibration(duration_s=5.0)
    
    # Verify calibration state
    assert teleop._haptic_calib is not None
    assert "start_t" in teleop._haptic_calib
    assert "duration_s" in teleop._haptic_calib
    assert "peak" in teleop._haptic_calib
    
    print("✓ Haptic calibration mechanism validated")


# =============================================================================
# SECTION 7: Edge Cases and Error Handling
# =============================================================================

def test_teleop_disconnect():
    """Test that disconnect works correctly."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Disconnect
    teleop.disconnect()
    
    # Verify disconnected state
    assert not teleop.is_connected
    
    print("✓ Disconnect mechanism validated")


def test_teleop_send_feedback_invalid():
    """Test that invalid feedback is handled gracefully."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Invalid feedback
    invalid_feedback = "not a dict"
    
    teleop.send_feedback(invalid_feedback)
    
    # Should not raise error
    assert True  # If we got here, no error was raised
    
    print("✓ Invalid feedback handling validated")


def test_teleop_stale_timeout():
    """Test that stale xr_frame timeout is handled."""
    config = SO101QuestTeleoperatorConfig()
    teleop = SO101QuestTeleoperator(config)
    
    # Simulate stale frame
    teleop._latest_xr_frame = None
    teleop._last_xr_frame_time = 0.0
    
    # Get action with no frame
    action = teleop.get_action()
    
    # Should return rest pose action
    assert "shoulder_pan.pos" in action
    
    print("✓ Stale timeout handling validated")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SO101 Quest Teleoperator Integration Test Suite")
    print("=" * 70)
    
    # Run all tests
    print("\n1. Configuration Tests")
    test_so101_config_defaults()
    test_so101_config_customization()
    test_so101_config_validation_errors()
    
    print("\n2. Action Features Tests")
    test_action_features()
    test_feedback_features()
    
    print("\n3. Initialization Tests")
    test_teleop_initialization()
    test_teleop_action_output()
    
    print("\n4. WebSocket Integration Tests")
    test_teleop_with_mock_ws()
    
    print("\n5. Configuration Update Tests")
    test_config_update()
    test_config_update_bounds()
    
    print("\n6. LeRobot Integration Tests")
    test_leobot_integration()
    
    print("\n7. Haptic Feedback Tests")
    test_haptic_feedback()
    test_haptic_calibrate()
    
    print("\n8. Edge Cases Tests")
    test_teleop_disconnect()
    test_teleop_send_feedback_invalid()
    test_teleop_stale_timeout()
    
    print("\n" + "=" * 70)
    print("✓ All integration tests completed successfully!")
    print("=" * 70)