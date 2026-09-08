"""Joystick-driven base velocity mapper for VR teleoperators."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class JoystickDriveConfig:
    """Configuration for joystick→body velocity mapping.
    
    Attributes:
        max_velocity: Maximum linear velocity magnitude (m/s)
        max_angular_velocity: Maximum angular velocity (rad/s)
        deadzone: Deadzone threshold for stick inputs
        deadzone_multiplier: Factor to scale deadzone based on stick position
        precision_button_index: Index of button for precision modifier
        precision_factor:
</think>

"""Joystick-driven base velocity mapper for VR teleoperators."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class JoystickDriveConfig:
    """Configuration for joystick→body velocity mapping.

    Attributes:
        max_velocity: Maximum linear velocity magnitude (m/s)
        max_angular_velocity: Maximum angular velocity (rad/s)
        deadzone: Deadzone threshold for stick inputs
        deadzone_multiplier: Factor to scale deadzone based on stick position
        precision_button_index: Index of button for precision modifier
        precision_factor: Factor to scale velocities when precision button is pressed
    """
    max_velocity: float = 1.0  # m/s
    max_angular_velocity: float = 0.5  # rad/s
    deadzone: float = 0.1
    deadzone_multiplier: float = 1.0
    precision_button_index: int = 4
    precision_factor: float = 0.25


class JoystickDrive:
    """Joystick-driven base velocity mapper for VR teleoperators.

    Maps thumbstick axes to linear and angular velocities using the same API as
    the guidance module, but placed under lerobot/ for consistency with the repo
    organization.
    """

    def __init__(self, config: Optional[JoystickDriveConfig] = None) -> None:
        """Initialize the joystick drive mapper.

        Args:
            config: JoystickDriveConfig instance with mapping parameters
        """
        self.config = config or JoystickDriveConfig()
        self._precision_mode = False
    
    def _dz(self, input_val: float, deadzone: float = None, multiplier: float = 1.0) -> float:
        """Apply deadzone to input value.

        Args:
            input_val: Raw input value from joystick
            deadzone: Deadzone threshold (defaults to config value)
            multiplier: Factor to scale deadzone (for precision mode)

        Returns:
            Deadzoned input value
        """
        dz = (deadzone or self.config.deadzone) * multiplier
        if abs(input_val) <= dz:
            return 0.0
        return max(-input_val, input_val - dz)

    def axes_to_body_vel(
        self,
        x: float,
        y: float,
        theta: float,
        precision_mode: bool = False,
    ) -> dict[str, float]:
        """Convert joystick axes to body velocity commands.

        This follows the same API as the guidance module, mapping:
        - x (non-driving hand thumbstick) → x.vel (linear velocity)
        - y (non-driving hand thumbstick) → y.vel (linear velocity)
        - theta (driving hand X thumbstick) → theta.vel (angular velocity)

        Args:
            x: X-axis value from non-driving hand thumbstick (range: -1.0 to 1.0)
            y: Y-axis value from non-driving hand thumbstick (range: -1.0 to 1.0)
            theta: X-axis value from driving hand thumbstick (range: -1.0 to 1.0)
            precision_mode: Whether precision modifier button is pressed

        Returns:
            Dictionary with velocity commands:
            - x.vel: Linear X velocity (m/s)
            - y.vel: Linear Y velocity (m/s)
            - theta.vel: Angular velocity (rad/s)
        """
        # Apply deadzone to all inputs
        x = self._dz(x, self.config.deadzone, self.config.deadzone_multiplier)
        y = self._dz(y, self.config.deadzone, self.config.deadzone_multiplier)
        theta = self._dz(theta, self.config.deadzone, self.config.deadzone_multiplier)

        # Apply precision modifier if enabled
        if precision_mode:
            x *= self.config.precision_factor
            y *= self.config.precision_factor
            theta *= self.config.precision_factor

        # Calculate velocity commands
        x_vel = self.config.max_velocity * x
        y_vel = self.config.max_velocity * y
        theta_vel = self.config.max_angular_velocity * theta
        return {
        "x.vel": x_vel,
        "y.vel": y_vel,
        "theta.vel": theta_vel,
    }