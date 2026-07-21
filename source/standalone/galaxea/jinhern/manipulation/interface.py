"""Backend-agnostic manipulator interface.

Conventions shared by every implementation:

- Positions are 3-vectors ``[x, y, z]`` in meters, expressed in the robot's
  base frame (Isaac: env-local frame; real R1: the frame the Arm Pose Control
  node expects, i.e. ``gripper_link`` relative to ``torso_link4``).
- Orientations are unit quaternions ``[w, x, y, z]`` (Isaac Lab order). The
  ROS backend converts to ``[x, y, z, w]`` internally.
- All values cross this interface as plain sequences / numpy arrays. No torch,
  no ROS types.

The interface is deliberately non-blocking: command methods only update the
target, ``step()`` advances the backend by one control tick, and the reach
checks report progress. A state machine calls these once per tick.
"""

from abc import ABC, abstractmethod

import numpy as np

LEFT = "left"
RIGHT = "right"


def _require_finite(name, value):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Expected finite {name}, got {value}")


def as_position(value):
    position = np.asarray(value, dtype=np.float64).reshape(-1)
    if position.shape != (3,):
        raise ValueError(f"Expected position [x, y, z], got shape {position.shape}")
    _require_finite("position [x, y, z]", position)
    return position


def as_quaternion(value):
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,):
        raise ValueError(
            f"Expected quaternion [w, x, y, z], got shape {quaternion.shape}"
        )
    _require_finite("quaternion [w, x, y, z]", quaternion)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("Quaternion has near-zero norm")
    return quaternion / norm


def orientation_error(quaternion_a, quaternion_b):
    """Distance between two rotations in [0, 1]; 0 when identical.

    Uses ``1 - |a . b|`` so that q and -q (the same rotation) compare equal.
    """
    return 1.0 - abs(float(np.dot(as_quaternion(quaternion_a), as_quaternion(quaternion_b))))


class ManipulatorInterface(ABC):
    """Minimal dual-arm manipulator API shared by sim and real backends."""

    # -- commands ----------------------------------------------------------

    @abstractmethod
    def move_to(self, arm, position, orientation=None):
        """Command one arm's end effector toward a pose. Non-blocking."""

    @abstractmethod
    def set_gripper(self, arm, open_gripper):
        """Command one gripper fully open (True) or closed (False)."""

    # -- feedback ----------------------------------------------------------

    @abstractmethod
    def get_ee_pose(self, arm):
        """Return the measured end-effector pose ``(position, quaternion)``.

        Returns ``(None, None)`` while no feedback is available yet.
        """

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    def step(self):
        """Advance one control tick (sim step / ROS spin + rate sleep)."""

    @property
    @abstractmethod
    def step_dt(self):
        """Duration of one ``step()`` in seconds."""

    @property
    def is_running(self):
        """False once the backend wants to stop (app closed / ROS shutdown)."""
        return True

    def shutdown(self):
        """Release backend resources. Safe to call more than once."""

    def abort(self):
        """Best-effort stop/hold hook used when a skill fails."""

    # -- reach checks ------------------------------------------------------

    def reached(
        self,
        arm,
        position,
        orientation=None,
        position_tolerance=0.01,
        orientation_tolerance=0.03,
    ):
        """True when the measured EE pose is within tolerance of a target."""
        actual_position, actual_orientation = self.get_ee_pose(arm)
        if actual_position is None:
            return False

        position_error = float(np.linalg.norm(as_position(position) - actual_position))
        if position_error >= position_tolerance:
            return False

        if orientation is None:
            return True
        if actual_orientation is None:
            return False
        return orientation_error(orientation, actual_orientation) < orientation_tolerance

    def command_reached(
        self,
        arm,
        position,
        orientation=None,
        position_tolerance=0.05,
        orientation_tolerance=0.05,
    ):
        """True when the *commanded* target has converged on a waypoint.

        Backends that stream a smoothed command (Isaac) override this to check
        the generated command instead of the measured pose, which lets an
        approach waypoint hand over to the next state even if the physical IK
        retains a small error. The default simply defers to ``reached()``.
        """
        return self.reached(
            arm,
            position,
            orientation,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )

    def pose_error(self, arm, position, orientation=None):
        """Measured (position_error_m, orientation_error) for logging."""
        actual_position, actual_orientation = self.get_ee_pose(arm)
        if actual_position is None:
            return float("nan"), float("nan")
        position_error = float(np.linalg.norm(as_position(position) - actual_position))
        if orientation is None or actual_orientation is None:
            return position_error, 0.0
        return position_error, orientation_error(orientation, actual_orientation)

    # -- convenience wrappers (the API requested by callers) ---------------

    def move_left_to(self, position, orientation=None):
        self.move_to(LEFT, position, orientation)

    def move_right_to(self, position, orientation=None):
        self.move_to(RIGHT, position, orientation)

    def open_left_gripper(self):
        self.set_gripper(LEFT, True)

    def close_left_gripper(self):
        self.set_gripper(LEFT, False)

    def open_right_gripper(self):
        self.set_gripper(RIGHT, True)

    def close_right_gripper(self):
        self.set_gripper(RIGHT, False)
