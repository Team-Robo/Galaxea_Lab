"""Real Galaxea R1 implementation of ManipulatorInterface (ROS 2, rclpy).

Run this with the system Python that has ROS 2 sourced (the Galaxea R1 SDK
targets ROS 2 Humble; topics are plain pub/sub so a Jazzy client works as
long as both sides see the same DDS domain).

The robot's Arm Pose Control node (``mobiman``) runs the MPC on-board: we
only publish a target end-effector pose and it generates the joint motion.
This module does not perform collision checking or motion planning.

Default topics (verify with ``ros2 topic list`` on your unit — names have
shifted between SDK releases; every one is a constructor argument):

    /motion_target/target_pose_arm_left    geometry_msgs/PoseStamped  (input)
    /motion_target/target_pose_arm_right   geometry_msgs/PoseStamped  (input)
    /motion_target/target_position_gripper_left    std_msgs/Float32   (input)
    /motion_target/target_position_gripper_right   std_msgs/Float32   (input)
    /motion_control/pose_ee_arm_left       geometry_msgs/PoseStamped  (feedback)
    /motion_control/pose_ee_arm_right      geometry_msgs/PoseStamped  (feedback)

Frame convention: per the Galaxea docs the target pose is the transform of
``gripper_link`` relative to ``torso_link4``. Targets you send must be
expressed in that frame — Isaac env-local coordinates are NOT the same frame;
calibrate the offset before reusing sim waypoints.

Quaternions cross the ManipulatorInterface as ``[w, x, y, z]`` and are
converted to ROS ``[x, y, z, w]`` here.

Open-loop progression is only allowed in ``dry_run`` mode, where pose and
gripper targets are logged but not published.
"""

import time

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float32

from .interface import LEFT, RIGHT, ManipulatorInterface, as_position, as_quaternion


class RealRobotManipulator(ManipulatorInterface):
    """Publishes pose / gripper targets to the R1's on-board controllers."""

    def __init__(
        self,
        node_name="r1_manipulation_client",
        rate_hz=20.0,
        frame_id="torso_link4",
        left_pose_topic="/motion_target/target_pose_arm_left",
        right_pose_topic="/motion_target/target_pose_arm_right",
        left_gripper_topic="/motion_target/target_position_gripper_left",
        right_gripper_topic="/motion_target/target_position_gripper_right",
        left_ee_feedback_topic="/motion_control/pose_ee_arm_left",
        right_ee_feedback_topic="/motion_control/pose_ee_arm_right",
        gripper_open_value=100.0,
        gripper_close_value=0.0,
        open_loop_wait_s=0.0,
        feedback_timeout_s=0.5,
        dry_run=False,
        min_command_z=None,
        max_command_delta=None,
    ):
        """
        Args:
            rate_hz: control-loop rate; ``step()`` sleeps to hold this rate.
            gripper_open_value / gripper_close_value: values published on the
                gripper topics (docs specify a 0-100 range, 100 = open).
            open_loop_wait_s: if > 0 and no EE-pose feedback has arrived,
                ``reached()`` falls back to "target unchanged for this many
                seconds". This is only allowed with ``dry_run=True`` because
                position tolerances are NOT enforced in that mode.
            feedback_timeout_s: maximum age of EE feedback used for reach checks.
            dry_run: when True, build commands and state transitions but do not
                publish pose or gripper targets.
            min_command_z: optional lower bound for commanded target z.
            max_command_delta: optional maximum Cartesian jump from the last
                command, or from fresh feedback for the first command.
        """
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be finite and > 0, got {rate_hz!r}")
        if not np.isfinite(open_loop_wait_s) or open_loop_wait_s < 0.0:
            raise ValueError(
                "open_loop_wait_s must be finite and >= 0, "
                f"got {open_loop_wait_s!r}"
            )
        if open_loop_wait_s > 0.0 and not dry_run:
            raise ValueError("open_loop_wait_s is only allowed with dry_run=True")
        if not np.isfinite(feedback_timeout_s) or feedback_timeout_s <= 0.0:
            raise ValueError(
                "feedback_timeout_s must be finite and > 0, "
                f"got {feedback_timeout_s!r}"
            )
        if not np.isfinite(gripper_open_value) or not np.isfinite(gripper_close_value):
            raise ValueError("gripper command values must be finite")
        if min_command_z is not None and not np.isfinite(min_command_z):
            raise ValueError(f"min_command_z must be finite, got {min_command_z!r}")
        if max_command_delta is not None:
            if not np.isfinite(max_command_delta) or max_command_delta <= 0.0:
                raise ValueError(
                    "max_command_delta must be finite and > 0, "
                    f"got {max_command_delta!r}"
                )

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.node = Node(node_name)
        self.frame_id = frame_id
        self.rate_hz = float(rate_hz)
        self.dry_run = bool(dry_run)
        self.gripper_values = {
            True: float(gripper_open_value),
            False: float(gripper_close_value),
        }
        self.open_loop_wait_s = float(open_loop_wait_s)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.min_command_z = None if min_command_z is None else float(min_command_z)
        self.max_command_delta = (
            None if max_command_delta is None else float(max_command_delta)
        )

        self._pose_publishers = {
            LEFT: self.node.create_publisher(PoseStamped, left_pose_topic, 10),
            RIGHT: self.node.create_publisher(PoseStamped, right_pose_topic, 10),
        }
        self._gripper_publishers = {
            LEFT: self.node.create_publisher(Float32, left_gripper_topic, 10),
            RIGHT: self.node.create_publisher(Float32, right_gripper_topic, 10),
        }
        self._feedback = {LEFT: None, RIGHT: None}  # (position, quat_wxyz)
        self._feedback_received_at = {LEFT: None, RIGHT: None}
        self.node.create_subscription(
            PoseStamped, left_ee_feedback_topic, self._make_feedback_callback(LEFT), 10
        )
        self.node.create_subscription(
            PoseStamped, right_ee_feedback_topic, self._make_feedback_callback(RIGHT), 10
        )

        # Last commanded target and when it last *changed* (for open loop).
        self._last_command = {LEFT: None, RIGHT: None}  # (position, quat_wxyz)
        self._last_gripper = {LEFT: None, RIGHT: None}
        self._command_changed_at = {LEFT: None, RIGHT: None}
        self._warned_no_feedback = {LEFT: False, RIGHT: False}
        self._warned_stale_feedback = {LEFT: False, RIGHT: False}
        self._warned_frame_mismatch = {LEFT: False, RIGHT: False}
        self._next_step_time = None
        self._closed = False

    def _make_feedback_callback(self, arm):
        def callback(msg):
            feedback_frame = msg.header.frame_id
            if feedback_frame and feedback_frame != self.frame_id:
                if not self._warned_frame_mismatch[arm]:
                    self._warned_frame_mismatch[arm] = True
                    self.node.get_logger().warning(
                        f"Ignoring {arm} feedback in frame {feedback_frame!r}; "
                        f"expected {self.frame_id!r}."
                    )
                return

            position = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            )
            orientation = as_quaternion(
                [
                    msg.pose.orientation.w,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                ]
            )
            self._feedback[arm] = (position, orientation)
            self._feedback_received_at[arm] = time.monotonic()
            self._warned_no_feedback[arm] = False
            self._warned_stale_feedback[arm] = False

        return callback

    def _feedback_is_fresh(self, arm):
        received_at = self._feedback_received_at[arm]
        if self._feedback[arm] is None or received_at is None:
            return False
        age_s = time.monotonic() - received_at
        if age_s <= self.feedback_timeout_s:
            return True
        if not self._warned_stale_feedback[arm]:
            self._warned_stale_feedback[arm] = True
            self.node.get_logger().warning(
                f"Ignoring stale {arm} feedback: age {age_s:.3f}s exceeds "
                f"{self.feedback_timeout_s:.3f}s."
            )
        return False

    def _validate_command(self, arm, position):
        if self.min_command_z is not None and position[2] < self.min_command_z:
            raise ValueError(
                f"{arm} target z={position[2]:.3f} is below configured "
                f"minimum {self.min_command_z:.3f}"
            )

        if self.max_command_delta is None:
            return
        reference = None
        if self._last_command[arm] is not None:
            reference = self._last_command[arm][0]
        elif self._feedback_is_fresh(arm):
            reference = self._feedback[arm][0]
        if reference is None:
            return
        delta = float(np.linalg.norm(position - reference))
        if delta > self.max_command_delta:
            raise ValueError(
                f"{arm} target jump {delta:.3f}m exceeds configured maximum "
                f"{self.max_command_delta:.3f}m"
            )

    # -- ManipulatorInterface ----------------------------------------------

    def move_to(self, arm, position, orientation=None):
        position = as_position(position)
        if orientation is not None:
            orientation = as_quaternion(orientation)
        else:
            orientation = self._fallback_orientation(arm)
        self._validate_command(arm, position)

        message = PoseStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        # Interface uses [w, x, y, z]; ROS wants x, y, z, w.
        message.pose.orientation.w = float(orientation[0])
        message.pose.orientation.x = float(orientation[1])
        message.pose.orientation.y = float(orientation[2])
        message.pose.orientation.z = float(orientation[3])

        previous = self._last_command[arm]
        command_changed = (
            previous is None
            or not np.allclose(previous[0], position, atol=1e-6)
            or not np.allclose(previous[1], orientation, atol=1e-6)
        )
        if self.dry_run:
            if command_changed:
                self.node.get_logger().info(
                    f"[dry-run] {arm} pose target "
                    f"pos={position.tolist()} quat_wxyz={orientation.tolist()}"
                )
        else:
            self._pose_publishers[arm].publish(message)

        if command_changed:
            self._command_changed_at[arm] = time.monotonic()
        self._last_command[arm] = (position, orientation)

    def _fallback_orientation(self, arm):
        """Orientation to publish when the caller did not provide one."""
        if self._last_command[arm] is not None:
            return self._last_command[arm][1]
        if self._feedback_is_fresh(arm):
            return self._feedback[arm][1]
        raise ValueError(
            f"move_to({arm!r}) needs an orientation: none was given, none was "
            "commanded before, and no fresh end-effector feedback has arrived yet."
        )

    def set_gripper(self, arm, open_gripper):
        message = Float32()
        message.data = self.gripper_values[bool(open_gripper)]
        if self.dry_run:
            if self._last_gripper[arm] != message.data:
                self.node.get_logger().info(
                    f"[dry-run] {arm} gripper target {message.data:.3f}"
                )
        else:
            self._gripper_publishers[arm].publish(message)
        self._last_gripper[arm] = message.data

    def get_ee_pose(self, arm):
        if not self._feedback_is_fresh(arm):
            return None, None
        position, orientation = self._feedback[arm]
        return position.copy(), orientation.copy()

    def reached(
        self,
        arm,
        position,
        orientation=None,
        position_tolerance=0.01,
        orientation_tolerance=0.03,
    ):
        if self._feedback_is_fresh(arm):
            return super().reached(
                arm,
                position,
                orientation,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )

        if not self._warned_no_feedback[arm]:
            self._warned_no_feedback[arm] = True
            self.node.get_logger().warning(
                f"No fresh end-effector feedback for {arm} arm; "
                + (
                    f"dry-run open-loop wait is {self.open_loop_wait_s}s."
                    if self.open_loop_wait_s > 0
                    else "reached() will stay False until feedback arrives."
                )
            )

        if self.open_loop_wait_s > 0:
            changed_at = self._command_changed_at[arm]
            return (
                changed_at is not None
                and time.monotonic() - changed_at >= self.open_loop_wait_s
            )
        return False

    def step(self):
        rclpy.spin_once(self.node, timeout_sec=0.0)
        now = time.monotonic()
        if self._next_step_time is None:
            self._next_step_time = now
        self._next_step_time += 1.0 / self.rate_hz
        sleep_s = self._next_step_time - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # Fell behind (e.g. after a debugger pause): resynchronize.
            self._next_step_time = now

    @property
    def step_dt(self):
        return 1.0 / self.rate_hz

    @property
    def is_running(self):
        return rclpy.ok()

    def wait_for_feedback(self, timeout_s=2.0, arms=(LEFT, RIGHT)):
        """Spin until EE feedback arrives on the requested arms."""
        arms = tuple(arms)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if all(self._feedback_is_fresh(arm) for arm in arms):
                return True
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return all(self._feedback_is_fresh(arm) for arm in arms)

    def hold_current_pose(self, arms=(LEFT, RIGHT)):
        """Ask the pose controller to hold the latest fresh feedback pose."""
        for arm in arms:
            if not self._feedback_is_fresh(arm):
                continue
            position, orientation = self._feedback[arm]
            message = PoseStamped()
            message.header.stamp = self.node.get_clock().now().to_msg()
            message.header.frame_id = self.frame_id
            message.pose.position.x = float(position[0])
            message.pose.position.y = float(position[1])
            message.pose.position.z = float(position[2])
            message.pose.orientation.w = float(orientation[0])
            message.pose.orientation.x = float(orientation[1])
            message.pose.orientation.y = float(orientation[2])
            message.pose.orientation.z = float(orientation[3])
            if self.dry_run:
                self.node.get_logger().info(f"[dry-run] hold current {arm} pose")
            else:
                self._pose_publishers[arm].publish(message)

    def abort(self):
        self.hold_current_pose()

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
