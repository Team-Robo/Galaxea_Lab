import argparse

from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab.utils.math import quat_mul
from omni.isaac.lab_tasks.galaxea.manager_based.lift.config.agents.init_pose import (
    LEFT_EE_POSE,
    RIGHT_EE_POSE,
)
from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


class R1Manipulator:
    """Small, non-blocking API for the R1 left arm.

    Call ``pick`` or ``place`` once per simulation loop, then send the result of
    ``build_action`` to the environment. Each method returns True when its whole
    operation has finished.
    """

    def __init__(self, env, device, num_envs):
        self.env = env
        self.device = device
        self.num_envs = num_envs

        self.left_position = self._batch_pose(LEFT_EE_POSE[0:3])
        self.left_orientation = self._batch_pose(LEFT_EE_POSE[3:])
        self.right_position = self._batch_pose(RIGHT_EE_POSE[0:3])
        self.right_orientation = self._batch_pose(RIGHT_EE_POSE[3:])

        self.left_gripper = torch.ones((num_envs, 1), device=device)
        self.right_gripper = torch.ones((num_envs, 1), device=device)

        self.left_ee_frame = env.unwrapped.scene["left_ee_frame"]
        self.position_tolerance = 0.01
        self.orientation_tolerance = 0.03
        # The approach only needs to be close enough to begin the slower descent.
        # This matches the 0.05 command tolerance used by 7_sm.py.
        self.approach_position_tolerance = 0.05
        self.approach_orientation_tolerance = 0.05
        self.move_fraction = 0.02
        self.gripper_wait_steps = 50

        self.operation = None
        self.state = "IDLE"
        self.state_count = 0
        self.targets = {}

    def _batch_pose(self, value):
        return value.to(self.device).unsqueeze(0).repeat(self.num_envs, 1).clone()

    def _as_batch(self, value, size):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0).repeat(self.num_envs, 1)
        if value.shape != (self.num_envs, size):
            raise ValueError(
                f"Expected shape ({self.num_envs}, {size}), got {tuple(value.shape)}"
            )
        return value.clone()

    def open_left_gripper(self):
        self.left_gripper.fill_(1.0)

    def close_left_gripper(self):
        self.left_gripper.fill_(-1.0)

    def move_left_to(self, position, orientation=None, smooth=True):
        position = self._as_batch(position, 3)
        if smooth:
            self.left_position = torch.lerp(
                self.left_position, position, self.move_fraction
            )
        else:
            self.left_position = position

        if orientation is not None:
            orientation = self._as_batch(orientation, 4)
            if smooth:
                # q and -q are the same rotation. Select the target sign that is
                # closest to the current quaternion before interpolating.
                dot = torch.sum(self.left_orientation * orientation, dim=-1, keepdim=True)
                orientation = torch.where(dot < 0.0, -orientation, orientation)
                orientation = torch.lerp(
                    self.left_orientation, orientation, self.move_fraction
                )
            # IK expects unit quaternions.
            self.left_orientation = torch.nn.functional.normalize(
                orientation, dim=-1
            )

    def left_command_reached(self, position, orientation=None):
        """Check the gradually generated command, not the physical EE pose.

        This is useful for an approach waypoint: the physical IK solution may
        retain a small error, but the next state is still safe because DOWN also
        moves gradually and checks the measured end-effector position.
        """
        position = self._as_batch(position, 3)
        position_error = torch.norm(position - self.left_position, dim=-1)
        position_ok = torch.all(
            position_error < self.approach_position_tolerance
        )

        if orientation is None:
            return bool(position_ok.item())

        orientation = torch.nn.functional.normalize(
            self._as_batch(orientation, 4), dim=-1
        )
        current_orientation = torch.nn.functional.normalize(
            self.left_orientation, dim=-1
        )
        similarity = torch.abs(
            torch.sum(orientation * current_orientation, dim=-1)
        )
        orientation_ok = torch.all(
            1.0 - similarity < self.approach_orientation_tolerance
        )
        return bool((position_ok & orientation_ok).item())

    def left_pose_errors(self, position, orientation=None):
        """Return maximum measured position and orientation errors for logging."""
        position = self._as_batch(position, 3)
        actual_position = self.left_ee_frame.data.target_pos_w[..., 0, :]
        actual_position = actual_position - self.env.unwrapped.scene.env_origins
        position_error = torch.norm(position - actual_position, dim=-1).max()

        if orientation is None:
            return position_error.item(), 0.0

        orientation = torch.nn.functional.normalize(
            self._as_batch(orientation, 4), dim=-1
        )
        actual_orientation = torch.nn.functional.normalize(
            self.left_ee_frame.data.target_quat_w[..., 0, :], dim=-1
        )
        similarity = torch.abs(
            torch.sum(orientation * actual_orientation, dim=-1)
        )
        orientation_error = (1.0 - similarity).max()
        return position_error.item(), orientation_error.item()

    def left_pose_reached(self, position, orientation=None):
        position = self._as_batch(position, 3)
        actual_position = self.left_ee_frame.data.target_pos_w[..., 0, :]
        actual_position = actual_position - self.env.unwrapped.scene.env_origins
        position_ok = torch.norm(position - actual_position, dim=-1)
        position_ok = torch.all(position_ok < self.position_tolerance)

        if orientation is None:
            return bool(position_ok.item())

        orientation = torch.nn.functional.normalize(
            self._as_batch(orientation, 4), dim=-1
        )
        actual_orientation = torch.nn.functional.normalize(
            self.left_ee_frame.data.target_quat_w[..., 0, :], dim=-1
        )
        # q and -q describe the same rotation, hence abs(dot).
        similarity = torch.abs(torch.sum(orientation * actual_orientation, dim=-1))
        orientation_ok = torch.all(1.0 - similarity < self.orientation_tolerance)
        return bool((position_ok & orientation_ok).item())

    def _enter_state(self, state):
        self.state = state
        self.state_count = 0

    def _begin_pick(self, object_position, orientation):
        # Snapshot every target once. Do not derive LIFT from the moving object.
        object_position = self._as_batch(object_position, 3)
        orientation = self._as_batch(orientation, 4)

        approach = object_position.clone()
        approach[:, 2] += 0.15

        grasp = object_position.clone()
        grasp[:, 2] -= 0.15

        lift = grasp.clone()
        lift[:, 2] += 0.10

        self.targets = {
            "approach": approach,
            "grasp": grasp,
            "lift": lift,
            "orientation": orientation,
        }
        self.operation = "PICK"
        self._enter_state("APPROACH")

    def pick(self, object_position, orientation=None):
        """Advance a pick operation and return True when the object is lifted."""
        if self.operation is None:
            if orientation is None:
                orientation = self.left_orientation
            self._begin_pick(object_position, orientation)
        elif self.operation != "PICK":
            raise RuntimeError(f"Cannot pick while {self.operation} is active")

        target_orientation = self.targets["orientation"]

        if self.state == "APPROACH":
            self.open_left_gripper()
            self.move_left_to(self.targets["approach"], target_orientation)
            if self.left_command_reached(
                self.targets["approach"], target_orientation
            ):
                self._enter_state("DOWN")

        elif self.state == "DOWN":
            self.move_left_to(self.targets["grasp"], target_orientation)
            if self.left_pose_reached(self.targets["grasp"], target_orientation):
                self._enter_state("CLOSE")

        elif self.state == "CLOSE":
            self.close_left_gripper()
            if self.state_count >= self.gripper_wait_steps:
                self._enter_state("LIFT")

        elif self.state == "LIFT":
            self.close_left_gripper()
            self.move_left_to(self.targets["lift"], target_orientation)
            if self.left_pose_reached(self.targets["lift"], target_orientation):
                self.operation = None
                self._enter_state("IDLE")
                return True

        self.state_count += 1
        return False

    def _begin_place(self, target_position, orientation):
        target_position = self._as_batch(target_position, 3)
        orientation = self._as_batch(orientation, 4)

        above = target_position.clone()
        above[:, 2] += 0.15

        release = target_position.clone()
        release[:, 2] -= 0.07

        self.targets = {
            "above": above,
            "release": release,
            "orientation": orientation,
        }
        self.operation = "PLACE"
        self._enter_state("ABOVE")

    def place(self, target_position, orientation=None):
        """Advance a place operation and return True after releasing and retreating."""
        if self.operation is None:
            if orientation is None:
                orientation = self.left_orientation
            self._begin_place(target_position, orientation)
        elif self.operation != "PLACE":
            raise RuntimeError(f"Cannot place while {self.operation} is active")

        target_orientation = self.targets["orientation"]

        if self.state == "ABOVE":
            self.close_left_gripper()
            self.move_left_to(self.targets["above"], target_orientation)
            if self.left_command_reached(self.targets["above"], target_orientation):
                self._enter_state("DOWN")

        elif self.state == "DOWN":
            self.close_left_gripper()
            self.move_left_to(self.targets["release"], target_orientation)
            if self.left_pose_reached(self.targets["release"], target_orientation):
                self._enter_state("OPEN")

        elif self.state == "OPEN":
            self.open_left_gripper()
            if self.state_count >= self.gripper_wait_steps:
                self._enter_state("RETREAT")

        elif self.state == "RETREAT":
            self.open_left_gripper()
            self.move_left_to(self.targets["above"], target_orientation)
            if self.left_pose_reached(self.targets["above"], target_orientation):
                self.operation = None
                self._enter_state("IDLE")
                return True

        self.state_count += 1
        return False

    def build_action(self):
        return torch.cat(
            [
                self.left_position,
                self.left_orientation,
                self.left_gripper,
                self.right_position,
                self.right_orientation,
                self.right_gripper,
            ],
            dim=-1,
        )


def main():
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-R1-IK-Abs-v0",
        num_envs=args_cli.num_envs,
    )
    env_cfg.episode_length_s = 100.0
    env = gym.make("Isaac-Lift-Cube-R1-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    robot = R1Manipulator(env, device, num_envs)

    object_data = env.unwrapped.scene["object"].data
    object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins

    rotation_90_z = torch.tensor(
        [0.7071, 0.0, 0.0, -0.7071], device=device
    ).repeat(num_envs, 1)
    rotation_180_y = torch.tensor(
        [0.0, 0.0, -1.0, 0.0], device=device
    ).repeat(num_envs, 1)
    pick_orientation = quat_mul(
        quat_mul(object_data.root_quat_w.clone(), rotation_180_y), rotation_90_z
    )

    # Keep the demo target fixed; the API snapshots it again when place starts.
    place_position = object_position.clone()
    place_position[:, 1] += 0.20

    task = "PICK"
    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            if task == "PICK":
                if robot.pick(object_position, pick_orientation):
                    task = "PLACE"
            elif task == "PLACE":
                if robot.place(place_position, pick_orientation):
                    task = "DONE"

            env.step(robot.build_action())

            if count % 100 == 0:
                active_target = robot.targets.get(
                    robot.state.lower(), robot.left_position
                )
                target_orientation = robot.targets.get(
                    "orientation", robot.left_orientation
                )
                position_error, orientation_error = robot.left_pose_errors(
                    active_target, target_orientation
                )
                print(
                    f"task={task}, robot_state={robot.state}, "
                    f"position_error={position_error:.4f}, "
                    f"orientation_error={orientation_error:.4f}"
                )
            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
