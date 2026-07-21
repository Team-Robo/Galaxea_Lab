"""Isaac Lab implementation of ManipulatorInterface.

Import this module only after ``AppLauncher`` has started the simulation app
(it pulls in torch and omni modules at import time).

The backend keeps the command-smoothing behavior of ``8_simpleAPI.py``:
``move_to`` lerps a persistent command toward the target every tick and
``step()`` sends the full dual-arm IK-absolute action to the environment.
Commands are broadcast to every environment instance; feedback and reach
checks are designed for ``num_envs=1`` (reach checks require *all* envs to
satisfy the tolerance, ``get_ee_pose`` reports env 0).
"""

import numpy as np
import torch

from omni.isaac.lab_tasks.galaxea.manager_based.lift.config.agents.init_pose import (
    LEFT_EE_POSE,
    RIGHT_EE_POSE,
)

from .interface import ManipulatorInterface, as_position, as_quaternion


class IsaacManipulator(ManipulatorInterface):
    """Drives the R1 in the Isaac lift environment via IK-absolute actions."""

    def __init__(
        self,
        env,
        simulation_app=None,
        move_fraction=0.02,
        command_position_tolerance=0.05,
        command_orientation_tolerance=0.05,
    ):
        self.env = env
        self.simulation_app = simulation_app
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.move_fraction = move_fraction
        self.command_position_tolerance = command_position_tolerance
        self.command_orientation_tolerance = command_orientation_tolerance

        self._ee_frames = {
            "left": env.unwrapped.scene["left_ee_frame"],
            "right": env.unwrapped.scene["right_ee_frame"],
        }
        self._positions = {
            "left": self._batch(LEFT_EE_POSE[0:3]),
            "right": self._batch(RIGHT_EE_POSE[0:3]),
        }
        self._orientations = {
            "left": self._batch(LEFT_EE_POSE[3:]),
            "right": self._batch(RIGHT_EE_POSE[3:]),
        }
        self._grippers = {
            "left": torch.ones((self.num_envs, 1), device=self.device),
            "right": torch.ones((self.num_envs, 1), device=self.device),
        }
        self._closed = False

    # -- helpers -----------------------------------------------------------

    def _batch(self, value):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return value.unsqueeze(0).repeat(self.num_envs, 1).clone()

    def _batch_position(self, position):
        return self._batch(torch.from_numpy(as_position(position).astype(np.float32)))

    def _batch_quaternion(self, orientation):
        return self._batch(torch.from_numpy(as_quaternion(orientation).astype(np.float32)))

    def _measured_pose(self, arm):
        frame = self._ee_frames[arm]
        position = (
            frame.data.target_pos_w[..., 0, :]
            - self.env.unwrapped.scene.env_origins
        )
        orientation = torch.nn.functional.normalize(
            frame.data.target_quat_w[..., 0, :], dim=-1
        )
        return position, orientation

    # -- ManipulatorInterface ----------------------------------------------

    def move_to(self, arm, position, orientation=None):
        target_position = self._batch_position(position)
        self._positions[arm] = torch.lerp(
            self._positions[arm], target_position, self.move_fraction
        )

        if orientation is not None:
            target_orientation = self._batch_quaternion(orientation)
            # q and -q are the same rotation: pick the sign nearest the
            # current command before interpolating.
            dot = torch.sum(
                self._orientations[arm] * target_orientation, dim=-1, keepdim=True
            )
            target_orientation = torch.where(
                dot < 0.0, -target_orientation, target_orientation
            )
            self._orientations[arm] = torch.nn.functional.normalize(
                torch.lerp(
                    self._orientations[arm], target_orientation, self.move_fraction
                ),
                dim=-1,
            )

    def set_gripper(self, arm, open_gripper):
        self._grippers[arm].fill_(1.0 if open_gripper else -1.0)

    def get_ee_pose(self, arm):
        position, orientation = self._measured_pose(arm)
        return (
            position[0].cpu().numpy().astype(np.float64),
            orientation[0].cpu().numpy().astype(np.float64),
        )

    def reached(
        self,
        arm,
        position,
        orientation=None,
        position_tolerance=0.01,
        orientation_tolerance=0.03,
    ):
        # Batched override: every env must be within tolerance.
        actual_position, actual_orientation = self._measured_pose(arm)
        position_error = torch.norm(
            self._batch_position(position) - actual_position, dim=-1
        )
        position_ok = torch.all(position_error < position_tolerance)
        if orientation is None:
            return bool(position_ok.item())

        similarity = torch.abs(
            torch.sum(self._batch_quaternion(orientation) * actual_orientation, dim=-1)
        )
        orientation_ok = torch.all(1.0 - similarity < orientation_tolerance)
        return bool((position_ok & orientation_ok).item())

    def command_reached(
        self,
        arm,
        position,
        orientation=None,
        position_tolerance=None,
        orientation_tolerance=None,
    ):
        """Check the smoothed *command*, not the physical pose.

        Lets an approach waypoint hand over even if IK retains a small error;
        the next state still moves gradually and checks the measured pose.
        """
        if position_tolerance is None:
            position_tolerance = self.command_position_tolerance
        if orientation_tolerance is None:
            orientation_tolerance = self.command_orientation_tolerance

        position_error = torch.norm(
            self._batch_position(position) - self._positions[arm], dim=-1
        )
        position_ok = torch.all(position_error < position_tolerance)
        if orientation is None:
            return bool(position_ok.item())

        similarity = torch.abs(
            torch.sum(
                self._batch_quaternion(orientation)
                * torch.nn.functional.normalize(self._orientations[arm], dim=-1),
                dim=-1,
            )
        )
        orientation_ok = torch.all(1.0 - similarity < orientation_tolerance)
        return bool((position_ok & orientation_ok).item())

    def step(self):
        with torch.inference_mode():
            self.env.step(self._build_action())

    def _build_action(self):
        return torch.cat(
            [
                self._positions["left"],
                self._orientations["left"],
                self._grippers["left"],
                self._positions["right"],
                self._orientations["right"],
                self._grippers["right"],
            ],
            dim=-1,
        )

    @property
    def step_dt(self):
        return float(self.env.unwrapped.step_dt)

    @property
    def is_running(self):
        if self.simulation_app is None:
            return True
        return self.simulation_app.is_running()

    def shutdown(self):
        if not self._closed:
            self._closed = True
            self.env.close()
