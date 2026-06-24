import argparse

from omni.isaac.lab.app import AppLauncher

# launch Isaac Sim first
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# import after Isaac app launch
import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.galaxea.manager_based.lift.config.agents.init_pose import (
    LEFT_EE_POSE,
    RIGHT_EE_POSE,
)

from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    # create environment config
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-R1-IK-Abs-v0",
        num_envs=args_cli.num_envs,
    )
    env_cfg.episode_length_s = 5.0
    # create environment
    env = gym.make("Isaac-Lift-Cube-R1-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # ----------------------------
    # LEFT ARM: move to fixed pose
    # ----------------------------
    left_position = LEFT_EE_POSE[0:3].to(device).unsqueeze(0).repeat(num_envs, 1)

    left_orientation = LEFT_EE_POSE[3:].to(device).unsqueeze(0).repeat(num_envs, 1)

    left_gripper = torch.ones((num_envs, 1), device=device)

    # ----------------------------
    # RIGHT ARM: keep at rest pose
    # ----------------------------
    right_position = RIGHT_EE_POSE[0:3].to(device).unsqueeze(0).repeat(num_envs, 1)

    right_orientation = RIGHT_EE_POSE[3:].to(device).unsqueeze(0).repeat(num_envs, 1)

    right_gripper = torch.ones((num_envs, 1), device=device)

    # Full R1 action:
    # left_xyz + left_quat + left_gripper
    # right_xyz + right_quat + right_gripper
    actions = torch.cat(
        [
            left_position,
            left_orientation,
            left_gripper,
            right_position,
            right_orientation,
            right_gripper,
        ],
        dim=-1,
    )

    print("actions shape:", actions.shape)
    print("action space:", env.unwrapped.action_space.shape)

    # Read the actual starting TCP position so interpolation begins at the
    # current end-effector pose instead of at the world origin.
    left_ee_frame = env.unwrapped.scene["left_ee_frame"]
    left_start_position = (
        left_ee_frame.data.target_pos_w[..., 0, :].clone()
        - env.unwrapped.scene.env_origins
    )
    left_start_orientation = left_ee_frame.data.target_quat_w[..., 0, :].clone()

    # Quaternions q and -q represent the same orientation. Choose the target
    # sign that gives the shorter interpolation path from the starting pose.
    orientation_dot = torch.sum(
        left_start_orientation * left_orientation, dim=-1, keepdim=True
    )
    left_target_orientation = torch.where(
        orientation_dot < 0.0, -left_orientation, left_orientation
    )

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # read object pose
            object_data = env.unwrapped.scene["object"].data

            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # Move the left arm 15 cm above the object.
            left_position_target = object_position.clone()
            left_position_target[:, 2] += 0.15

            # keep left gripper open
            left_gripper = torch.ones((num_envs, 1), device=device)

            # Smoothly interpolate for 100 simulation steps, then hold/follow
            # the target position.
            alpha = min((count + 1) / 100.0, 1.0)
            left_command_position = torch.lerp(
                left_start_position, left_position_target, alpha
            )
            left_command_orientation = torch.nn.functional.normalize(
                torch.lerp(
                    left_start_orientation, left_target_orientation, alpha
                ),
                dim=-1,
            )

            actions = torch.cat(
                [
                    left_command_position,
                    left_command_orientation,
                    left_gripper,
                    right_position,
                    right_orientation,
                    right_gripper,
                ],
                dim=-1,
            )

            env.step(actions)

            if count % 50 == 0:
                print("object position:", object_position)
                print("left target position:", left_position_target)
                print("left commanded position:", left_command_position)
                print("left commanded orientation:", left_command_orientation)

            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
