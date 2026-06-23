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
    RIGHT_EE_POSE, LEFT_EE_POSE)

from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    # create environment config
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Bin-R1-IK-Abs-v0",
        num_envs=args_cli.num_envs,
    )

    # create environment
    env = gym.make("Isaac-Lift-Bin-R1-IK-Abs-v0", cfg=env_cfg)
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
    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():

            # read left TCP pose
            left_ee_frame = env.unwrapped.scene["left_ee_frame"]

            left_tcp_position = (
                left_ee_frame.data.target_pos_w[..., 0, :].clone()
                - env.unwrapped.scene.env_origins
            )

            left_tcp_orientation = left_ee_frame.data.target_quat_w[..., 0, :].clone()
            if count % 50 == 0:
                print("left TCP position:", left_tcp_position)
                print("left TCP orientation:", left_tcp_orientation)

            # keep robot at rest
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

            env.step(actions)
            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()