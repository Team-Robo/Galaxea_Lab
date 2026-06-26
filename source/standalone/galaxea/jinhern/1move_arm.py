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
    # LEFT ARM TARGET: move to fixed pose
    # ----------------------------
    left_target_position = torch.tensor(
    [[0.68, 0.080, 1.1475]],
    device=device,
    ).repeat(num_envs, 1)

    left_target_orientation = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        device=device,
    ).repeat(num_envs, 1)

    # ----------------------------
    # LEFT ARM INITIAL:
    # ----------------------------
    left_position = LEFT_EE_POSE[0:3].to(device).unsqueeze(0)

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
    print("Information: ", left_position,
            left_orientation,
            left_gripper,
            right_position,
            right_orientation,
            right_gripper)
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
    print("left position", left_position)
    print("left orientation", left_orientation)
    num_steps = 500
    with torch.inference_mode():
        while simulation_app.is_running():
            for count in range(num_steps):
                if not simulation_app.is_running():
                    break
                t = (count + 1) / num_steps
                actions = torch.cat(
                    [
                        left_position + t * (left_target_position - left_position),
                        left_orientation + t * (left_target_orientation - left_orientation),
                        left_gripper,
                        right_position,
                        right_orientation,
                        right_gripper,
                    ],
                    dim=-1,
                )
                env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()