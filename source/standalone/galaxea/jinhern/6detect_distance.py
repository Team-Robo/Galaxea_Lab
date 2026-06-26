import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type = int, default = 1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import time
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab.utils.math import quat_mul
from omni.isaac.lab_tasks.galaxea.manager_based.lift.config.agents.init_pose import (
    RIGHT_EE_POSE, LEFT_EE_POSE)

from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

def main():
    # create environment config
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-R1-IK-Abs-v0",
        num_envs=args_cli.num_envs,
    )
    env_cfg.episode_length_s = 100.0
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
    object_data = env.unwrapped.scene["object"].data
    left_ee_frame = env.unwrapped.scene["left_ee_frame"]
    left_start_orientation = left_ee_frame.data.target_quat_w[..., 0, :].clone()
    rotation_90_z = torch.tensor(
        [0.7071, 0.0, 0.0, 0.7071], device=device
    ).repeat(num_envs, 1)
    left_target_orientation = quat_mul(left_orientation, rotation_90_z)

    count = 0
    go_down = False
    print("left orientation: ", left_orientation)
    while simulation_app.is_running():
        with torch.inference_mode():

            # read object pose
            object_data = env.unwrapped.scene["object"].data

            object_position = (
                object_data.root_pos_w
                - env.unwrapped.scene.env_origins
            )
            target_position = object_position.clone()
            target_position[:, 2] += 0.15
            distance = torch.norm(left_position - target_position, dim = -1)
            distance_ori = torch.norm(left_target_orientation - left_orientation, dim = 1)
            print("Distance and Orientation distance", distance, distance_ori)
            if (distance[0] > 0.2 or distance_ori[0] > 0.1) and not go_down:
                left_position = torch.lerp(left_position, target_position, 0.01)
                alpha = min((count + 1) / 500.0, 1.0)
                left_orientation = torch.nn.functional.normalize(
                    torch.lerp(
                        left_start_orientation, left_target_orientation, alpha
                    ),
                    dim=-1,
                )
            else:
                go_down = True
                down_position = object_position.clone()
                down_position[:, 2] += 0.05
                left_position = torch.lerp(left_position, down_position, 0.01) 

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

            if count % 100 == 0:
                print("object position:", object_position)
                print("left target position:", left_position)

            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()