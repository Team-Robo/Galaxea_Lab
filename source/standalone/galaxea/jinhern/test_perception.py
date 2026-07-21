"""Beginner-friendly test of the Galaxea R1 front-camera inputs.

Run from the Isaac Lab repository root with cameras enabled.  This script only
holds the robot at its initial end-effector poses, reads sensor tensors, prints
diagnostics, and saves image previews.  It does not perform object detection or
manipulation.
"""

import argparse

from omni.isaac.lab.app import AppLauncher


# Isaac Sim must be launched before importing gym, torch, or Isaac Lab modules.
parser = argparse.ArgumentParser(description="Inspect Galaxea R1 perception inputs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulated environments (default: 1).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# These imports intentionally come after AppLauncher creates the simulation app.
import os

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401  (registers the Gym environments)
from omni.isaac.lab.sensors import save_images_to_file
from omni.isaac.lab_tasks.galaxea.manager_based.lift.config.agents.init_pose import (
    LEFT_EE_POSE,
    RIGHT_EE_POSE,
)
from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


ENV_ID = "Isaac-Lift-Bin-R1-IK-Abs-v0"
OUTPUT_DIR = "./data/perception_test"


def rgb_for_saving(rgb: torch.Tensor) -> torch.Tensor:
    """Return environment 0 as float RGB in the save helper's [N, H, W, C] layout."""
    assert rgb.ndim == 4, f"Expected RGB [num_envs, H, W, C], received {tuple(rgb.shape)}"
    assert rgb.shape[0] >= 1, "The RGB tensor contains no environments."
    assert rgb.shape[-1] in (3, 4), f"Expected 3 (RGB) or 4 (RGBA) channels, received {rgb.shape[-1]}"

    # Select environment 0, discard alpha if RGBA, and retain a batch dimension.
    image = rgb[0:1, ..., :3]
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    elif image.is_floating_point():
        # Some camera backends return float pixels in [0, 1], others in [0, 255].
        image = image.float()
        finite = image[torch.isfinite(image)]
        if finite.numel() > 0 and finite.max() > 1.0:
            image = image / 255.0
    else:
        image = image.float() / 255.0

    # Saving requires finite values in [0, 1].  This copy does not change rgb.
    return torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def depth_for_saving(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a normalized preview and return the finite mask for metric depth."""
    assert depth.ndim in (3, 4), (
        f"Expected depth [num_envs, H, W] or [num_envs, H, W, 1], received {tuple(depth.shape)}"
    )
    assert depth.shape[0] >= 1, "The depth tensor contains no environments."

    metric_depth = depth[0]  # a view of environment 0; values are still metric
    if metric_depth.ndim == 3:
        assert metric_depth.shape[-1] == 1, "A 4-D depth tensor must have one channel."
        metric_depth = metric_depth[..., 0]

    valid = torch.isfinite(metric_depth)
    preview = torch.zeros_like(metric_depth, dtype=torch.float32)
    if valid.any():
        valid_values = metric_depth[valid].float()
        depth_min = valid_values.min()
        depth_max = valid_values.max()
        span = depth_max - depth_min
        if span > 0:
            preview[valid] = (valid_values - depth_min) / span
        # If all valid pixels have one depth, leave the preview black.

    # save_images_to_file expects [N, H, W, C].
    return preview.unsqueeze(0).unsqueeze(-1), valid


def main():
    """Create the environment and continuously inspect its camera outputs."""
    env = None
    try:
        env_cfg: LiftEnvCfg = parse_env_cfg(ENV_ID, num_envs=args_cli.num_envs)
        env = gym.make(ENV_ID, cfg=env_cfg)
        env.reset()

        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        scene = env.unwrapped.scene

        print("Scene keys:", scene.keys())
        assert "front_camera" in scene.keys(), "front_camera was not found in the scene. Did you pass --enable_cameras?"
        assert "object" in scene.keys(), "object was not found in the scene."

        camera = scene["front_camera"]
        print("Camera output keys:", camera.data.output.keys())
        assert "rgb" in camera.data.output.keys(), "The front camera has no RGB output."
        assert "distance_to_image_plane" in camera.data.output.keys(), "The front camera has no depth output."

        # LEFT_EE_POSE and RIGHT_EE_POSE each contain xyz (3) followed by a
        # wxyz quaternion (4).  repeat() creates one target per environment.
        left_position = LEFT_EE_POSE[:3].to(device).unsqueeze(0).repeat(num_envs, 1)
        left_orientation = LEFT_EE_POSE[3:].to(device).unsqueeze(0).repeat(num_envs, 1)
        right_position = RIGHT_EE_POSE[:3].to(device).unsqueeze(0).repeat(num_envs, 1)
        right_orientation = RIGHT_EE_POSE[3:].to(device).unsqueeze(0).repeat(num_envs, 1)
        left_gripper = torch.ones((num_envs, 1), device=device)
        right_gripper = torch.ones((num_envs, 1), device=device)

        # 3 + 4 + 1 values for the left arm, then 3 + 4 + 1 for the right arm.
        actions = torch.cat(
            (
                left_position,
                left_orientation,
                left_gripper,
                right_position,
                right_orientation,
                right_gripper,
            ),
            dim=-1,
        )
        print("Full action tensor shape:", actions.shape)
        print("Environment action-space shape:", env.unwrapped.action_space.shape)
        assert actions.shape == (num_envs, 16), f"R1 action must be [num_envs, 16], got {tuple(actions.shape)}"
        assert actions.shape == env.unwrapped.action_space.shape, "Action tensor does not match the batched action space."
        print("Tensor device:", device, "(CUDA means GPU; cpu means system memory)")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        step_count = 0

        with torch.inference_mode():
            while simulation_app.is_running():
                # Exactly one environment step per loop.  Stepping updates the
                # renderer/sensors, so camera tensors read below are current.
                env.step(actions)
                step_count += 1

                # Camera tensors normally live on the GPU.  Reductions can stay
                # there; the save helper handles moving image data as needed.
                rgb = camera.data.output["rgb"]
                depth = camera.data.output["distance_to_image_plane"]

                assert rgb.ndim == 4, f"RGB should be [N, H, W, C], got {tuple(rgb.shape)}"
                assert rgb.shape[0] == num_envs, "RGB batch size does not match num_envs."
                assert depth.ndim in (3, 4), f"Unexpected depth layout: {tuple(depth.shape)}"
                assert depth.shape[0] == num_envs, "Depth batch size does not match num_envs."

                if step_count % 100 == 0:
                    finite_rgb = rgb[torch.isfinite(rgb)] if rgb.is_floating_point() else rgb.reshape(-1)
                    print(f"\n--- Perception diagnostics at step {step_count} ---")
                    print("RGB shape:", rgb.shape)
                    print("RGB dtype:", rgb.dtype)
                    if finite_rgb.numel() > 0:
                        print("RGB min/max:", finite_rgb.min().item(), finite_rgb.max().item())
                    else:
                        print("RGB min/max: no finite pixels")

                    valid_depth = torch.isfinite(depth)
                    print("Depth shape:", depth.shape)
                    print("Depth dtype:", depth.dtype)
                    if valid_depth.any():
                        values = depth[valid_depth]
                        print("Valid depth min/max:", values.min().item(), values.max().item())
                    else:
                        print("Valid depth min/max: no finite depth pixels")

                    # Isaac Lab 4.0 CameraData exposes these world-frame fields.
                    if hasattr(camera.data, "pos_w"):
                        print("Camera position world [m]:", camera.data.pos_w)
                    if hasattr(camera.data, "quat_w_world"):
                        print("Camera orientation world [w, x, y, z]:", camera.data.quat_w_world)

                    # Ground truth comes directly from simulator state, not RGB/depth.
                    object_data = scene["object"].data
                    object_position_world = object_data.root_pos_w
                    object_orientation_world = object_data.root_quat_w
                    object_position_env = object_position_world - scene.env_origins
                    print("Object position world [m]:", object_position_world)
                    print("Object orientation world [w, x, y, z]:", object_orientation_world)
                    print("Object position environment-local [m]:", object_position_env)

                if step_count % 200 == 0:
                    rgb_preview = rgb_for_saving(rgb)
                    depth_preview, valid_depth_env0 = depth_for_saving(depth)
                    rgb_path = os.path.join(OUTPUT_DIR, f"rgb_{step_count:06d}.png")
                    depth_path = os.path.join(OUTPUT_DIR, f"depth_{step_count:06d}.png")
                    save_images_to_file(rgb_preview, rgb_path)
                    save_images_to_file(depth_preview, depth_path)
                    print(f"Saved {rgb_path}")
                    if valid_depth_env0.any():
                        print(f"Saved {depth_path} (normalized visualization only)")
                    else:
                        print(f"Saved {depth_path} (black: no finite depth pixels)")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
