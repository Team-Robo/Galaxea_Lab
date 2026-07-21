"""Combine the R1 simple manipulation API with front-camera diagnostics.

The pick position is read from the simulator's ``scene["object"]`` state.  It is
therefore not hardcoded, but it is still simulation ground truth--the RGB/depth
images are inspected and saved, not used to estimate the object's pose yet.

This file reuses ``R1Manipulator`` from ``8_simpleAPI.py``.  Loading that module
also performs the repository's normal argparse/AppLauncher startup, before the
Isaac Lab imports below are evaluated.
"""

import importlib.util
from pathlib import Path


# Python module names cannot normally start with a digit, so load the existing
# beginner API by its file path. Its top-level code parses AppLauncher arguments
# and launches Isaac Sim, but its main() is not called during this import.
_simple_api_path = Path(__file__).with_name("8_simpleAPI.py")
_simple_api_spec = importlib.util.spec_from_file_location("galaxea_simple_api", _simple_api_path)
assert _simple_api_spec is not None and _simple_api_spec.loader is not None
_simple_api = importlib.util.module_from_spec(_simple_api_spec)
_simple_api_spec.loader.exec_module(_simple_api)

args_cli = _simple_api.args_cli
simulation_app = _simple_api.simulation_app
R1Manipulator = _simple_api.R1Manipulator


# Isaac Sim is running now, so Isaac Lab and torch imports are safe.
import os

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab.sensors import save_images_to_file
from omni.isaac.lab.utils.math import quat_mul
from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


ENV_ID = "Isaac-Lift-Cube-R1-IK-Abs-v0"
OUTPUT_DIR = "./data/perception_simple_api"


def prepare_rgb_preview(rgb: torch.Tensor) -> torch.Tensor:
    """Select environment 0 and return RGB float pixels in [0, 1]."""
    assert rgb.ndim == 4, f"Expected RGB [N, H, W, C], got {tuple(rgb.shape)}"
    assert rgb.shape[-1] in (3, 4), f"Expected RGB/RGBA channels, got {rgb.shape[-1]}"

    # Keep [N, H, W, C], select env 0, and remove alpha when it is present.
    image = rgb[0:1, ..., :3].float()
    if rgb.dtype == torch.uint8:
        image = image / 255.0
    else:
        finite = image[torch.isfinite(image)]
        if finite.numel() > 0 and finite.max() > 1.0:
            image = image / 255.0
    return torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def prepare_depth_preview(depth: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize a copy of env 0 depth for display; preserve metric ``depth``."""
    assert depth.ndim in (3, 4), f"Expected depth [N,H,W] or [N,H,W,1], got {tuple(depth.shape)}"
    metric = depth[0]
    if metric.ndim == 3:
        assert metric.shape[-1] == 1
        metric = metric[..., 0]

    valid = torch.isfinite(metric)
    preview = torch.zeros_like(metric, dtype=torch.float32)
    has_valid_depth = bool(valid.any().item())
    if has_valid_depth:
        values = metric[valid].float()
        minimum = values.min()
        maximum = values.max()
        if maximum > minimum:
            preview[valid] = (values - minimum) / (maximum - minimum)

    # The repository helper expects [N, H, W, C].
    return preview.unsqueeze(0).unsqueeze(-1), has_valid_depth


def print_perception(scene, camera, rgb: torch.Tensor, depth: torch.Tensor, step: int):
    """Print camera tensors and the live object pose every 100 steps."""
    print(f"\n--- Step {step}: perception and manipulation ---")
    print("RGB shape/dtype:", rgb.shape, rgb.dtype)
    rgb_values = rgb[torch.isfinite(rgb)] if rgb.is_floating_point() else rgb.reshape(-1)
    if rgb_values.numel() > 0:
        print("RGB min/max:", rgb_values.min().item(), rgb_values.max().item())
    else:
        print("RGB min/max: no finite pixels")

    print("Depth shape/dtype:", depth.shape, depth.dtype)
    valid_depth = torch.isfinite(depth)
    if valid_depth.any():
        values = depth[valid_depth]
        print("Valid metric depth min/max [m]:", values.min().item(), values.max().item())
    else:
        print("Valid metric depth min/max: no finite pixels")

    if hasattr(camera.data, "pos_w"):
        print("Camera position world [m]:", camera.data.pos_w)
    if hasattr(camera.data, "quat_w_world"):
        print("Camera orientation world [wxyz]:", camera.data.quat_w_world)
    object_data = scene["object"].data
    object_position_world = object_data.root_pos_w
    object_orientation_world = object_data.root_quat_w
    object_position_env = object_position_world - scene.env_origins
    print("Object position world [m]:", object_position_world)
    print("Object orientation world [wxyz]:", object_orientation_world)
    print("Object position environment-local [m]:", object_position_env)


def main():
    env = None
    try:
        env_cfg: LiftEnvCfg = parse_env_cfg(ENV_ID, num_envs=args_cli.num_envs)
        # Give the simple state machine enough time to finish before truncation.
        env_cfg.episode_length_s = 100.0
        env = gym.make(ENV_ID, cfg=env_cfg)
        env.reset()

        scene = env.unwrapped.scene
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        print("Scene keys:", scene.keys())
        assert "front_camera" in scene.keys(), "Use --enable_cameras so front_camera is available."
        assert "object" in scene.keys()

        camera = scene["front_camera"]
        print("Camera output keys:", camera.data.output.keys())
        robot = R1Manipulator(env, device, num_envs)
        initial_actions = robot.build_action()
        print("Full action tensor shape:", initial_actions.shape)
        print("Environment action-space shape:", env.unwrapped.action_space.shape)
        assert initial_actions.shape == env.unwrapped.action_space.shape

        # Read the object pose from simulation instead of typing xyz coordinates.
        # clone() snapshots the pose: after grasping, the object moves with the arm,
        # but the approach/grasp targets must remain fixed.
        object_data = scene["object"].data
        object_position_env = (object_data.root_pos_w - scene.env_origins).clone()

        # Match the grasp orientation convention already used by 8_simpleAPI.py.
        rotation_90_z = torch.tensor([0.7071, 0.0, 0.0, -0.7071], device=device).repeat(num_envs, 1)
        rotation_180_y = torch.tensor([0.0, 0.0, -1.0, 0.0], device=device).repeat(num_envs, 1)
        pick_orientation = quat_mul(
            quat_mul(object_data.root_quat_w.clone(), rotation_180_y), rotation_90_z
        )

        # Placement remains relative to the simulator-provided pose, rather than
        # an absolute xyz location: move the object 20 cm in local +Y.
        place_position_env = object_position_env.clone()
        place_position_env[:, 1] += 0.20

        print("Ground-truth pick position (environment frame):", object_position_env)
        print("Relative place position (environment frame):", place_position_env)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        task = "PICK"
        step_count = 0
        with torch.inference_mode():
            while simulation_app.is_running():
                if task == "PICK":
                    if robot.pick(object_position_env, pick_orientation):
                        task = "PLACE"
                        print("Pick finished; starting place.")
                elif task == "PLACE":
                    if robot.place(place_position_env, pick_orientation):
                        task = "DONE"
                        print("Place finished; continuing camera inspection.")

                actions = robot.build_action()
                env.step(actions)  # exactly one simulation step per loop
                step_count += 1

                # Read camera buffers after stepping so the renderer has updated them.
                rgb = camera.data.output["rgb"]
                depth = camera.data.output["distance_to_image_plane"]

                if step_count % 100 == 0:
                    print("Task/state:", task, robot.state)
                    print_perception(scene, camera, rgb, depth, step_count)

                if step_count % 200 == 0:
                    rgb_path = os.path.join(OUTPUT_DIR, f"rgb_{step_count:06d}.png")
                    depth_path = os.path.join(OUTPUT_DIR, f"depth_{step_count:06d}.png")
                    depth_preview, has_valid_depth = prepare_depth_preview(depth)
                    save_images_to_file(prepare_rgb_preview(rgb), rgb_path)
                    save_images_to_file(depth_preview, depth_path)
                    print("Saved:", rgb_path)
                    print("Saved:", depth_path, "(normalized preview; valid depth:", has_valid_depth, ")")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
