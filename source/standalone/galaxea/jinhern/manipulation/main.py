"""Entry point: run pick / place / handover on Isaac sim or the real R1.

Isaac (from the repo root, inside the sim container):
    ./isaaclab.sh -p source/standalone/galaxea/jinhern/manipulation/main.py \
        --backend isaac --skill pick_place --enable_cameras --headless

Real robot (system Python with ROS 2 sourced; Arm Pose Control + gripper
nodes must already be running on the robot):
    python3 source/standalone/galaxea/jinhern/manipulation/main.py \
        --backend real --skill pick_place \
        --object-pos 0.45 0.25 0.10 \
        --orientation 0.5 0.5 -0.5 -0.5 \
        --approach-offset 0.10 --grasp-offset 0.0 --release-offset 0.0 \
        --lift-height 0.10 --min-command-z 0.0

Real-robot positions/orientations are in the frame the Arm Pose Control node
expects (gripper_link relative to torso_link4) — NOT Isaac env coordinates.
Start with fresh feedback, the arm far from the table, and generous offsets.
"""

import argparse
import math
import os
import sys

# Allow running this file directly: put the package's parent on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def finite_float(value):
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError(f"expected a finite float, got {value!r}")
    return result


def finalize_args(parser, args):
    """Resolve backend-specific defaults and reject unsafe combinations."""
    if args.timeout <= 0.0:
        parser.error("--timeout must be > 0")
    if args.gripper_wait < 0.0:
        parser.error("--gripper-wait must be >= 0")
    if args.rate_hz <= 0.0:
        parser.error("--rate-hz must be > 0")
    if args.open_loop_wait < 0.0:
        parser.error("--open-loop-wait must be >= 0")
    if args.feedback_timeout <= 0.0:
        parser.error("--feedback-timeout must be > 0")

    if args.grasp_offset is None:
        args.grasp_offset = 0.07 if args.backend == "isaac" else 0.0
    if args.release_offset is None:
        args.release_offset = -0.07 if args.backend == "isaac" else 0.0

    if args.backend == "real":
        if args.skill == "handover":
            parser.error(
                "--backend real --skill handover is disabled until calibrated "
                "giver/receiver poses and collision spacing are added"
            )
        if args.open_loop_wait > 0.0 and not args.dry_run:
            parser.error("--open-loop-wait is only allowed together with --dry-run")
    return args


def required_feedback_arms(skill):
    if skill == "handover":
        return ("left", "right")
    return ("left",)


def force_headless_without_display(args):
    """Avoid Isaac's windowed kit on Linux sessions with no display server."""
    if (
        sys.platform.startswith("linux")
        and not getattr(args, "headless", False)
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        print(
            "No DISPLAY/WAYLAND_DISPLAY detected; forcing --headless for Isaac "
            "startup. Use a graphical session or livestream if you need a viewer."
        )
        args.headless = True


def build_parser():
    parser = argparse.ArgumentParser(description="R1 manipulation pipeline")
    parser.add_argument("--backend", choices=["isaac", "real"], default="isaac")
    parser.add_argument(
        "--skill", choices=["pick", "pick_place", "handover"], default="pick_place"
    )
    parser.add_argument(
        "--object-pos",
        type=finite_float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Object position. Isaac: defaults to the scene object. "
        "Real: required, in the robot's pose-control frame.",
    )
    parser.add_argument(
        "--orientation",
        type=finite_float,
        nargs=4,
        default=None,
        metavar=("W", "X", "Y", "Z"),
        help="Grasp orientation quaternion [w x y z]. Isaac: computed from "
        "the object if omitted. Real: required.",
    )
    parser.add_argument(
        "--place-offset",
        type=finite_float,
        nargs=3,
        default=[0.0, 0.20, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Place target relative to the object position.",
    )
    parser.add_argument("--approach-offset", type=finite_float, default=0.15)
    parser.add_argument(
        "--grasp-offset",
        type=finite_float,
        default=None,
        help="Pick grasp z offset. Default: Isaac 0.07, real 0.0.",
    )
    parser.add_argument(
        "--release-offset",
        type=finite_float,
        default=None,
        help="Place release z offset. Default: Isaac -0.07, real 0.0.",
    )
    parser.add_argument("--lift-height", type=finite_float, default=0.10)
    parser.add_argument("--gripper-wait", type=finite_float, default=1.0)
    parser.add_argument("--timeout", type=finite_float, default=120.0)

    real = parser.add_argument_group("real robot options")
    real.add_argument("--rate-hz", type=finite_float, default=20.0)
    real.add_argument("--frame-id", default="torso_link4")
    real.add_argument(
        "--left-pose-topic", default="/motion_target/target_pose_arm_left"
    )
    real.add_argument(
        "--right-pose-topic", default="/motion_target/target_pose_arm_right"
    )
    real.add_argument(
        "--left-gripper-topic",
        default="/motion_target/target_position_gripper_left",
    )
    real.add_argument(
        "--right-gripper-topic",
        default="/motion_target/target_position_gripper_right",
    )
    real.add_argument(
        "--left-feedback-topic", default="/motion_control/pose_ee_arm_left"
    )
    real.add_argument(
        "--right-feedback-topic", default="/motion_control/pose_ee_arm_right"
    )
    real.add_argument("--gripper-open-value", type=finite_float, default=100.0)
    real.add_argument("--gripper-close-value", type=finite_float, default=0.0)
    real.add_argument("--feedback-timeout", type=finite_float, default=0.5)
    real.add_argument(
        "--min-command-z",
        type=finite_float,
        default=0.0,
        help="Real backend: reject derived pose targets below this z value.",
    )
    real.add_argument(
        "--max-command-delta",
        type=finite_float,
        default=None,
        help="Real backend: optional max Cartesian jump per pose command.",
    )
    real.add_argument(
        "--dry-run",
        action="store_true",
        help="Real backend: do not publish pose or gripper targets.",
    )
    real.add_argument(
        "--open-loop-wait",
        type=finite_float,
        default=0.0,
        help="Real backend dry-run only: with no EE feedback, treat each waypoint "
        "as reached after this many seconds.",
    )
    isaac = parser.add_argument_group("Isaac options")
    isaac.add_argument("--num_envs", type=int, default=1)
    isaac.add_argument(
        "--linger",
        action="store_true",
        help="Isaac backend: keep rendering after the skill finishes.",
    )
    return parser


def pick_config(args):
    return dict(
        approach_offset=args.approach_offset,
        grasp_offset=args.grasp_offset,
        lift_height=args.lift_height,
        gripper_wait_s=args.gripper_wait,
    )


def run_skills(robot, args, object_position, orientation, receiver_orientation=None):
    """Shared skill sequence for both backends."""
    import numpy as np

    from manipulation import skills

    object_position = np.asarray(object_position, dtype=np.float64)

    ok = skills.pick(
        robot,
        object_position,
        orientation,
        timeout_s=args.timeout,
        **pick_config(args),
    )
    print(f"pick: {'done' if ok else 'FAILED'}")
    if not ok or args.skill == "pick":
        return ok

    if args.skill == "pick_place":
        place_position = object_position + np.asarray(args.place_offset)
        ok = skills.place(
            robot,
            place_position,
            orientation,
            above_offset=args.approach_offset,
            release_offset=args.release_offset,
            gripper_wait_s=args.gripper_wait,
            timeout_s=args.timeout,
        )
        print(f"place: {'done' if ok else 'FAILED'}")
        return ok

    # handover: bring the object up and across, pass left -> right.
    handover_position = object_position + np.asarray([0.0, -0.10, 0.35])
    ok = skills.handover(
        robot,
        handover_position,
        giver_orientation=orientation,
        receiver_orientation=receiver_orientation,
        gripper_wait_s=args.gripper_wait,
        timeout_s=args.timeout,
    )
    print(f"handover: {'done' if ok else 'FAILED'}")
    return ok


def main_isaac(parser):
    from omni.isaac.lab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = finalize_args(parser, parser.parse_args())
    if args.num_envs != 1:
        parser.error("manipulation/main.py currently supports --num_envs 1 only")
    force_headless_without_display(args)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac imports are only valid after the app is up.
    import gymnasium as gym
    import torch
    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab.utils.math import quat_mul
    from omni.isaac.lab_tasks.galaxea.manager_based.lift.config.agents.init_pose import (
        RIGHT_EE_POSE,
    )
    from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg
    from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

    from manipulation.isaac_robot import IsaacManipulator

    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-R1-IK-Abs-v0", num_envs=args.num_envs
    )
    env_cfg.episode_length_s = 100.0
    env = gym.make("Isaac-Lift-Cube-R1-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    robot = IsaacManipulator(env, simulation_app)
    device = env.unwrapped.device

    object_data = env.unwrapped.scene["object"].data
    if args.object_pos is not None:
        object_position = args.object_pos
    else:
        object_position = (
            (object_data.root_pos_w - env.unwrapped.scene.env_origins)[0]
            .cpu()
            .numpy()
        )

    if args.orientation is not None:
        orientation = args.orientation
    else:
        # Same construction as 8_simpleAPI.py: flip the gripper downward and
        # twist 90 degrees around z, relative to the object's yaw.
        rotation_90_z = torch.tensor([0.7071, 0.0, 0.0, -0.7071], device=device)
        rotation_180_y = torch.tensor([0.0, 0.0, -1.0, 0.0], device=device)
        orientation = (
            quat_mul(
                quat_mul(object_data.root_quat_w[0:1].clone(), rotation_180_y.unsqueeze(0)),
                rotation_90_z.unsqueeze(0),
            )[0]
            .cpu()
            .numpy()
        )

    # Receiver default: the right arm's rest orientation (demo placeholder —
    # tune per scene for a stable dual-grasp).
    receiver_orientation = RIGHT_EE_POSE[3:].cpu().numpy()

    ok = False
    try:
        ok = run_skills(robot, args, object_position, orientation, receiver_orientation)
        while args.linger and robot.is_running:
            robot.step()
        return ok
    finally:
        robot.shutdown()
        simulation_app.close()


def main_real(parser):
    args = finalize_args(parser, parser.parse_args())
    if args.object_pos is None or args.orientation is None:
        parser.error("--backend real requires --object-pos and --orientation")

    from manipulation.r1_robot import RealRobotManipulator

    robot = RealRobotManipulator(
        rate_hz=args.rate_hz,
        frame_id=args.frame_id,
        left_pose_topic=args.left_pose_topic,
        right_pose_topic=args.right_pose_topic,
        left_gripper_topic=args.left_gripper_topic,
        right_gripper_topic=args.right_gripper_topic,
        left_ee_feedback_topic=args.left_feedback_topic,
        right_ee_feedback_topic=args.right_feedback_topic,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
        open_loop_wait_s=args.open_loop_wait,
        feedback_timeout_s=args.feedback_timeout,
        dry_run=args.dry_run,
        min_command_z=args.min_command_z,
        max_command_delta=args.max_command_delta,
    )
    feedback_arms = required_feedback_arms(args.skill)
    try:
        if (
            not args.dry_run
            and not robot.wait_for_feedback(timeout_s=3.0, arms=feedback_arms)
        ):
            print(
                "No fresh end-effector feedback received for "
                f"{', '.join(feedback_arms)} arm(s). Check feedback topic names "
                "with `ros2 topic list` and verify frame ids before commanding "
                "hardware."
            )
            return False
        ok = run_skills(
            robot,
            args,
            args.object_pos,
            args.orientation,
            receiver_orientation=args.orientation,
        )
        if not ok:
            robot.abort()
        return ok
    except Exception:
        robot.abort()
        raise
    finally:
        robot.shutdown()


def main():
    parser = build_parser()
    known_args, _ = parser.parse_known_args()
    if known_args.backend == "isaac":
        return main_isaac(parser)
    return main_real(parser)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
