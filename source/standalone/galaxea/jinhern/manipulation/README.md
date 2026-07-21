# manipulation — one pipeline, two robots

The same state machines and skills drive either the Isaac Lab sim or the
real Galaxea R1. Only the backend changes:

```
skills.pick() / place() / handover()          (blocking wrappers)
        │
PickStateMachine / PlaceStateMachine / ...    (tick-based, numpy only)
        │
ManipulatorInterface                          (move_to / set_gripper / reached / step)
   ├── IsaacManipulator     smooths a command, env.step(IK-abs action)
   └── RealRobotManipulator publishes PoseStamped; the R1's on-board MPC
                            (Arm Pose Control node) does IK + joint control
```

## Files

| file | contents |
|---|---|
| `interface.py` | `ManipulatorInterface` + shared reach checks. Positions `[x,y,z]` m, quaternions `[w,x,y,z]`. |
| `isaac_robot.py` | Sim backend (port of `8_simpleAPI.py`'s command smoothing). Import only after `AppLauncher`. |
| `r1_robot.py` | Real backend, rclpy. Every topic name is a constructor arg. |
| `state_machine.py` | `PickStateMachine`, `PlaceStateMachine`, `HandoverStateMachine`. |
| `skills.py` | `pick()`, `place()`, `handover()`, `run_state_machine()`. |
| `main.py` | `--backend isaac\|real`, `--skill pick\|pick_place\|handover`. |

## Run

Sim (inside the Isaac container, py3.10 env):

```bash
./isaaclab.sh -p source/standalone/galaxea/jinhern/manipulation/main.py \
    --backend isaac --skill pick_place --enable_cameras --headless
```

The Isaac entry point is intentionally single-env (`--num_envs 1`). It exits
after the skill succeeds or times out; add `--linger` when you want the viewer
to stay open afterward. If a Linux session has no `DISPLAY`/`WAYLAND_DISPLAY`,
`main.py` forces headless startup to avoid Kit's windowed GLFW crash; omit
`--headless` only from a working graphical session or when using livestream.

Real robot (system python3 with ROS 2 sourced; robot's HDAS + Arm Pose
Control + gripper nodes running):

```bash
python3 source/standalone/galaxea/jinhern/manipulation/main.py \
    --backend real --skill pick \
    --object-pos 0.45 0.25 0.10 --orientation 0.5 0.5 -0.5 -0.5 \
    --approach-offset 0.10 --grasp-offset 0.0 --lift-height 0.10 \
    --min-command-z 0.0
```

No-publish dry run (lets the state machine tick without sending hardware
commands):

```bash
python3 source/standalone/galaxea/jinhern/manipulation/main.py \
    --backend real --skill pick --dry-run --open-loop-wait 4.0 \
    --object-pos 0.45 0.25 0.10 --orientation 0.5 0.5 -0.5 -0.5
```

## Real-robot topics (defaults — verify with `ros2 topic list`)

| topic | type | direction |
|---|---|---|
| `/motion_target/target_pose_arm_left` / `_right` | `geometry_msgs/PoseStamped` | command |
| `/motion_target/target_position_gripper_left` / `_right` | `std_msgs/Float32` (0–100, 100 = open) | command |
| `/motion_control/pose_ee_arm_left` / `_right` | `geometry_msgs/PoseStamped` | feedback |

Topic names have shifted between Galaxea SDK releases (older docs:
`/motion_target/pose_ee_arm_*`, `/motion_control/position_control_gripper_*`)
— all names, the gripper value range, and the frame id are
`RealRobotManipulator` constructor arguments and CLI flags:
`--left-pose-topic`, `--right-pose-topic`, `--left-gripper-topic`,
`--right-gripper-topic`, `--left-feedback-topic`, `--right-feedback-topic`,
`--frame-id`, `--gripper-open-value`, and `--gripper-close-value`.

For `pick` and `pick_place`, the CLI waits only for left-arm feedback. Real
`handover` is disabled in `main.py` until separate calibrated giver/receiver
poses and collision spacing are added.

## Before touching hardware

- **Frames differ.** The pose controller expects `gripper_link` relative to
  `torso_link4`; Isaac waypoints are env-local. Do not replay sim coordinates.
- **Offsets differ by backend.** Isaac uses the observed working
  `grasp_offset=0.07` for this scene and `release_offset=-0.07`; the real CLI
  defaults both to `0.0`. Keep offsets explicit while commissioning hardware.
- **No fresh feedback → no hardware progress.** Hardware runs require fresh EE
  feedback (`--feedback-timeout`, default 0.5 s). `--open-loop-wait` is
  accepted only with `--dry-run`, which does not publish pose or gripper
  targets.
- **Guard the z floor.** Real runs reject derived waypoints below
  `--min-command-z` (default `0.0`). Adjust this only after confirming the
  robot's pose-control frame and table geometry.
- **No collision planner here.** The real backend publishes target poses to
  the on-board pose controller; it does not check self-collision, table
  collision, workspace reachability, gripper force, or object-held state.
- Echo `/motion_control/pose_ee_arm_left` first and confirm the reported pose
  matches where the arm actually is before sending any target.
