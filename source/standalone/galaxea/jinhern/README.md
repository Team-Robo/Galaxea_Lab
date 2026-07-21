# Galaxea R1 with Isaac Sim 4.0 and Isaac Lab in Docker

This guide shows how to build the repository's Docker image, start and re-enter
the container, verify GPU access, and work through the Galaxea R1 tutorials.
It also covers headless camera rendering and optional WebRTC streaming.

The commands were tested with the `laptop4050-working` branch and an NVIDIA RTX
4050 laptop. Replace values written as `<LIKE_THIS>` for your machine.

> **Important:** Running the Isaac Sim container means accepting NVIDIA's
> applicable license terms. Review those terms before leaving `ACCEPT_EULA=Y`
> in `docker/.env.base`.

## What you will be able to do

After completing this guide, you will be able to:

- run Isaac Sim 4.0 and Isaac Lab with NVIDIA GPU acceleration;
- edit Galaxea source files on the host and run them in Docker;
- launch the Galaxea R1 lift environments;
- inspect RGB, metric depth, and simulator ground-truth object poses;
- run the beginner arm-control and pick-and-place examples;
- view a running simulation through WebRTC when needed.

## 1. Host requirements

Use an Ubuntu host with:

- a supported NVIDIA GPU and current NVIDIA driver;
- Docker Engine and Docker Compose v2;
- NVIDIA Container Toolkit;
- Git;
- at least 16 GB RAM (32 GB is recommended);
- enough SSD space for the Isaac Sim image, build layers, and caches.

Check the host GPU and Docker installation:

```bash
nvidia-smi
docker --version
docker compose version
```

Confirm that Docker can use the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

If this command cannot see the GPU, fix the NVIDIA driver or NVIDIA Container
Toolkit installation before continuing. The Isaac Lab image will not work
correctly without GPU access.

Useful installation references:

- Docker Engine: <https://docs.docker.com/engine/install/ubuntu/>
- Docker post-installation steps: <https://docs.docker.com/engine/install/linux-postinstall/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

## 2. Clone the repository

Keep the repository somewhere under your home directory:

```bash
mkdir -p "$HOME/robotics"
cd "$HOME/robotics"
git clone https://github.com/Team-Robo/Galaxea_Lab.git
cd Galaxea_Lab
git switch laptop4050-working
git status
```

All remaining **host commands** assume this directory is the current directory.

## 3. Understand the Docker layout

This repository already provides:

- `docker/Dockerfile.base` for Isaac Sim 4.0 plus Isaac Lab;
- `docker/docker-compose.yaml` for GPU access, networking, and volumes;
- `docker/container.sh` as the supported build/start/enter/stop wrapper;
- `docker/.env.base` for image and container paths.

The default paths in `docker/.env.base` are:

| Item | Default |
|---|---|
| Isaac Sim version | `4.0.0` |
| Isaac Sim in the container | `/isaac-sim` |
| repository in the container | `/workspace/isaaclab` |
| container home | `/root` |
| base container name | `isaac-lab-base` |

The Compose configuration bind-mounts `source/`, `docs/`, and `tools/`. Changes
to tutorial Python files therefore appear on the host immediately and survive a
container rebuild. It also uses named volumes for shader caches, logs, and
`data_storage`.

> Files written elsewhere in the container may be lost when the container is
> removed. The perception tutorials currently save under `./data/`; copy those
> results to the host before running `container.sh stop`.

## 4. Build and start the container

From the repository root on the host:

```bash
./docker/container.sh start base
```

The first build can take a long time because it downloads the Isaac Sim base
image and installs Isaac Lab dependencies. Later starts reuse Docker layers and
named cache volumes.

The wrapper may ask whether to enable X11 forwarding. Choose `n` if you intend
to use headless mode or WebRTC. X11 is not required for the tutorials in this
guide.

Enter the running container:

```bash
./docker/container.sh enter base
```

You should now be in `/workspace/isaaclab` inside the container. Verify it:

```bash
pwd
git status
nvidia-smi
```

Expected repository path:

```text
/workspace/isaaclab
```

### Daily start and enter commands

After the image has been built once, use the same commands:

```bash
cd "$HOME/robotics/Galaxea_Lab"
./docker/container.sh start base
./docker/container.sh enter base
```

Do not create a new container with a separate `docker run` command unless you
deliberately want to maintain a second Docker configuration.

## 5. Verify Isaac Lab and Galaxea R1

Run all tutorial commands from `/workspace/isaaclab` inside Docker:

```bash
cd /workspace/isaaclab
```

Start with the basic robot example:

```bash
./isaaclab.sh -p source/standalone/galaxea/basic/spawn_robot.py --headless
```

The first Isaac Sim launch may pause while shaders and extensions are cached.
Wait for the environment to finish loading. Stop a running example with
`Ctrl+C` before launching another Isaac Sim process.

## 6. Beginner Galaxea tutorial sequence

The files under `source/standalone/galaxea/jinhern/` are intended to be read in
roughly this order:

| Script | Topic |
|---|---|
| `1move_arm.py` | Build the 16-value dual-arm action and move one arm |
| `2open_close_gripper.py` | Send gripper commands |
| `3read_TCP.py` | Read the tool-center-point/end-effector pose |
| `4move_above_object.py` | Read the simulator object pose and move above it |
| `5simple_state_machine.py` | Organize a task as sequential states |
| `6detect_distance.py` | Work with target distance |
| `7_sm.py` | Expanded state-machine example |
| `8_simpleAPI.py` | Use the reusable non-blocking `R1Manipulator` API |
| `test_perception.py` | Inspect and save RGB/depth without manipulation |
| `test_perception_simple_api.py` | Combine camera diagnostics with the simple API |

Run an arm-control tutorial like this:

```bash
./isaaclab.sh -p source/standalone/galaxea/jinhern/1move_arm.py \
  --num_envs 1 \
  --headless
```

Run the reusable simple API example:

```bash
./isaaclab.sh -p source/standalone/galaxea/jinhern/8_simpleAPI.py \
  --num_envs 1 \
  --headless
```

`8_simpleAPI.py` uses `Isaac-Lift-Cube-R1-IK-Abs-v0`. Its state machine calls
`robot.pick(...)` or `robot.place(...)` once per simulation loop and sends one
complete action with `env.step(robot.build_action())`.

## 7. Perception-only tutorial

Run:

```bash
./isaaclab.sh -p source/standalone/galaxea/jinhern/test_perception.py \
  --num_envs 1 \
  --headless \
  --enable_cameras
```

This script uses `Isaac-Lift-Bin-R1-IK-Abs-v0` and:

- holds both R1 arms at their initial end-effector poses;
- reads `front_camera` RGB and `distance_to_image_plane` depth tensors;
- prints tensor shapes, types, ranges, and the camera pose every 100 steps;
- prints the simulator ground-truth object pose in world and environment frames;
- saves RGB and normalized depth previews every 200 steps.

Images are written inside the container to:

```text
/workspace/isaaclab/data/perception_test/
```

The RGB tensor is normally `[num_envs, height, width, channels]`. The configured
camera is 480×640, and RGB may contain a fourth alpha channel. Depth is normally
`[num_envs, height, width]`; each pixel stores metric distance to the image
plane. Depth is normalized only for the PNG preview—the original tensor remains
metric.

## 8. Combined perception and simple-API tutorial

Run:

```bash
./isaaclab.sh -p source/standalone/galaxea/jinhern/test_perception_simple_api.py \
  --num_envs 1 \
  --headless \
  --enable_cameras
```

This example:

1. reads the object's current simulator pose;
2. converts its world position to the environment-local frame;
3. gives that position to `R1Manipulator.pick()`;
4. places the object at a position relative to its initial position;
5. prints and saves camera diagnostics while the state machine runs.

The pick position is not a hardcoded XYZ coordinate. However, it is still
**simulation ground truth**, not a pose estimated from RGB or depth. A real
perception pipeline would need segmentation or object detection, camera
intrinsics, depth back-projection, and coordinate-frame transforms.

Output images are written to:

```text
/workspace/isaaclab/data/perception_simple_api/
```

## 9. Why `--headless` and `--enable_cameras` matter

Use `--headless` in Docker unless working X11 forwarding has been configured.
Without it, Isaac Sim may try to open a native window and report GLFW or
windowing-plugin errors.

Headless mode does not enable RTX camera rendering by default. Any script that
uses RGB, depth, segmentation, or other rendered sensors also needs:

```text
--enable_cameras
```

Therefore, the normal camera combination is:

```text
--headless --enable_cameras
```

## 10. Optional WebRTC visualization

Do not start `isaac-sim.headless.webrtc.sh` separately and then launch an Isaac
Lab tutorial. The tutorial itself starts Isaac Sim. Instead, enable WebRTC on
the tutorial command:

```bash
./isaaclab.sh -p source/standalone/galaxea/jinhern/test_perception.py \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --livestream 2
```

The Compose service uses host networking, which simplifies access to the
stream. On the host, open the WebRTC client supported by your Isaac Sim 4.0
installation. A commonly used local URL is:

```text
http://127.0.0.1:8211/streaming/webrtc-demo/?server=127.0.0.1
```

Only one streaming client can connect to an Isaac Sim instance. If the page
cannot connect, check that the tutorial is still running and that the port is
listening:

```bash
ss -ltnp | grep 8211
```

If the page returns `Not Found`, use the WebRTC client/URL supplied with your
specific Isaac Sim 4.0 container build; streaming client packaging can differ.

## 11. Copy generated images to the host

Because these tutorials currently write to `/workspace/isaaclab/data`, copy
their output before removing the container. Run this on the host while the
container exists:

```bash
mkdir -p ./tutorial_output
docker cp isaac-lab-base:/workspace/isaaclab/data/perception_test ./tutorial_output/
docker cp isaac-lab-base:/workspace/isaaclab/data/perception_simple_api ./tutorial_output/
```

Alternatively, change a tutorial's output directory to
`/workspace/isaaclab/data_storage/...`; `data_storage` is backed by the
`isaac-lab-data` named volume in this repository's Compose configuration.

## 12. Stop or remove the container

Exit the container shell without stopping the background container:

```bash
exit
```

To stop and remove the Compose container:

```bash
./docker/container.sh stop base
```

Your host-side source edits and Git history remain. Named Docker volumes retain
the configured caches and `data_storage`, but unmounted files elsewhere in the
container do not.

## 13. Edit with VS Code

Open the host checkout:

```bash
cd "$HOME/robotics/Galaxea_Lab"
code .
```

Files under `source/`, `docs/`, and `tools/` are bind-mounted by Compose, so
edits appear inside the running container. Confirm with:

```bash
git diff
```

Avoid editing files as root in the container when possible; that may create
host files owned by root.

## 14. Correct Isaac Lab application startup pattern

Standalone scripts must launch the application before importing Isaac Lab
runtime modules:

```python
import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import gymnasium, torch, and Isaac Lab runtime modules after this point.
```

Passing the complete `args_cli` namespace preserves options such as
`--headless`, `--enable_cameras`, and `--livestream`.

## 15. Frames and pose sources used by the tutorials

- **World frame:** the global simulator coordinate frame. Object position is
  available as `object.data.root_pos_w`.
- **Environment-local frame:** world position minus `scene.env_origins`. Actions
  for these manager-based lift environments use this local frame.
- **Ground-truth pose:** read directly from simulator state; exact apart from
  simulation numerical effects.
- **Perception-estimated pose:** calculated from sensor observations; affected by
  visibility, calibration, depth quality, and estimation errors.

The combined tutorial uses ground truth so that beginners can first validate
the camera and robot API independently. It does not claim that the camera has
located the object.

## 16. Troubleshooting

### Docker cannot see the GPU

Run on the host:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

If the first command works but the second does not, reinstall or reconfigure
NVIDIA Container Toolkit and restart Docker.

### `permission denied` when running Docker

Either use `sudo` temporarily or complete Docker's Linux post-installation steps
to add your account to the `docker` group. Log out and back in afterward.

### `GLFW initialization failed`

Isaac Sim tried to open a native window. Add:

```text
--headless
```

### Camera initialization or rendering error

Use both flags:

```text
--headless --enable_cameras
```

Also ensure that Docker can access the GPU and that no other Isaac Sim process
is consuming the required resources.

### CUDA out of memory

Stop other GPU workloads, use `--num_envs 1`, and run only one Isaac Sim process.
Check usage with `nvidia-smi`.

### First launch appears frozen

Shader compilation and extension startup can be slow on the first run. Watch
the terminal output and GPU activity before terminating it. Later launches
should reuse the named cache volumes.

### `ModuleNotFoundError` when using system Python

Run scripts through Isaac Lab:

```bash
./isaaclab.sh -p <SCRIPT.py> [OPTIONS]
```

Do not use the host's ordinary `python3` for Isaac Sim scripts.

### WebRTC does not connect

- include `--livestream 2` on the tutorial command;
- keep the tutorial process running;
- use only one client;
- confirm port `8211` is listening;
- verify that another Isaac Sim process is not already running.

### Source edit does not appear inside Docker

Confirm that the host file is under `source/`, `docs/`, or `tools/`, then check:

```bash
docker inspect isaac-lab-base
git diff
```

The default Compose file does not bind-mount every repository-root file.

## 17. Optional ROS 2 image

ROS 2 is not needed for the tutorials above. If later work requires ROS 2, use
the repository's ROS 2 image profile:

```bash
./docker/container.sh start ros2
./docker/container.sh enter ros2
```

Do not add ROS 2 complexity until the standalone Isaac Lab examples work.

## 18. Git workflow

Before editing:

```bash
git status
git pull --ff-only
git switch -c <YOUR_NAME>/<FEATURE>
```

Review and commit only the intended source files:

```bash
git diff
git add source/standalone/galaxea/jinhern/<FILE.py>
git commit -m "Add Galaxea perception tutorial"
git push -u origin <YOUR_NAME>/<FEATURE>
```

Do not commit credentials, caches, logs, generated images, or large datasets.

## Quick-start checklist

- [ ] `nvidia-smi` works on the host
- [ ] the CUDA Docker test sees the GPU
- [ ] repository is on `laptop4050-working`
- [ ] `./docker/container.sh start base` completes
- [ ] `nvidia-smi` works inside the container
- [ ] `spawn_robot.py --headless` starts successfully
- [ ] camera examples use `--headless --enable_cameras`
- [ ] only one Isaac Sim process is running
- [ ] generated data is copied before removing the container

## Minimal daily workflow

On the host:

```bash
cd "$HOME/robotics/Galaxea_Lab"
./docker/container.sh start base
./docker/container.sh enter base
```

Inside Docker:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p source/standalone/galaxea/jinhern/test_perception.py \
  --num_envs 1 --headless --enable_cameras
```

Stop the example with `Ctrl+C`, type `exit` to leave the container shell, and
copy any required generated files before removing the container.