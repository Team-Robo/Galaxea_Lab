"""Generate a planning URDF for the R1 from the Isaac USD asset.

The URDF shipped in ~/URDF/R1 (r1_v2_1_0) is a different hardware revision
than the r1_DVT USD that the lift environments simulate: arm link offsets
differ by centimeters, so MoveIt FK based on that URDF is 10-20 cm off from
Isaac. This script instead reads the PhysX joint frames straight out of the
USD and emits a URDF whose kinematics match the simulated robot exactly.

The generated model contains:
  * base_link (identical to the Isaac articulation root frame),
  * both 6-DOF arm chains with the USD joint limits,
  * a ``left_flange``/``right_flange`` link that coincides with Isaac's
    ``left/right_arm_link6`` *body* frame, so poses of the flange can be
    compared 1:1 with Isaac commands and measurements.

The torso is welded in the USD, so its (fixed) transform is folded into the
first arm joint origins.

Run with any Python that has ``usd-core`` and ``numpy``:

    pip install usd-core
    python3 source/standalone/galaxea/jinhern/9_moveit2_gen_urdf.py

It writes ``r1_from_usd.urdf`` next to this script and prints an FK
self-check of the generated chains against the USD rest pose.
"""

import argparse
import math
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_USD = (
    "/home/jinhern424/jh/Galaxea_Lab/source/extensions/omni.isaac.lab_assets/"
    "data/Robots/Galaxea/r1_DVT_colored.usd"
)
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "r1_from_usd.urdf")

def quat_to_matrix(quat_wxyz):
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform(pos, quat_wxyz):
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(quat_wxyz)
    t[:3, 3] = np.asarray(pos, dtype=float)
    return t


def matrix_to_rpy(rotation):
    """Rotation matrix to URDF rpy (fixed-axis XYZ convention)."""
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2, 0])))
    if abs(rotation[2, 0]) < 1.0 - 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def origin_xml(t):
    roll, pitch, yaw = matrix_to_rpy(t[:3, :3])
    x, y, z = t[:3, 3]
    return (
        f'<origin xyz="{x:.9f} {y:.9f} {z:.9f}" '
        f'rpy="{roll:.9f} {pitch:.9f} {yaw:.9f}"/>'
    )


def read_usd(usd_path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    cache = UsdGeom.XformCache()

    def world_transform(link_name):
        prim = stage.GetPrimAtPath(f"/R1/{link_name}")
        m = cache.GetLocalToWorldTransform(prim)
        tr = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        return transform([tr[0], tr[1], tr[2]], [q.GetReal(), *q.GetImaginary()])

    torso4_world = world_transform("torso_link4")
    link6_world = {
        side: world_transform(f"{side}_arm_link6") for side in ("left", "right")
    }

    joints = {}
    for side in ("left", "right"):
        for index in range(1, 7):
            prim = stage.GetPrimAtPath(f"/R1/{'torso_link4' if index == 1 else f'{side}_arm_link{index - 1}'}/{side}_arm_joint{index}")
            if not prim:
                # Some assets nest joints under the child link instead.
                for candidate in stage.Traverse():
                    if candidate.GetName() == f"{side}_arm_joint{index}":
                        prim = candidate
                        break
            attrs = {a.GetName(): a.Get() for a in prim.GetAttributes()}
            rot0 = attrs["physics:localRot0"]
            rot1 = attrs["physics:localRot1"]
            joints[f"{side}_arm_joint{index}"] = {
                "frame0": transform(
                    list(attrs["physics:localPos0"]),
                    [rot0.GetReal(), *rot0.GetImaginary()],
                ),
                "frame1": transform(
                    list(attrs["physics:localPos1"]),
                    [rot1.GetReal(), *rot1.GetImaginary()],
                ),
                "lower": math.radians(attrs["physics:lowerLimit"]),
                "upper": math.radians(attrs["physics:upperLimit"]),
            }
    return torso4_world, link6_world, joints


def build_chains(torso4_world, joints):
    """Convert PhysX joint frames to URDF joint origins.

    URDF puts the child link frame at the joint frame, while PhysX stores a
    frame in the parent body (frame0) and one in the child body (frame1) that
    coincide at q = 0 and rotate about their local Z. Choosing the URDF link
    frame U_c = X_c * frame1_c (child body frame shifted into the joint
    frame) makes every URDF joint origin = frame1_parent^-1 * frame0_joint
    with axis (0, 0, 1), and reproduces the PhysX kinematics exactly.
    """
    chains = {}
    for side in ("left", "right"):
        chain = []
        parent_frame1 = torso4_world  # base_link -> torso_link4 weld, A = X_tl4
        for index in range(1, 7):
            joint = joints[f"{side}_arm_joint{index}"]
            origin = parent_frame1 @ joint["frame0"] if index == 1 else (
                np.linalg.inv(parent_frame1) @ joint["frame0"]
            )
            chain.append(
                {
                    "name": f"{side}_arm_joint{index}",
                    "child": f"{side}_arm_link{index}",
                    "origin": origin,
                    "lower": joint["lower"],
                    "upper": joint["upper"],
                }
            )
            parent_frame1 = joint["frame1"]
        # Fixed joint from the last URDF link frame back to the Isaac link6
        # body frame, so downstream code can work in Isaac's EE convention.
        chain.append(
            {
                "name": f"{side}_flange_joint",
                "child": f"{side}_flange",
                "origin": np.linalg.inv(parent_frame1),
                "lower": None,
                "upper": None,
            }
        )
        chains[side] = chain
    return chains


def build_urdf(chains):
    lines = [
        "<?xml version=\"1.0\"?>",
        "<!-- Generated by 9_moveit2_gen_urdf.py from r1_DVT_colored.usd.",
        "     Kinematics match the Isaac articulation; do not edit by hand. -->",
        '<robot name="r1_dvt">',
        '  <link name="base_link"/>',
    ]
    for side, chain in chains.items():
        parent = "base_link"
        for joint in chain:
            fixed = joint["lower"] is None
            lines.append(f'  <link name="{joint["child"]}"/>')
            lines.append(
                f'  <joint name="{joint["name"]}" type="{"fixed" if fixed else "revolute"}">'
            )
            lines.append(f"    {origin_xml(joint['origin'])}")
            lines.append(f'    <parent link="{parent}"/>')
            lines.append(f'    <child link="{joint["child"]}"/>')
            if not fixed:
                lines.append('    <axis xyz="0 0 1"/>')
                lines.append(
                    f'    <limit lower="{joint["lower"]:.6f}" upper="{joint["upper"]:.6f}"'
                    ' effort="87" velocity="2.175"/>'
                )
            lines.append("  </joint>")
            parent = joint["child"]
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def fk_flange(chain, joint_values):
    pose = np.eye(4)
    for joint, value in zip(chain, [*joint_values, 0.0]):
        pose = pose @ joint["origin"]
        if joint["lower"] is not None:
            rz = np.eye(4)
            c, s = math.cos(value), math.sin(value)
            rz[0, 0], rz[0, 1], rz[1, 0], rz[1, 1] = c, -s, s, c
            pose = pose @ rz
    return pose


def self_check(chains, link6_world):
    """FK of the generated chains at q=0 must land on the USD rest pose."""
    ok = True
    for side, chain in chains.items():
        flange = fk_flange(chain, [0.0] * 6)
        expected = link6_world[side]
        position_error = np.linalg.norm(flange[:3, 3] - expected[:3, 3])
        rotation_error = np.linalg.norm(flange[:3, :3] - expected[:3, :3])
        print(
            f"[self-check] {side}: flange(q=0)=({flange[0, 3]:.4f},"
            f" {flange[1, 3]:.4f}, {flange[2, 3]:.4f}),"
            f" position_error={position_error * 1000:.3f} mm,"
            f" rotation_error={rotation_error:.6f}"
        )
        ok = ok and position_error < 1e-4 and rotation_error < 1e-3
    print(f"[self-check] {'PASSED' if ok else 'FAILED'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default=DEFAULT_USD)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torso4_world, link6_world, joints = read_usd(args.usd)
    chains = build_chains(torso4_world, joints)
    urdf = build_urdf(chains)
    with open(args.output, "w") as f:
        f.write(urdf)
    print(f"wrote {args.output}")
    self_check(chains, link6_world)


if __name__ == "__main__":
    main()