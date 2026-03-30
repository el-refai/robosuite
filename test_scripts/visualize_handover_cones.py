#!/usr/bin/env python3
"""
Visualize the two handover cones (arm1 and arm0) in viser using the same parameters
as test_handover_step_two_cone.py. No robot env required; uses a fixed workspace center.

Usage:
  python test_scripts/visualize_handover_cones.py --cone_base_offset 0.2,0,0 --cone_base_radius 0.05
  python test_scripts/visualize_handover_cones.py --cone_base_offset 0,-0.2,0 --cone_base_radius 0.08 --x_shift 0 --y_shift 0
"""

import os
import sys
import argparse
import time
import numpy as np
import viser
import viser.transforms as vtf

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)


def parse_offset_list(offset_str):
    try:
        values = [float(x.strip()) for x in offset_str.split(",")]
        if len(values) != 3:
            raise ValueError("Offset must contain exactly 3 values (x, y, z)")
        return np.array(values)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid offset format '{offset_str}': {e}. Expected format: 'x,y,z'"
        )


def cone_surface_points(
    apex: np.ndarray,
    axis_toward_base: np.ndarray,
    depth: float,
    base_radius: float,
    n_depth: int = 12,
    n_theta: int = 24,
) -> np.ndarray:
    """Sample points on the cone lateral surface and base circle. Returns (N, 3)."""
    axis = axis_toward_base / (np.linalg.norm(axis_toward_base) + 1e-9)
    if abs(axis[2]) < 0.9:
        v1 = np.array([0.0, 0.0, 1.0])
    else:
        v1 = np.array([1.0, 0.0, 0.0])
    v1 = v1 - np.dot(v1, axis) * axis
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = np.cross(axis, v1)
    points = []
    # Lateral surface: for each d in (0, depth], circle of radius r = d * (base_radius/depth)
    for i in range(1, n_depth + 1):
        d = depth * i / n_depth
        r_max_d = d * (base_radius / depth)
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            pt = apex + d * axis + r_max_d * (np.cos(theta) * v1 + np.sin(theta) * v2)
            points.append(pt)
    # Base circle
    for j in range(n_theta):
        theta = 2.0 * np.pi * j / n_theta
        pt = apex + depth * axis + base_radius * (np.cos(theta) * v1 + np.sin(theta) * v2)
        points.append(pt)
    # Apex
    points.append(apex)
    return np.array(points, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize handover cones (arm1 and arm0) in viser with same params as test_handover_step_two_cone.py"
    )
    parser.add_argument(
        "--cone_base_offset",
        type=parse_offset_list,
        default="0.2,0.0,0.0",
        help="Offset of cone base center from handover (x,y,z). Default: 0.2,0,0",
    )
    parser.add_argument(
        "--cone_base_radius",
        type=float,
        default=0.05,
        help="Radius (m) of cone base. Default: 0.05",
    )
    parser.add_argument(
        "--x_shift",
        type=float,
        default=0.0,
        help="X shift for handover position. Default: 0",
    )
    parser.add_argument(
        "--y_shift",
        type=float,
        default=0.0,
        help="Y shift for handover position. Default: 0",
    )
    parser.add_argument(
        "--angle_shift",
        type=float,
        default=0.0,
        help="Angle (rad) for handover rotation about Z. Default: 0",
    )
    parser.add_argument(
        "--center",
        type=parse_offset_list,
        default="0.0,0.6,0.9",
        help="Workspace center (between arms), x,y,z. Default: 0,0.6,0.9",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port. Default: 8080",
    )
    parser.add_argument(
        "--no_arms",
        action="store_true",
        help="Do not load robot arms (skip panda_description dependency)",
    )
    args = parser.parse_args()

    cone_base_offset = np.array(args.cone_base_offset)
    cone_base_radius = float(args.cone_base_radius)
    x_shift = float(args.x_shift)
    y_shift = float(args.y_shift)
    angle_shift = float(args.angle_shift)
    center = np.array(args.center)

    depth = float(np.linalg.norm(cone_base_offset))
    if depth < 1e-9:
        print("cone_base_offset has zero length; cannot visualize.")
        sys.exit(1)

    # Same geometry as test_handover_step_two_cone: Rz, handover_offset, arm0_offset
    Rz_quat = np.array(
        [np.cos(angle_shift / 2), 0, 0, np.sin(angle_shift / 2)]
    )  # wxyz
    Rz = vtf.SO3(wxyz=Rz_quat).as_matrix()
    handover_offset = np.array([-0.15 - x_shift, 0.1 + y_shift, 0.0])
    handover_pos = center + Rz @ handover_offset

    # Both cones share the same apex (the handover position). Arm0 cone is mirrored (YZ and XZ) so it opens the opposite way.
    cone_base_offset_arm0 = np.array(
        [-cone_base_offset[0], -cone_base_offset[1], cone_base_offset[2]]
    )
    depth_arm0 = float(np.linalg.norm(cone_base_offset_arm0))
    if depth_arm0 < 1e-9:
        depth_arm0 = 1.0
        axis_toward_base_arm0 = Rz @ np.array([-1.0, 0.0, 0.0])
    else:
        axis_toward_base_arm0 = Rz @ cone_base_offset_arm0 / depth_arm0

    # Arm1 cone: apex = handover_pos, opens in direction of cone_base_offset
    cone_base_arm1 = handover_pos + Rz @ cone_base_offset
    axis_toward_base_arm1 = (cone_base_arm1 - handover_pos) / np.linalg.norm(
        cone_base_arm1 - handover_pos
    )

    # Build cone surface point clouds
    pts_arm1 = cone_surface_points(
        handover_pos, axis_toward_base_arm1, depth, cone_base_radius
    )
    pts_arm0 = cone_surface_points(
        handover_pos, axis_toward_base_arm0, depth_arm0, cone_base_radius
    )

    # Viser server
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Arm1 cone (e.g. green) – waypoint cone
    n1 = len(pts_arm1)
    colors_arm1 = np.zeros((n1, 3), dtype=np.uint8)
    colors_arm1[:, 1] = 200
    colors_arm1[:, 0] = 80
    server.scene.add_point_cloud(
        "/cone_arm1",
        pts_arm1,
        colors_arm1,
        point_size=0.012,
        point_shape="circle",
    )

    # Arm0 cone (e.g. blue) – receiver cone
    n0 = len(pts_arm0)
    colors_arm0 = np.zeros((n0, 3), dtype=np.uint8)
    colors_arm0[:, 2] = 255
    colors_arm0[:, 0] = 80
    server.scene.add_point_cloud(
        "/cone_arm0",
        pts_arm0,
        colors_arm0,
        point_size=0.012,
        point_shape="circle",
    )

    # Single frame for handover apex (shared by both cones)
    server.scene.add_frame(
        "/handover_apex",
        position=tuple(handover_pos),
        axes_length=0.08,
        axes_radius=0.004,
    )
    server.scene.add_frame(
        "/workspace_center",
        position=tuple(center),
        axes_length=0.06,
        axes_radius=0.003,
    )

    # Optional: add both Panda arms at table height (same z as workspace center)
    if not args.no_arms:
        try:
            from robot_descriptions.loaders.yourdfpy import load_robot_description
            from viser.extras import ViserUrdf

            urdf = load_robot_description("panda_description")
            z_base = float(center[2])
            server.scene.add_frame(
                "/arm0_base",
                position=(0.0, 0.0, z_base),
                axes_length=0.05,
                axes_radius=0.003,
            )
            server.scene.add_frame(
                "/arm1_base",
                position=(0.0, 1.2, z_base),
                axes_length=0.05,
                axes_radius=0.003,
            )
            urdf_vis_0 = ViserUrdf(server, urdf, root_node_name="/arm0_base/robot")
            urdf_vis_1 = ViserUrdf(server, urdf, root_node_name="/arm1_base/robot")
            # Default home pose (panda arm 7-dof + gripper 2-dof) so arms are visible
            home_arm = np.array(
                [
                    0.0,
                    np.pi / 16.0,
                    0.0,
                    -np.pi / 2.0 - np.pi / 3.0,
                    0.0,
                    np.pi - 0.2,
                    np.pi / 4.0,
                ],
                dtype=np.float64,
            )
            home_gripper = np.array([0.020833, -0.020833], dtype=np.float64)
            home_cfg = np.concatenate([home_arm, home_gripper])
            urdf_vis_0.update_cfg(home_cfg)
            urdf_vis_1.update_cfg(home_cfg)
            print("  Robot arms loaded (arm0 at y=0, arm1 at y=1.2).")
        except Exception as e:
            print(f"  Could not load robot arms: {e}")
            print("  Run with --no_arms to suppress this message.")

    print(f"Handover cones visualization at http://localhost:{args.port}")
    print(f"  Handover apex (shared): {handover_pos}")
    print(f"  Arm1 cone: depth={depth:.3f}m, base_radius={cone_base_radius}m (opens in +cone_base_offset)")
    print(f"  Arm0 cone: depth={depth_arm0:.3f}m (mirrored), base_radius={cone_base_radius}m (opens opposite)")
    print("Close the browser or Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Exiting.")


if __name__ == "__main__":
    main()
