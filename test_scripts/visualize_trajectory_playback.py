#!/usr/bin/env python3
"""Playback trajectory visualization for handover joint data.

Loads joint trajectories saved by test_handover_step.py and visualizes them
in an interactive viser viewer. Supports playback of both arms with a shared
timeline slider.

Usage:
  python test_scripts/visualize_trajectory_playback.py dataset/handover_yellow_neg0_28_0_065_0_0_duct_neg0_1_neg0_065_0_0
  python test_scripts/visualize_trajectory_playback.py dataset/handover_yellow_neg0_28_0_065_0_0_duct_neg0_1_neg0_065_0_0 --port 8081
  python test_scripts/visualize_trajectory_playback.py dataset/handover_yellow_neg0_28_0_065_0_0_duct_neg0_1_neg0_065_0_0 --fps 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

# Ensure we import the real pyroki (with viewer) from api/pyroki/src
_pyroki_src = Path(__file__).resolve().parent.parent / "api" / "pyroki" / "src"
if _pyroki_src.exists() and str(_pyroki_src) not in sys.path:
    sys.path.insert(0, str(_pyroki_src))
import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk


def compute_yoshikawa_index(
    robot: pk.Robot, cfg: np.ndarray, target_link_name: str
) -> float:
    """Compute Yoshikawa manipulability index for a joint configuration."""
    target_link_index = robot.links.names.index(target_link_name)
    cfg_jax = jnp.array(cfg, dtype=jnp.float64)

    def target_pos(q):
        fk = robot.forward_kinematics(q)
        pose = fk[target_link_index]  # (7,) wxyz_xyz
        return jaxlie.SE3(pose).translation()  # (3,)

    jacobian = jax.jacfwd(target_pos)(cfg_jax)  # (3, n_joints)
    JJT = jacobian @ jacobian.T
    det = float(jnp.linalg.det(JJT))
    return float(jnp.sqrt(jnp.maximum(0.0, det)))


def load_joint_trajectories(dataset_dir: str, base_name: str = "handover"):
    """Load joint trajectories from the npz files produced by test_handover_step.py.

    Args:
        dataset_dir: Path to directory containing handover_robot0_joints.npz
            and handover_robot1_joints.npz.
        base_name: Base filename (default: handover).

    Returns:
        Tuple of (robot0_configs, robot1_configs) where each is (T, 9) float64.
        Config format: 7 arm joints + 2 gripper finger positions.
    """
    robot0_path = os.path.join(dataset_dir, f"{base_name}_robot0_joints.npz")
    robot1_path = os.path.join(dataset_dir, f"{base_name}_robot1_joints.npz")

    if not os.path.exists(robot0_path):
        raise FileNotFoundError(f"Robot0 joints not found: {robot0_path}")
    if not os.path.exists(robot1_path):
        raise FileNotFoundError(f"Robot1 joints not found: {robot1_path}")

    r0 = np.load(robot0_path)
    r1 = np.load(robot1_path)

    # Panda: 7 arm + 2 finger joints (9-DOF). Use full gripper for correct FK.
    def to_config(joints: np.ndarray, gripper: np.ndarray) -> np.ndarray:
        return np.concatenate([joints, gripper], axis=1)

    cfg0 = to_config(r0["joint_positions"], r0["gripper_positions"])
    cfg1 = to_config(r1["joint_positions"], r1["gripper_positions"])

    T = max(cfg0.shape[0], cfg1.shape[0])
    if cfg0.shape[0] < T:
        cfg0 = np.concatenate([cfg0, np.tile(cfg0[-1:], (T - cfg0.shape[0], 1))], axis=0)
    if cfg1.shape[0] < T:
        cfg1 = np.concatenate([cfg1, np.tile(cfg1[-1:], (T - cfg1.shape[0], 1))], axis=0)

    return cfg0, cfg1


def main():
    parser = argparse.ArgumentParser(
        description="Visualize handover joint trajectories in viser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to dataset directory (e.g. dataset/handover_yellow_neg0_28_0_065_0_0_duct_neg0_1_neg0_065_0_0)",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="handover",
        help="Base filename for joint files (default: handover)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port (default: 8080)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Playback rate in frames per second (default: 5, use lower for slower playback)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(root_dir, dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f"Error: {dataset_dir} is not a directory")
        sys.exit(1)

    print(f"Loading joint trajectories from {dataset_dir}...")
    cfg0, cfg1 = load_joint_trajectories(dataset_dir, base_name=args.base_name)
    T = cfg0.shape[0]
    print(f"Loaded {T} timesteps for both arms (config dim: {cfg0.shape[1]})")

    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)
    target_link_name = "panda_hand"
    n_actuated = robot.joints.num_actuated_joints
    print(f"Robot actuated joints: {n_actuated}")

    def to_robot_cfg(cfg: np.ndarray) -> np.ndarray:
        """Ensure config matches robot actuated joint count."""
        if len(cfg) == n_actuated:
            return cfg
        if len(cfg) < n_actuated:
            return np.pad(cfg, (0, n_actuated - len(cfg)), constant_values=0)
        return cfg[:n_actuated]

    # Tape handover env uses parallel config: robots at y=-0.6 and y=+0.6 (1.2m apart)
    arm1_offset_y = 1.2

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=4, height=4, cell_size=0.1)

    # Arm 0 at origin
    urdf_vis_0 = ViserUrdf(server, urdf, root_node_name="/arm0")
    # manip_ellipse_0 = pk.viewer.ManipulabilityEllipse(
    #     server,
    #     robot,
    #     root_node_name="/manip_arm0",
    #     target_link_name=target_link_name,
    #     scaling_factor=0.25,
    #     color=(100, 200, 255),
    # )
    # Yoshikawa displays: use add_text for reliable viser GUI updates
    with server.gui.add_folder("Yoshikawa Index"):
        yoshikawa_display_0 = server.gui.add_text(
            "Arm 0", initial_value="0.0000", disabled=True
        )
        yoshikawa_display_1 = server.gui.add_text(
            "Arm 1", initial_value="0.0000", disabled=True
        )

    # Arm 1 offset to match tape handover env (parallel config, 1.2m apart)
    server.scene.add_frame("/arm1_base", position=(0.0, arm1_offset_y, 0.0))
    urdf_vis_1 = ViserUrdf(server, urdf, root_node_name="/arm1_base/robot")
    # manip_ellipse_1 = pk.viewer.ManipulabilityEllipse(
    #     server,
    #     robot,
    #     root_node_name="/arm1_base/robot/manip_arm1",
    #     target_link_name=target_link_name,
    #     scaling_factor=0.25,
    #     color=(255, 200, 100),
    # )

    # GUI controls
    slider = server.gui.add_slider(
        "Timestep", min=0, max=T - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    print(f"Viser playback at http://localhost:{args.port}")
    print("Use the slider to scrub, or enable Playing for automatic playback.")

    dt = 1.0 / args.fps
    while True:
        if playing.value:
            slider.value = (slider.value + 1) % T

        t = int(slider.value)
        cfg0_t = cfg0[t]
        cfg1_t = cfg1[t]
        cfg0_robot = to_robot_cfg(cfg0_t)
        cfg1_robot = to_robot_cfg(cfg1_t)

        # manip_ellipse_0.update(cfg0_robot)
        # manip_ellipse_1.update(cfg1_robot)

        # Yoshikawa index display (use text for reliable viser updates)
        y0 = compute_yoshikawa_index(robot, cfg0_robot, target_link_name)
        yoshikawa_display_0.value = f"{y0:.4f}"
        y1 = compute_yoshikawa_index(robot, cfg1_robot, target_link_name)
        yoshikawa_display_1.value = f"{y1:.4f}"

        urdf_vis_0.update_cfg(cfg0_robot)
        urdf_vis_1.update_cfg(cfg1_robot)
        time.sleep(dt)
        


if __name__ == "__main__":
    main()
