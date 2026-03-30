import pathlib
import time
from typing import Any
import sys

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from robosuite.environments.custom.base_env import BaseEnv
from api import pyroki_snippets as pks  # type: ignore
from api.base_api import ApiBase
# from api.grasp_graspnet import init_contact_graspnet
# from api.owlvit import init_owlvit
# from api.pyroki import init_pyroki

# from api.pyroki_context import get_pyroki_context  # type: ignore
# from api.sam2 import init_sam2
# from api.sam3 import init_sam3, visualize_sam3_results
# from utils.camera_utils import obs_get_rgb
# from utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud, depth_to_rgb


# ------------------------------- Control API ------------------------------
class FrankaControlTapeHandoverPrivilegedApi(ApiBase):
    """Robot control helpers for Franka tape handover task.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(robot_name: str, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper(robot_name: str) -> None
      - close_gripper(robot_name: str) -> None
    """

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] = [0.0, 0.0, -0.107],
        use_sam3: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        from api import pyroki_snippets as pks  # type: ignore
        from api.pyroki_context import get_pyroki_context  # type: ignore
        try:
            from api import pyroki as pk
            sys.modules["pyroki"] = pk
        except ImportError:
            pass
        ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self._robot = ctx.robot
        self._target_link_name = ctx.target_link_name
        self._pks = pks
        self._vtf = vtf
        self.cfg = None
        # For Arm 1 (robot1), use same robot model but different config
        self.cfg_1 = None
        self._pending_randomization: dict | None = None
        self._skip_next_hook = False
        self._goto_counter = 0

    def functions(self) -> dict[str, Any]:
        fns = {
            "get_object_pose": self.get_object_pose,
            # "sample_grasp_pose": self.sample_grasp_pose,
            "get_arm0_gripper_pose": self.get_arm0_gripper_pose,
            "get_arm1_gripper_pose": self.get_arm1_gripper_pose,
            "goto_pose_arm0": self.goto_pose_arm0,
            "goto_pose_arm1": self.goto_pose_arm1,
            "open_gripper_arm0": self.open_gripper_arm0,
            "close_gripper_arm0": self.close_gripper_arm0,
            "open_gripper_arm1": self.open_gripper_arm1,
            "close_gripper_arm1": self.close_gripper_arm1,
            "goto_home_joint_position_arm0": self.goto_home_joint_position_arm0,
            "goto_home_joint_position_arm1": self.goto_home_joint_position_arm1,
            "get_arm_base_midpoint_z": self.get_arm_base_midpoint_z,
            "get_handover_params": self.get_handover_params,
            "set_randomization": self.set_randomization,
        }
        return fns

    def get_handover_params(self) -> dict:
        """Return live handover geometry parameters.

        When running with viser, these values are driven by the sidebar sliders.
        Falls back to the env's stored defaults (set from CLI args) otherwise.
        """
        params = getattr(self._env, "handover_params", {})
        defaults = {"x_shift": 0.0, "y_shift": 0.0, "angle_shift": 0.0, "perturb_radius": 0.02}
        return {**defaults, **params}

    def set_randomization(self, shape_type: str, **params) -> None:
        """Declare a randomization shape for the NEXT goto_pose_arm* call.

        Args:
            shape_type: "sphere" or "cone"
            **params: shape-specific parameters
                sphere: radius (float, default 0.05),
                        offset (ndarray (3,), default zeros) — displacement
                        of the sphere center from the goto target in robot0 frame.
                cone: depth (float, default 0.2), base_radius (float, default 0.05)
                keep_endpoint (bool, default True): when True the randomized
                    point is an intermediate waypoint before the original goto
                    target; when False the randomized point replaces the target.
        """
        config = {"type": shape_type, **params}
        config.setdefault("keep_endpoint", True)
        if shape_type == "sphere":
            config.setdefault("radius", 0.05)
        elif shape_type == "cone":
            config.setdefault("depth", 0.2)
            config.setdefault("base_radius", 0.05)
        else:
            raise ValueError(f"Unknown randomization type: {shape_type!r}. Use 'sphere' or 'cone'.")
        self._pending_randomization = config

    def get_arm_base_midpoint_z(self) -> float:
        """Z-coordinate of the midpoint between the two arm bases, in robot0's base frame.
        Arm 0 base is at the origin; arm 1 base is transformed into robot0 frame and averaged.
        """
        if not hasattr(self._env, "base_link_wxyz_xyz_0") or not hasattr(self._env, "base_link_wxyz_xyz_1"):
            raise RuntimeError("Environment does not provide base transforms.")
        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        base1_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_1)
        base0_transform_inv = base0_transform.inverse()
        # Arm 0 base in robot0 frame is origin; arm 1 base in robot0 frame
        base1_in_robot0 = np.asarray(
            (base0_transform_inv @ self._vtf.SE3.from_translation(base1_transform.translation())).translation()
        )
        midpoint_z = (0.0 + base1_in_robot0[2]) / 2.0
        return float(midpoint_z)

    def get_object_pose(
        self, object_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of an object in the environment from a natural language description.
        The object's pose will be returned in *robot0's frame*.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()

        # Get base transform for robot0 (world to robot0 base frame)
        if not hasattr(self._env, "base_link_wxyz_xyz_0"):
            raise RuntimeError("Environment does not provide base transforms.")
        
        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        base0_transform_inv = base0_transform.inverse()

        if (
            "yellow tape" in object_name.lower()
        ):  # TODO: Slightly problematic that these are hardcoded language descriptions
            # Transform from world frame to robot0's base frame
            # Robosuite returns quaternion in XYZW format, convert to WXYZ
            yellow_tape_quat_xyzw = obs["yellow_tape_quat"]
            yellow_tape_quat_wxyz = np.array([yellow_tape_quat_xyzw[3], yellow_tape_quat_xyzw[0], yellow_tape_quat_xyzw[1], yellow_tape_quat_xyzw[2]])
            
            # Create SE3 transform in world frame
            yellow_tape_pose_world = self._vtf.SE3.from_rotation_and_translation(
                rotation=self._vtf.SO3(wxyz=yellow_tape_quat_wxyz),
                translation=obs["yellow_tape_pos"],
            )
            
            # Transform to robot0's base frame
            yellow_tape_pose_robot0 = base0_transform_inv @ yellow_tape_pose_world
            
            return (
                yellow_tape_pose_robot0.translation(),
                yellow_tape_pose_robot0.rotation().wxyz,
            )
        elif "duct tape" in object_name.lower():
            # Transform from world frame to robot0's base frame
            # Robosuite returns quaternion in XYZW format, convert to WXYZ
            duct_tape_quat_xyzw = obs["duct_tape_quat"]
            duct_tape_quat_wxyz = np.array([duct_tape_quat_xyzw[3], duct_tape_quat_xyzw[0], duct_tape_quat_xyzw[1], duct_tape_quat_xyzw[2]])
            
            # Create SE3 transform in world frame
            duct_tape_pose_world = self._vtf.SE3.from_rotation_and_translation(
                rotation=self._vtf.SO3(wxyz=duct_tape_quat_wxyz),
                translation=obs["duct_tape_pos"],
            )
            
            # Transform to robot0's base frame
            duct_tape_pose_robot0 = base0_transform_inv @ duct_tape_pose_world
            
            return (
                duct_tape_pose_robot0.translation(),
                duct_tape_pose_robot0.rotation().wxyz,
            )
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters, in robot0's base frame.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()

        # Get base transform for robot0 (world to robot0 base frame)
        if not hasattr(self._env, "base_link_wxyz_xyz_0"):
            raise RuntimeError("Environment does not provide base transforms.")
        
        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        base0_transform_inv = base0_transform.inverse()

        if "yellow tape" in object_name.lower():
            # Transform yellow tape position from world frame to robot0's base frame
            yellow_tape_pose_world = self._vtf.SE3.from_translation(obs["yellow_tape_pos"])
            yellow_tape_pose_robot0 = base0_transform_inv @ yellow_tape_pose_world
            return yellow_tape_pose_robot0.translation(), np.array([0, 0, 1, 0])
        elif "duct tape" in object_name.lower():
            # Transform duct tape position from world frame to robot0's base frame
            duct_tape_pose_world = self._vtf.SE3.from_translation(obs["duct_tape_pos"])
            duct_tape_pose_robot0 = base0_transform_inv @ duct_tape_pose_world
            return duct_tape_pose_robot0.translation(), np.array([0, 0, 1, 0])
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def get_arm0_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of the gripper for arm 0."""
        obs = self._env.get_observation()
        if "robot0_cartesian_pos" not in obs:
            raise ValueError("Environment does not provide robot0_cartesian_pos.")
        return obs["robot0_cartesian_pos"][:3], obs["robot0_cartesian_pos"][3:7]

    def get_arm1_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of the gripper for arm 1."""
        obs = self._env.get_observation()
        if "robot1_cartesian_pos" not in obs:
            raise ValueError("Environment does not provide robot1_cartesian_pos.")
        return obs["robot1_cartesian_pos"][:3], obs["robot1_cartesian_pos"][3:7]

    def _viser_goto_hook(
        self, label: str, pos_robot0: np.ndarray, quat_wxyz: np.ndarray
    ) -> np.ndarray:
        """Show the goto target in viser, optionally pause and allow the user to drag it.

        If a pending randomization was set via set_randomization(), it is consumed
        here: when paused the user can edit the shape params; a waypoint is sampled
        and executed before the main goto proceeds.

        Returns the (possibly adjusted) target position in robot0 frame.
        """
        if self._skip_next_hook:
            self._skip_next_hook = False
            return pos_robot0

        rand_config = self._pending_randomization
        self._pending_randomization = None

        current_goto_idx = self._goto_counter
        self._goto_counter += 1

        interaction = getattr(self._env, "viser_interaction", None)
        if interaction is None:
            if rand_config is not None:
                self._sample_and_execute_waypoint(label, pos_robot0, quat_wxyz, rand_config)
                if not rand_config.get("keep_endpoint", True):
                    arm_key = "arm0" if "arm0" in label else "arm1"
                    cur, _ = (self.get_arm0_gripper_pose() if arm_key == "arm0"
                              else self.get_arm1_gripper_pose())
                    return cur
            return pos_robot0

        base0 = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        world_pos = (base0 @ self._vtf.SE3.from_translation(pos_robot0)).translation()
        world_wxyz = (base0.rotation() @ self._vtf.SO3(wxyz=quat_wxyz)).wxyz

        interaction.label = f"**{label}**  `{world_pos.round(3)}`"
        interaction.target_world_pos = world_pos.copy()
        interaction.target_world_wxyz = np.array(world_wxyz)
        interaction.goto_index = current_goto_idx

        if rand_config is not None:
            arm_key = "arm0" if "arm0" in label else "arm1"
            current_pos, _ = (self.get_arm0_gripper_pose() if arm_key == "arm0"
                              else self.get_arm1_gripper_pose())
            approach = pos_robot0 - current_pos
            approach_norm = np.linalg.norm(approach)
            axis_r0 = approach / approach_norm if approach_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            axis_world = np.asarray(base0.rotation().as_matrix() @ axis_r0)
            rand_config["_axis"] = axis_world
            rand_config.setdefault("keep_endpoint", True)

        if rand_config is not None:
            interaction.randomization = rand_config

        if not interaction.step_mode:
            if hasattr(self._env, "_viser_push_interaction"):
                self._env._viser_push_interaction()
            if rand_config is not None:
                self._sample_and_execute_waypoint(label, pos_robot0, quat_wxyz, rand_config)
                interaction.randomization = None
                if not rand_config.get("keep_endpoint", True):
                    arm_key = "arm0" if "arm0" in label else "arm1"
                    cur, _ = (self.get_arm0_gripper_pose() if arm_key == "arm0"
                              else self.get_arm1_gripper_pose())
                    return cur
            return pos_robot0

        interaction.paused = True
        interaction.resume_event.clear()
        if hasattr(self._env, "_viser_push_interaction"):
            self._env._viser_push_interaction()

        rand_label = f" [{rand_config['type']}]" if rand_config else ""
        print(f"[viser] Paused at: {label}{rand_label}  world={world_pos.round(3)}")
        interaction.resume_event.wait()
        print(f"[viser] Resumed.")

        final_config = interaction.randomization
        if final_config is not None:
            if "_axis" not in final_config:
                arm_key = "arm0" if "arm0" in label else "arm1"
                cur, _ = (self.get_arm0_gripper_pose() if arm_key == "arm0"
                          else self.get_arm1_gripper_pose())
                approach = pos_robot0 - cur
                a_norm = np.linalg.norm(approach)
                axis_r0 = approach / a_norm if a_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
                final_config["_axis"] = np.asarray(base0.rotation().as_matrix() @ axis_r0)
            self._sample_and_execute_waypoint(label, pos_robot0, quat_wxyz, final_config)
            interaction.randomization = None
            if not final_config.get("keep_endpoint", True):
                arm_key = "arm0" if "arm0" in label else "arm1"
                cur, _ = (self.get_arm0_gripper_pose() if arm_key == "arm0"
                          else self.get_arm1_gripper_pose())
                return cur

        ctrl = getattr(self._env, "_target_ctrl", None)
        if ctrl is not None:
            adjusted_world = np.array(ctrl.position)
            adjusted_robot0 = (base0.inverse() @ self._vtf.SE3.from_translation(adjusted_world)).translation()
            return np.asarray(adjusted_robot0, dtype=np.float64)

        return pos_robot0

    def _sample_and_execute_waypoint(
        self,
        label: str,
        target_pos_robot0: np.ndarray,
        target_quat_wxyz: np.ndarray,
        config: dict,
    ) -> None:
        """Sample a waypoint from the randomization shape and execute it via goto.

        If the user dragged the randomization gizmo, ``_center_world`` is set
        and the shape center is that world position (converted to robot0 frame).
        Otherwise the shape is centred at ``target_pos_robot0``.

        Sphere: lateral offset perpendicular to approach direction.
        Cone: random point inside cone volume with apex at shape center,
              axis toward the target.
        """
        arm_key = "arm0" if "arm0" in label else "arm1"
        current_pos, _ = (self.get_arm0_gripper_pose() if arm_key == "arm0"
                          else self.get_arm1_gripper_pose())

        # Resolve shape center (user may have dragged the gizmo)
        center_world = config.get("_center_world")
        if center_world is not None:
            base0 = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
            shape_center = np.asarray(
                (base0.inverse() @ self._vtf.SE3.from_translation(center_world)).translation(),
                dtype=np.float64,
            )
        else:
            shape_center = target_pos_robot0.copy()
            offset = config.get("offset")
            if offset is not None:
                shape_center = shape_center + np.asarray(offset, dtype=np.float64)

        shape = config.get("type")

        if shape == "sphere":
            radius = config.get("radius", 0.05)
            if radius <= 0:
                return
            approach = target_pos_robot0 - current_pos
            approach_hat = approach / (np.linalg.norm(approach) + 1e-9)
            up = np.array([0.0, 0.0, 1.0])
            lateral = np.cross(approach_hat, up)
            ln = np.linalg.norm(lateral)
            lateral = lateral / ln if ln > 1e-6 else np.array([1.0, 0.0, 0.0])
            vert = np.cross(lateral, approach_hat)
            side = np.random.choice([-1.0, 1.0])
            theta = np.random.uniform(-np.pi / 6, np.pi / 6)
            direction = side * np.cos(theta) * lateral + np.sin(theta) * vert
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            waypoint = shape_center + radius * direction

        elif shape == "cone":
            depth = config.get("depth", 0.2)
            base_radius = config.get("base_radius", 0.05)
            if depth <= 0 or base_radius <= 0:
                return
            approach = target_pos_robot0 - current_pos
            axis = approach / (np.linalg.norm(approach) + 1e-9)
            ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            v1 = ref - np.dot(ref, axis) * axis
            v1 = v1 / (np.linalg.norm(v1) + 1e-9)
            v2 = np.cross(axis, v1)
            d = depth * np.cbrt(np.random.uniform())
            r_max = d * (base_radius / depth)
            r = r_max * np.sqrt(np.random.uniform())
            phi = np.random.uniform(0, 2 * np.pi)
            waypoint = shape_center + d * axis + r * (np.cos(phi) * v1 + np.sin(phi) * v2)

        else:
            return

        self._skip_next_hook = True
        goto_fn = self.goto_pose_arm0 if arm_key == "arm0" else self.goto_pose_arm1
        goto_fn(waypoint, target_quat_wxyz)

    def goto_pose_arm0(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics for Arm 0 (robot0)."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        pos = self._viser_goto_hook("goto_pose_arm0", pos, quat_wxyz)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        # Get current joint state for arm0 to use as initial configuration if cfg is None
        initial_cfg = None
        if self.cfg is None:
            obs = self._env.get_observation()
            if "robot0_joint_pos" in obs:
                current_joints = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
                # Use only the 7 arm joints (no gripper) as initial configuration
                # The IK solver expects shape (robot.joints.num_actuated_joints,) which is 7
                initial_cfg = current_joints[:7] if len(current_joints) >= 7 else current_joints

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg is None:
                try:
                    self.cfg = self._pks.solve_ik(
                        robot=self._robot,
                        target_link_name=self._target_link_name,
                        target_position=z_offset_pos,
                        target_wxyz=quat_wxyz,
                        initial_cfg=initial_cfg,
                    )
                except Exception as e:
                    # Fallback: try without initial config if it fails
                    if initial_cfg is not None:
                        try:
                            self.cfg = self._pks.solve_ik(
                                robot=self._robot,
                                target_link_name=self._target_link_name,
                                target_position=z_offset_pos,
                                target_wxyz=quat_wxyz,
                                initial_cfg=None,
                            )
                        except Exception:
                            raise RuntimeError(f"IK solving failed for arm0: {e}")
                    else:
                        raise RuntimeError(f"IK solving failed for arm0: {e}")
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg,
                )
            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            print(f"joints from ik: {joints_z_offset.shape}")
            self._env.move_to_joints_blocking(joints_z_offset)

        if self.cfg is None:
            try:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=offset_pos,
                    target_wxyz=quat_wxyz,
                    initial_cfg=initial_cfg,
                )
            except Exception as e:
                # Fallback: try without initial config if it fails
                if initial_cfg is not None:
                    try:
                        self.cfg = self._pks.solve_ik(
                            robot=self._robot,
                            target_link_name=self._target_link_name,
                            target_position=offset_pos,
                            target_wxyz=quat_wxyz,
                            initial_cfg=None,
                        )
                    except Exception:
                        raise RuntimeError(f"IK solving failed for arm0: {e}")
                else:
                    raise RuntimeError(f"IK solving failed for arm0: {e}")
        else:
            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def open_gripper_arm0(self) -> None:
        """Open gripper fully for Arm 0 (robot0)."""
        self._env._set_gripper(1.0)
        for _ in range(40):
            self._env._step_once()

    def close_gripper_arm0(self) -> None:
        """Close gripper fully for Arm 0 (robot0)."""
        self._env._set_gripper(0.0)
        for _ in range(60):
            self._env._step_once()

    def goto_pose_arm1(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics for Arm 1 (robot1)."""
        if not hasattr(self._env, "move_to_joints_blocking_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        position = self._viser_goto_hook(
            "goto_pose_arm1", np.asarray(position, dtype=np.float64).reshape(3),
            np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4),
        )

        if not hasattr(self._env, "base_link_wxyz_xyz_0") or not hasattr(self._env, "base_link_wxyz_xyz_1"):
            raise RuntimeError("Environment does not provide base transforms.")

        pose_arm0_base = self._vtf.SE3.from_rotation_and_translation(
            rotation=self._vtf.SO3(wxyz=quaternion_wxyz),
            translation=position,
        )
        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        pose_world = base0_transform @ pose_arm0_base

        base1_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_1)
        base1_transform_inv = base1_transform.inverse()
        pose_arm1_base = base1_transform_inv @ pose_world

        pos = np.asarray(pose_arm1_base.translation(), dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(pose_arm1_base.rotation().wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        # Get current joint state for arm1 to use as initial configuration if cfg_1 is None
        initial_cfg = None
        if self.cfg_1 is None:
            obs = self._env.get_observation()
            if "robot1_joint_pos" in obs:
                current_joints = np.asarray(obs["robot1_joint_pos"], dtype=np.float64)
                # Use only the 7 arm joints (no gripper) as initial configuration
                # The IK solver expects shape (robot.joints.num_actuated_joints,) which is 7
                initial_cfg = current_joints[:7] if len(current_joints) >= 7 else current_joints

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg_1 is None:
                try:
                    self.cfg_1 = self._pks.solve_ik(
                        robot=self._robot,
                        target_link_name=self._target_link_name,
                        target_position=z_offset_pos,
                        target_wxyz=quat_wxyz,
                        initial_cfg=initial_cfg,
                    )
                except Exception as e:
                    # Fallback: try without initial config if it fails
                    if initial_cfg is not None:
                        try:
                            self.cfg_1 = self._pks.solve_ik(
                                robot=self._robot,
                                target_link_name=self._target_link_name,
                                target_position=z_offset_pos,
                                target_wxyz=quat_wxyz,
                                initial_cfg=None,
                            )
                        except Exception:
                            raise RuntimeError(f"IK solving failed for arm1: {e}")
                    else:
                        raise RuntimeError(f"IK solving failed for arm1: {e}")
            else:
                self.cfg_1 = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg_1,
                )
            joints_z_offset = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking_arm1(joints_z_offset)

        if self.cfg_1 is None:
            try:
                self.cfg_1 = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=offset_pos,
                    target_wxyz=quat_wxyz,
                    initial_cfg=initial_cfg,
                )
            except Exception as e:
                # Fallback: try without initial config if it fails
                if initial_cfg is not None:
                    try:
                        self.cfg_1 = self._pks.solve_ik(
                            robot=self._robot,
                            target_link_name=self._target_link_name,
                            target_position=offset_pos,
                            target_wxyz=quat_wxyz,
                            initial_cfg=None,
                        )
                    except Exception:
                        raise RuntimeError(f"IK solving failed for arm1: {e}")
                else:
                    raise RuntimeError(f"IK solving failed for arm1: {e}")
        else:
            self.cfg_1 = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=self.cfg_1,
            )
        joints = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm1(joints)

    def open_gripper_arm1(self) -> None:
        """Open gripper fully for Arm 1 (robot1)."""
        if not hasattr(self._env, "_set_gripper_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        self._env._set_gripper_arm1(1.0)
        for _ in range(40):
            self._env._step_once()

    def close_gripper_arm1(self) -> None:
        """Close gripper fully for Arm 1 (robot1)."""
        if not hasattr(self._env, "_set_gripper_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        self._env._set_gripper_arm1(0.0)
        for _ in range(60):
            self._env._step_once()
    
    def goto_home_joint_position_arm0(self) -> None:
        """Return the arm to its reset joint configuration with high manipulability"""
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            # Try to get from robosuite env
            if hasattr(self._env, "robosuite_env") and hasattr(self._env.robosuite_env, "robots"):
                 home = self._env.robosuite_env.robots[0].init_qpos
        
        if home is None:
            raise RuntimeError("Home joint position is unavailable in the current environment.")
            
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
        self.cfg = None

    def goto_home_joint_position_arm1(self) -> None:
        """Return the arm 1 to its reset joint configuration with high manipulability"""
        home = getattr(self._env, "home_joint_position_1", None)
        if home is None:
            # Try to get from robosuite env
            if hasattr(self._env, "robosuite_env") and hasattr(self._env.robosuite_env, "robots") and len(self._env.robosuite_env.robots) > 1:
                 home = self._env.robosuite_env.robots[1].init_qpos
        
        if home is None:
            raise RuntimeError("Home joint position for arm 1 is unavailable in the current environment.")
            
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm1(joints)
        self.cfg_1 = None