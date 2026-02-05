import pathlib
import time
from typing import Any

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
        }
        return fns

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

    def goto_pose_arm0(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics for Arm 0 (robot0)."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg is None:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                )
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg,
                )
            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking(joints_z_offset)

        if self.cfg is None:
            self.cfg = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
            )
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

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg_1 is None:
                self.cfg_1 = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                )
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
            self.cfg_1 = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
            )
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