import sys
import time
from typing import Any

import numpy as np
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as SciRotation

from robosuite.environments.custom.base_env import (
    BaseEnv,
)
from api import pyroki_snippets as pks  # type: ignore
from api.base_api import ApiBase


# ------------------------------- Control API ------------------------------
class FrankaControlTapeHandoverPrivilegedApi(ApiBase):
    """Robot control helpers for Franka tape handover task.

    Provides policy-optimized IK (with manipulability + smoothness costs),
    trajectory post-optimization for better policy learnability, and viser
    visualisation of manipulability ellipsoids and trajectory paths.

    Oracle API functions:
      - get_object_pose / sample_grasp_pose / get_arm*_gripper_pose
      - goto_pose_arm0 / goto_pose_arm1 (manipulability-aware IK)
      - open/close_gripper_arm0/arm1
      - goto_home_joint_position_arm0/arm1

    Post-processing methods (for policy training):
      - get/clear_trajectory_history, optimize_recorded_trajectory
      - optimize_trajectory, compute_trajectory_metrics
      - visualize_trajectories (interactive viser with ManipulabilityEllipse)
    """

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] = [0.0, 0.0, -0.107],
        use_sam3: bool = True,
        debug: bool = False,
        continuity_weight: float = 0.1,
        manipulability_weight: float = 0.5,
        enable_viser: bool = False,
        viser_port: int = 8080,
    ) -> None:
        super().__init__(env)
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)

        # Lazy-import to keep startup light
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

        # IK optimisation weights
        self._continuity_weight = continuity_weight
        self._manipulability_weight = manipulability_weight

        # Per-arm joint configuration state
        self.cfg = None      # arm0
        self.cfg_1 = None    # arm1

        # Trajectory history buffers (full IK configs including gripper joint)
        self._traj_history_0: list[np.ndarray] = []
        self._traj_history_1: list[np.ndarray] = []

        # Optional real-time viser visualisation
        self._viser_server = None
        self._manip_ellipse_0 = None
        self._manip_ellipse_1 = None
        self._urdf_vis_0 = None
        self._urdf_vis_1 = None
        self._manip_display_0 = None
        self._manip_display_1 = None
        if enable_viser:
            self._init_viser(viser_port)

    # ------------------------------------------------------------------
    # Real-time viser helpers
    # ------------------------------------------------------------------

    def _init_viser(self, port: int) -> None:
        """Initialize viser server for real-time manipulability visualisation."""
        import viser
        from viser.extras import ViserUrdf
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        try:
            from api import pyroki as pk
            sys.modules["pyroki"] = pk
        except ImportError:
            pass

        urdf = load_robot_description("panda_description")
        self._viser_server = viser.ViserServer(port=port)
        self._viser_server.scene.add_grid("/ground", width=2, height=2)

        self._urdf_vis_0 = ViserUrdf(self._viser_server, urdf, root_node_name="/arm0")
        # self._manip_ellipse_0 = pk.viewer.ManipulabilityEllipse(
        #     self._viser_server, self._robot,
        #     root_node_name="/manip0",
        #     target_link_name=self._target_link_name,
        #     color=(100, 200, 255),
        # )
        self._viser_server.scene.add_frame("/arm1_base", position=(0.0, 1.5, 0.0))
        self._urdf_vis_1 = ViserUrdf(
            self._viser_server, urdf, root_node_name="/arm1_base/robot"
        )
        # self._manip_ellipse_1 = pk.viewer.ManipulabilityEllipse(
        #     self._viser_server, self._robot,
        #     root_node_name="/manip1",
        #     target_link_name=self._target_link_name,
        #     color=(255, 200, 100),
        # )
        self._manip_display_0 = self._viser_server.gui.add_number(
            "Arm 0 Manipulability", 0.0, disabled=True
        )
        self._manip_display_1 = self._viser_server.gui.add_number(
            "Arm 1 Manipulability", 0.0, disabled=True
        )

    def _update_viser(self, arm_id: int, cfg: np.ndarray) -> None:
        """Update viser display after an IK solve."""
        if self._viser_server is None:
            return
        import jax.numpy as jnp
        if arm_id == 0 and self._urdf_vis_0 is not None:
            self._urdf_vis_0.update_cfg(cfg)
            self._manip_ellipse_0.update(jnp.array(cfg))
            if self._manip_display_0 is not None:
                self._manip_display_0.value = self._manip_ellipse_0.manipulability
        elif arm_id == 1 and self._urdf_vis_1 is not None:
            self._urdf_vis_1.update_cfg(cfg)
            self._manip_ellipse_1.update(jnp.array(cfg))
            if self._manip_display_1 is not None:
                self._manip_display_1.value = self._manip_ellipse_1.manipulability

    # ------------------------------------------------------------------
    # Previous-configuration helpers
    # ------------------------------------------------------------------

    def _get_prev_cfg_arm0(self) -> np.ndarray:
        """Current joint config for arm0; falls back to observation."""
        if self.cfg is not None:
            return self.cfg
        obs = self._env.get_observation()
        q = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
        n = self._robot.joints.num_actuated_joints
        if len(q) < n:
            return np.pad(q, (0, n - len(q)), constant_values=0)
        return q[:n]

    def _get_prev_cfg_arm1(self) -> np.ndarray:
        """Current joint config for arm1; falls back to observation."""
        if self.cfg_1 is not None:
            return self.cfg_1
        obs = self._env.get_observation()
        q = np.asarray(obs["robot1_joint_pos"], dtype=np.float64)
        n = self._robot.joints.num_actuated_joints
        if len(q) < n:
            return np.pad(q, (0, n - len(q)), constant_values=0)
        return q[:n]

    # ------------------------------------------------------------------
    # Oracle API functions
    # ------------------------------------------------------------------

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
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

    # ---- Object / gripper pose queries ----

    def get_object_pose(
        self, object_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of an object in the environment from a natural language description.
        The object's pose will be returned in *robot0's frame*.
        The quaternion from get_object_pose may be unreliable, so disregard it
        and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper
        down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()

        if not hasattr(self._env, "base_link_wxyz_xyz_0"):
            raise RuntimeError("Environment does not provide base transforms.")

        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        base0_transform_inv = base0_transform.inverse()

        if "yellow tape" in object_name.lower():
            yellow_tape_quat_xyzw = obs["yellow_tape_quat"]
            yellow_tape_quat_wxyz = np.array([
                yellow_tape_quat_xyzw[3], yellow_tape_quat_xyzw[0],
                yellow_tape_quat_xyzw[1], yellow_tape_quat_xyzw[2],
            ])
            yellow_tape_pose_world = self._vtf.SE3.from_rotation_and_translation(
                rotation=self._vtf.SO3(wxyz=yellow_tape_quat_wxyz),
                translation=obs["yellow_tape_pos"],
            )
            yellow_tape_pose_robot0 = base0_transform_inv @ yellow_tape_pose_world
            return (
                yellow_tape_pose_robot0.translation(),
                yellow_tape_pose_robot0.rotation().wxyz,
            )
        elif "duct tape" in object_name.lower():
            duct_tape_quat_xyzw = obs["duct_tape_quat"]
            duct_tape_quat_wxyz = np.array([
                duct_tape_quat_xyzw[3], duct_tape_quat_xyzw[0],
                duct_tape_quat_xyzw[1], duct_tape_quat_xyzw[2],
            ])
            duct_tape_pose_world = self._vtf.SE3.from_rotation_and_translation(
                rotation=self._vtf.SO3(wxyz=duct_tape_quat_wxyz),
                translation=obs["duct_tape_pos"],
            )
            duct_tape_pose_robot0 = base0_transform_inv @ duct_tape_pose_world
            return (
                duct_tape_pose_robot0.translation(),
                duct_tape_pose_robot0.rotation().wxyz,
            )
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters, in robot0's base frame.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()

        if not hasattr(self._env, "base_link_wxyz_xyz_0"):
            raise RuntimeError("Environment does not provide base transforms.")

        base0_transform = self._vtf.SE3(wxyz_xyz=self._env.base_link_wxyz_xyz_0)
        base0_transform_inv = base0_transform.inverse()

        if "yellow tape" in object_name.lower():
            yt_world = self._vtf.SE3.from_translation(obs["yellow_tape_pos"])
            yt_r0 = base0_transform_inv @ yt_world
            return yt_r0.translation(), np.array([0, 0, 1, 0])
        elif "duct tape" in object_name.lower():
            dt_world = self._vtf.SE3.from_translation(obs["duct_tape_pos"])
            dt_r0 = base0_transform_inv @ dt_world
            return dt_r0.translation(), np.array([0, 0, 1, 0])
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

    # ---- Arm 0 motion ----

    def goto_pose_arm0(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Go to pose using policy-optimized IK for Arm 0 (robot0).

        Uses manipulability-aware IK with continuity cost for smooth,
        learnable trajectories.  Records each configuration to the
        trajectory history buffer.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
            prev = self._get_prev_cfg_arm0()
            self.cfg = self._pks.solve_ik_policy_optimized(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=z_offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=prev,
                initial_cfg=prev,
                continuity_weight=self._continuity_weight,
                manipulability_weight=self._manipulability_weight,
            )
            self._traj_history_0.append(self.cfg.copy())
            self._update_viser(0, self.cfg)
            joints_z = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking(joints_z)

        prev = self._get_prev_cfg_arm0()
        self.cfg = self._pks.solve_ik_policy_optimized(
            robot=self._robot,
            target_link_name=self._target_link_name,
            target_position=offset_pos,
            target_wxyz=quat_wxyz,
            prev_cfg=prev,
            initial_cfg=prev,
            continuity_weight=self._continuity_weight,
            manipulability_weight=self._manipulability_weight,
        )
        self._traj_history_0.append(self.cfg.copy())
        self._update_viser(0, self.cfg)
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

    # ---- Arm 1 motion ----

    def goto_pose_arm1(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Go to pose using policy-optimized IK for Arm 1 (robot1).

        Transforms pose from robot0 frame to robot1 frame, then solves
        with manipulability-aware IK.
        """
        if not hasattr(self._env, "move_to_joints_blocking_arm1"):
            raise RuntimeError("Environment does not support Arm 1 control")
        if not hasattr(self._env, "base_link_wxyz_xyz_0") or not hasattr(
            self._env, "base_link_wxyz_xyz_1"
        ):
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
        quat_wxyz = np.asarray(
            pose_arm1_base.rotation().wxyz, dtype=np.float64
        ).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
            prev = self._get_prev_cfg_arm1()
            self.cfg_1 = self._pks.solve_ik_policy_optimized(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=z_offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=prev,
                initial_cfg=prev,
                continuity_weight=self._continuity_weight,
                manipulability_weight=self._manipulability_weight,
            )
            self._traj_history_1.append(self.cfg_1.copy())
            self._update_viser(1, self.cfg_1)
            joints_z = np.asarray(self.cfg_1[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking_arm1(joints_z)

        prev = self._get_prev_cfg_arm1()
        self.cfg_1 = self._pks.solve_ik_policy_optimized(
            robot=self._robot,
            target_link_name=self._target_link_name,
            target_position=offset_pos,
            target_wxyz=quat_wxyz,
            prev_cfg=prev,
            initial_cfg=prev,
            continuity_weight=self._continuity_weight,
            manipulability_weight=self._manipulability_weight,
        )
        self._traj_history_1.append(self.cfg_1.copy())
        self._update_viser(1, self.cfg_1)
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

    # ---- Home positions ----

    def goto_home_joint_position_arm0(self) -> None:
        """Return arm 0 to its reset joint configuration with high manipulability."""
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            if hasattr(self._env, "robosuite_env") and hasattr(
                self._env.robosuite_env, "robots"
            ):
                home = self._env.robosuite_env.robots[0].init_qpos
        if home is None:
            raise RuntimeError(
                "Home joint position is unavailable in the current environment."
            )
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
        self.cfg = None

    def goto_home_joint_position_arm1(self) -> None:
        """Return arm 1 to its reset joint configuration with high manipulability."""
        home = getattr(self._env, "home_joint_position_1", None)
        if home is None:
            if (
                hasattr(self._env, "robosuite_env")
                and hasattr(self._env.robosuite_env, "robots")
                and len(self._env.robosuite_env.robots) > 1
            ):
                home = self._env.robosuite_env.robots[1].init_qpos
        if home is None:
            raise RuntimeError(
                "Home joint position for arm 1 is unavailable."
            )
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking_arm1(joints)
        self.cfg_1 = None

    # ==================================================================
    # Trajectory optimisation & analysis  (post-processing for policy)
    # ==================================================================

    def get_trajectory_history(self, arm_id: int = 0) -> np.ndarray:
        """Get recorded trajectory history for an arm.

        Args:
            arm_id: 0 for arm0, 1 for arm1.

        Returns:
            (T, num_actuated_joints) array of joint configurations.
        """
        hist = self._traj_history_0 if arm_id == 0 else self._traj_history_1
        if not hist:
            raise ValueError(f"No trajectory history for arm {arm_id}")
        return np.stack(hist)

    def clear_trajectory_history(self, arm_id: int | None = None) -> None:
        """Clear trajectory history. arm_id=None clears both."""
        if arm_id is None or arm_id == 0:
            self._traj_history_0.clear()
        if arm_id is None or arm_id == 1:
            self._traj_history_1.clear()

    def optimize_recorded_trajectory(
        self, arm_id: int = 0, dt: float = 0.02, **kwargs
    ) -> tuple[np.ndarray, dict]:
        """Optimize recorded trajectory for smoothness, manipulability, etc.

        Args:
            arm_id: 0 for arm0, 1 for arm1.
            dt: Timestep between configurations (seconds).
            **kwargs: Forwarded to pyroki_snippets.optimize_trajectory.

        Returns:
            Tuple of (optimized_trajectory, metrics).
        """
        traj = self.get_trajectory_history(arm_id)
        return self._pks.optimize_trajectory(
            robot=self._robot,
            target_link_name=self._target_link_name,
            joint_trajectory=traj,
            dt=dt,
            **kwargs,
        )

    def optimize_trajectory(
        self, joint_trajectory: np.ndarray, dt: float = 0.02, **kwargs
    ) -> tuple[np.ndarray, dict]:
        """Optimize any joint trajectory for policy learnability.

        Args:
            joint_trajectory: (T, num_actuated_joints) joint configs.
            dt: Timestep between configurations (seconds).
            **kwargs: Forwarded to pyroki_snippets.optimize_trajectory.

        Returns:
            Tuple of (optimized_trajectory, metrics).
        """
        return self._pks.optimize_trajectory(
            robot=self._robot,
            target_link_name=self._target_link_name,
            joint_trajectory=joint_trajectory,
            dt=dt,
            **kwargs,
        )

    def compute_trajectory_metrics(self, joint_trajectory: np.ndarray) -> dict:
        """Compute trajectory quality metrics (smoothness, manipulability, etc.)."""
        return self._pks.compute_trajectory_metrics(
            robot=self._robot,
            target_link_name=self._target_link_name,
            joint_trajectory=joint_trajectory,
        )

    # ==================================================================
    # Viser trajectory visualisation (with manipulability ellipsoids)
    # ==================================================================

    def _compute_ee_positions(self, joint_trajectory: np.ndarray) -> np.ndarray:
        """Compute end-effector positions along a trajectory."""
        import jax.numpy as jnp
        import jaxlie
        target_link_index = self._robot.links.names.index(self._target_link_name)
        positions = []
        for t in range(joint_trajectory.shape[0]):
            fk = self._robot.forward_kinematics(jnp.array(joint_trajectory[t]))
            pos = jaxlie.SE3(fk[target_link_index]).translation()
            positions.append(np.array(pos))
        return np.stack(positions)

    def _compute_manipulability_values(
        self, joint_trajectory: np.ndarray
    ) -> np.ndarray:
        """Compute Yoshikawa manipulability index along a trajectory."""
        import jax
        import jax.numpy as jnp
        import jaxlie
        target_link_index = self._robot.links.names.index(self._target_link_name)
        values = []
        for t in range(joint_trajectory.shape[0]):
            cfg = jnp.array(joint_trajectory[t])
            jacobian = jax.jacfwd(
                lambda q: jaxlie.SE3(
                    self._robot.forward_kinematics(q)
                ).translation()
            )(cfg)[target_link_index]
            JJT = jacobian @ jacobian.T
            m = jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT)))
            values.append(float(m))
        return np.array(values)

    def visualize_trajectories(
        self,
        traj_arm0: np.ndarray | None = None,
        traj_arm1: np.ndarray | None = None,
        original_traj_arm0: np.ndarray | None = None,
        original_traj_arm1: np.ndarray | None = None,
        port: int = 8080,
    ) -> None:
        """Launch interactive viser visualisation of trajectories with
        manipulability ellipsoids for both arms.

        Displays the robot(s) animating through the trajectory with PyRoKi's
        ManipulabilityEllipse showing manipulability at each step.  EE paths
        are drawn as point clouds coloured by manipulability (red=low,
        green=high).  If original trajectories are given, a checkbox toggles
        between optimised and original for comparison.

        Args:
            traj_arm0: (T, n_joints) optimised trajectory for arm 0.
            traj_arm1: (T, n_joints) optimised trajectory for arm 1.
            original_traj_arm0: Optional original trajectory for arm 0.
            original_traj_arm1: Optional original trajectory for arm 1.
            port: Viser server port.
        """
        import viser
        from viser.extras import ViserUrdf
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        from api import pyroki as pk
        import jax.numpy as jnp

        urdf = load_robot_description("panda_description")
        robot = self._robot
        target_link_name = self._target_link_name

        has_arm0 = traj_arm0 is not None
        has_arm1 = traj_arm1 is not None
        has_orig_0 = original_traj_arm0 is not None
        has_orig_1 = original_traj_arm1 is not None

        T = 0
        if has_arm0:
            T = max(T, traj_arm0.shape[0])
        if has_arm1:
            T = max(T, traj_arm1.shape[0])
        if T == 0:
            print("No trajectories provided to visualize.")
            return

        server = viser.ViserServer(port=port)
        server.scene.add_grid("/ground", width=4, height=4, cell_size=0.1)

        # --- Arm 0 ---
        urdf_vis_0 = manip_ellipse_0 = manip_display_0 = None
        if has_arm0:
            urdf_vis_0 = ViserUrdf(server, urdf, root_node_name="/arm0")
            manip_ellipse_0 = pk.viewer.ManipulabilityEllipse(
                server, robot,
                root_node_name="/manip_arm0",
                target_link_name=target_link_name,
                color=(100, 200, 255),
            )
            manip_display_0 = server.gui.add_number(
                "Arm 0 Manipulability", 0.0, disabled=True
            )
            ee_pos_0 = self._compute_ee_positions(traj_arm0)
            manip_0 = self._compute_manipulability_values(traj_arm0)
            self._add_trajectory_pointcloud(server, "/arm0_traj", ee_pos_0, manip_0)
            if has_orig_0:
                ee_orig_0 = self._compute_ee_positions(original_traj_arm0)
                m_orig_0 = self._compute_manipulability_values(original_traj_arm0)
                self._add_trajectory_pointcloud(
                    server, "/arm0_traj_orig", ee_orig_0, m_orig_0, visible=False
                )

        # --- Arm 1 ---
        urdf_vis_1 = manip_ellipse_1 = manip_display_1 = None
        if has_arm1:
            server.scene.add_frame("/arm1_base", position=(0.0, 1.5, 0.0))
            urdf_vis_1 = ViserUrdf(
                server, urdf, root_node_name="/arm1_base/robot"
            )
            manip_ellipse_1 = pk.viewer.ManipulabilityEllipse(
                server, robot,
                root_node_name="/manip_arm1",
                target_link_name=target_link_name,
                color=(255, 200, 100),
            )
            manip_display_1 = server.gui.add_number(
                "Arm 1 Manipulability", 0.0, disabled=True
            )
            ee_pos_1 = self._compute_ee_positions(traj_arm1)
            manip_1 = self._compute_manipulability_values(traj_arm1)
            offset = np.array([0.0, 1.5, 0.0])
            self._add_trajectory_pointcloud(
                server, "/arm1_traj", ee_pos_1 + offset, manip_1
            )
            if has_orig_1:
                ee_orig_1 = self._compute_ee_positions(original_traj_arm1)
                m_orig_1 = self._compute_manipulability_values(original_traj_arm1)
                self._add_trajectory_pointcloud(
                    server, "/arm1_traj_orig", ee_orig_1 + offset, m_orig_1,
                    visible=False,
                )

        # --- GUI ---
        slider = server.gui.add_slider(
            "Timestep", min=0, max=T - 1, step=1, initial_value=0
        )
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        show_original = server.gui.add_checkbox("Show Original", initial_value=False)

        print(f"Viser trajectory visualisation at http://localhost:{port}")

        # --- Animation loop ---
        while True:
            if playing.value:
                slider.value = (slider.value + 1) % T

            t = int(slider.value)
            use_orig = show_original.value

            if has_arm0 and urdf_vis_0 is not None:
                active_0 = (
                    original_traj_arm0 if (use_orig and has_orig_0) else traj_arm0
                )
                if t < active_0.shape[0]:
                    urdf_vis_0.update_cfg(active_0[t])
                    manip_ellipse_0.update(jnp.array(active_0[t]))
                    if manip_display_0 is not None:
                        manip_display_0.value = manip_ellipse_0.manipulability

            if has_arm1 and urdf_vis_1 is not None:
                active_1 = (
                    original_traj_arm1 if (use_orig and has_orig_1) else traj_arm1
                )
                if t < active_1.shape[0]:
                    urdf_vis_1.update_cfg(active_1[t])
                    manip_ellipse_1.update(jnp.array(active_1[t]))
                    if manip_display_1 is not None:
                        manip_display_1.value = manip_ellipse_1.manipulability

            time.sleep(1.0 / 15.0)

    @staticmethod
    def _add_trajectory_pointcloud(
        server,
        name: str,
        ee_positions: np.ndarray,
        manip_values: np.ndarray,
        visible: bool = True,
    ) -> None:
        """Add a manipulability-coloured point cloud for a trajectory path."""
        if len(ee_positions) == 0:
            return
        mn, mx = manip_values.min(), manip_values.max()
        if mx - mn > 1e-8:
            norm = (manip_values - mn) / (mx - mn)
        else:
            norm = np.ones_like(manip_values) * 0.5
        # Red (low manipulability) -> Green (high manipulability)
        colors = np.zeros((len(ee_positions), 3), dtype=np.uint8)
        colors[:, 0] = ((1.0 - norm) * 230).astype(np.uint8)
        colors[:, 1] = (norm * 230).astype(np.uint8)
        colors[:, 2] = 40
        server.scene.add_point_cloud(
            name,
            points=ee_positions.astype(np.float32),
            colors=colors,
            point_size=0.008,
            visible=visible,
        )
