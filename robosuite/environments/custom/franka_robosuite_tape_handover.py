"""Low-level Robosuite Two-Arm Handover environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's TwoArmHandover environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import robosuite as suite
import viser

# Temporary viser debugging imports
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.utils.camera_utils import get_real_depth_map

from robosuite.environments.custom.base_env import BaseEnv
from robosuite.environments.custom.utils.camera_utils import obs_get_rgb
from robosuite.environments.custom.utils.depth_utils import depth_color_to_pointcloud

os.environ.setdefault("MUJOCO_GL", "egl")


class FrankaRobosuiteTapeHandover(BaseEnv):
    def __init__(
        self,
        controller_cfg: str = "robosuite/environments/custom/configs/panda_joint_ctrl.json",
        control_freq:float = 20,
        max_steps: int = 5000,
        seed: int | None = None,
        viser_debug: bool = False,  # TODO: move the viser visualization manager into a separate class, low level env agnostic
        privileged: bool = True,
        enable_render: bool = False,
        use_wrist_cameras: bool = False,
        yellow_tape_offset: np.ndarray | None = None,
        duct_tape_offset: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.controller_cfg = controller_cfg
        self.max_steps = max_steps
        self.use_wrist_cameras = use_wrist_cameras
        self.save_camera_name = "agentview"  # Scene-level camera to show both arms
        self.render_camera_names = ["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"] if self.use_wrist_cameras else ["agentview"]  # Scene-level camera for observations
        self.segmentation_level = "instance"

        self._render_width = 512
        self._render_height = 512

        # Initialize Robosuite environment
        # TwoArmHandover requires 2 robots or 1 bimanual robot
        privileged = True
        if privileged:
            # Load controller config for both robots (same config for both)
            controller_config = load_composite_controller_config(controller=self.controller_cfg)

            # self.robosuite_env = suite.environments.manipulation.two_arm_handover.TwoArmHandover(
            #     robots=["Panda", "Panda"],  # Two separate robots for handover
            #     env_configuration="opposed",  # Robots on opposite sides of table
            #     has_renderer=True,
            #     has_offscreen_renderer=True,
            #     camera_names=self.render_camera_names,
            #     renderer="mujoco",
            #     camera_heights=self._render_height,
            #     camera_widths=self._render_width,
            #     controller_configs=[controller_config, controller_config],  # One config per robot
            #     horizon=max_steps,
            #     prehensile=True,  # Hammer starts on table
            #     reward_shaping=False,  # Use sparse reward (2.0 for success)
            #     use_object_obs=True,  # Required for hammer_pos, hammer_quat, handle_xpos observations
            #     use_camera_obs=True,  # Required for camera observations
            # )

            self.robosuite_env = suite.environments.manipulation.two_arm_tape_handover.TwoArmTapeHandover(
                robots=["Panda", "Panda"],  # Two separate robots for handover
                env_configuration="parallel",  # Robots on opposite sides of table
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=[controller_config, controller_config],  # One config per robot
                horizon=max_steps,
                reward_shaping=False,  # Use sparse reward (2.0 for success)
                use_object_obs=True,  # Required for hammer_pos, hammer_quat, handle_xpos observations
                use_camera_obs=True,  # Required for camera observations
                yellow_tape_offset=yellow_tape_offset,
                duct_tape_offset=duct_tape_offset,
                control_freq=control_freq,
                initialization_noise={"type": "sphere", "magnitude": 0.02},
            )
            # Get camera ID and modify its position and orientation
            agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
            self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [-1.2434677502317038, 4.965421871106301e-08, 2.091455182752329] #[1.5, 0.0, 2.5]
            self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.65309799, 0.2710408, -0.27104062, -0.65309811] #[0.653, 0.271, 0.271, 0.653]
        else:
            raise NotImplementedError

        # State tracking
        self._step_count = 0
        self._sim_step_count = 0
        self._rng = np.random.default_rng(seed)

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._camera_frame_buffers: dict[str, list[np.ndarray]] = {}  # Per-camera frame buffers
        self._subsample_rate = 1

        # Joint state collection
        self._collect_joint_states = False
        self._joint_state_buffer: list[dict[str, np.ndarray]] = []
        self._joint_state_collect_freq = 1  # Collect every N simulation steps

        # Robot link indices for transforms (robot0 and robot1)
        # Base links are fixed, so we cache their transforms
        self.gripper_metric_length = 0.04
        self.base_link_idx_0 = self.robosuite_env.sim.model.body_name2id("fixed_mount0_base")
        self.gripper_link_idx_0 = self.robosuite_env.sim.model.body_name2id("gripper0_right_eef")
        self.base_link_idx_1 = self.robosuite_env.sim.model.body_name2id("fixed_mount1_base")
        self.gripper_link_idx_1 = self.robosuite_env.sim.model.body_name2id("gripper1_right_eef")

        # Cache base transforms (these are constant, base doesn't move)
        self.base_link_wxyz_xyz_0 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_0],
            ]
        )
        self.base_link_wxyz_xyz_1 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx_1],
                self.robosuite_env.sim.data.xpos[self.base_link_idx_1],
            ]
        )

        # Gripper state (read from robosuite when needed, not stored)
        self._gripper_fraction_0 = 1.0  # Target gripper state for robot0
        self._gripper_fraction_1 = 1.0  # Target gripper state for robot1

        # Temporary viser debugging
        if viser_debug:
            self.viser_server = viser.ViserServer()

            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584
        self.reset()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()

        # Re-apply camera adjustment after reset (in case model was reloaded)
        agentview_cam_id = self.robosuite_env.sim.model.camera_name2id("agentview")
        self.robosuite_env.sim.model.cam_pos[agentview_cam_id] = [-1.2434677502317038, 4.965421871106301e-08, 2.091455182752329] #[1.5, 0.0, 2.5]
        self.robosuite_env.sim.model.cam_quat[agentview_cam_id] = [0.65309799, 0.2710408, -0.27104062, -0.65309811] #[0.653, 0.271, 0.271, 0.653]

        self._step_count = 0
        self._sim_step_count = 0
        
        # Clear joint state buffer on reset if collection is enabled
        if self._collect_joint_states:
            self._joint_state_buffer.clear()

        obs = self.get_observation()
        info = {
            "task_prompt": "Arm 0 should pick up the hammer, lift it, and hand it over to Arm 1. Arm 1 should then grasp the hammer handle. Quaternions are WXYZ."
        }
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._step_count += 1
        # This is a fallback; normally FrankaControlApi methods are used
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    # ----------------------- FrankaControlApi Interface -----------------------

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 1000
    ) -> None:
        """Move robot0 to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)

        steps = 0
        while steps < max_steps:
            if self._sim_step_count >= self.max_steps:
                break
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
            robot1_joints = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

            # Check convergence
            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            # Build Robosuite action for both robots
            # Each robot action = [7 joints, 1 gripper] = 8 dims
            if error > 10 * tolerance:
                perturbed_target = target + self._rng.normal(0, 0.005, target.shape)
            else:
                perturbed_target = target
            robot0_action = np.concatenate([perturbed_target, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([robot1_joints, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            self.robosuite_env.step(action)

            if hasattr(self, "viser_server"):
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            if self._collect_joint_states and self._sim_step_count % self._joint_state_collect_freq == 0:
                self._record_joint_state()

            steps += 1
            self._sim_step_count += 1

    def move_to_joints_blocking_arm1(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 1000
    ) -> None:
        """Move robot1 to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)

        steps = 0
        while steps < max_steps:
            if self._sim_step_count >= self.max_steps:
                break
            # Get current state from robosuite
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)
            robot0_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)

            # Check convergence
            error = np.linalg.norm(current - target)
            if error < tolerance:
                break
            # Build Robosuite action for both robots
            if error > 10 * tolerance:
                perturbed_target = target + self._rng.normal(0, 0.005, target.shape)
            else:
                perturbed_target = target
            robot0_action = np.concatenate([robot0_joints, [1.0 - self._gripper_fraction_0 * 2.0]])
            robot1_action = np.concatenate([perturbed_target, [1.0 - self._gripper_fraction_1 * 2.0]])
            action = np.concatenate([robot0_action, robot1_action])

            # Step the environment
            self.robosuite_env.step(action)

            if hasattr(self, "viser_server"):
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            if self._collect_joint_states and self._sim_step_count % self._joint_state_collect_freq == 0:
                self._record_joint_state()

            steps += 1
            self._sim_step_count += 1

    def _set_gripper(self, fraction: float) -> None:
        """Set target gripper opening fraction for robot0.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction_0 = float(np.clip(fraction, 0.0, 1.0))

    def _set_gripper_arm1(self, fraction: float) -> None:
        """Set target gripper opening fraction for robot1.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction_1 = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        """Execute one simulation step maintaining current joint positions and gripper states."""
        # Get current joint positions from robosuite
        robosuite_obs = self.robosuite_env._get_observations()
        robot0_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        robot1_joints = np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64)

        # Build action for both robots (maintain current joints, apply gripper targets)
        robot0_action = np.concatenate([robot0_joints, [1.0 - self._gripper_fraction_0 * 2.0]])
        robot1_action = np.concatenate([robot1_joints, [1.0 - self._gripper_fraction_1 * 2.0]])
        action = np.concatenate([robot0_action, robot1_action])

        self.robosuite_env.step(action)
        self._sim_step_count += 1

        if hasattr(self, "viser_server"):
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

        if self._collect_joint_states and self._sim_step_count % self._joint_state_collect_freq == 0:
            self._record_joint_state()

    def compute_reward(self) -> float:
        """Compute sparse handover reward.

        Returns:
            1.0 if handover is successful (only Arm 1 gripping handle, lifted above threshold), 0.0 otherwise
        """
        # Use the robosuite environment's built-in reward function
        # It returns 2.0 for success when reward_shaping=False, normalized by reward_scale/2.0
        # So max reward is 1.0 (when reward_scale=1.0)
        reward = float(self.robosuite_env.reward())

        # The robosuite reward is already normalized to [0, 1.0] for success
        return reward

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations()
        
        for camera_name in self.render_camera_names:
            if camera_name not in robosuite_obs:
                robosuite_obs[camera_name] = {}

            cam_world_wxyz_xyz = np.concatenate(
                [
                    vtf.SO3.from_matrix(
                        self.robosuite_env.sim.data.get_camera_xmat(camera_name)
                    ).wxyz,
                    self.robosuite_env.sim.data.get_camera_xpos(camera_name),
                ]
            )
            cam_robot_tf = (
                (
                    vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse()
                    @ vtf.SE3(wxyz_xyz=cam_world_wxyz_xyz)
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0),
                    translation=np.array([0, 0, 0]),
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi),
                    translation=np.array([0, 0, 0]),
                )
            )

            robosuite_obs[camera_name]["pose"] = np.concatenate(
                [
                    cam_robot_tf.translation(),
                    cam_robot_tf.rotation().wxyz,
                ]
            )
            robosuite_obs[camera_name]["pose_mat"] = cam_robot_tf.as_matrix()

            cam_id = self.robosuite_env.sim.model.camera_name2id(camera_name)
            fovy = self.robosuite_env.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)

            K = np.array(
                [[f, 0, 0.5 * self._render_width], [0, f, 0.5 * self._render_height], [0, 0, 1]]
            )
            robosuite_obs[camera_name]["intrinsics"] = K

            robosuite_obs[camera_name]["images"] = {}
            if camera_name + "_image" in robosuite_obs:
                robosuite_obs[camera_name]["images"]["rgb"] = robosuite_obs[camera_name + "_image"][
                    ::-1
                ]
            if camera_name + "_depth" in robosuite_obs:
                # converts openGL z buffer to metric
                depth_metric = get_real_depth_map(
                    self.robosuite_env.sim, robosuite_obs[camera_name + "_depth"][::-1]
                )

                robosuite_obs[camera_name]["images"]["depth"] = depth_metric
            if camera_name + "_segmentation_" + self.segmentation_level in robosuite_obs:
                robosuite_obs[camera_name]["images"]["segmentation"] = robosuite_obs[
                    camera_name + "_segmentation_" + self.segmentation_level
                ][::-1]

        # Compute gripper pose for robot0
        gripper_link_wxyz_xyz_0 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx_0],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx_0],
            ]
        )
        gripper_robot_base_0 = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse()
            @ vtf.SE3(wxyz_xyz=gripper_link_wxyz_xyz_0)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        # Compute gripper pose for robot1 in robot0 frame
        gripper_link_wxyz_xyz_1 = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx_1],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx_1],
            ]
        )

        # First compute in robot1's base frame
        gripper_robot_base_1_local = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_1).inverse()
            @ vtf.SE3(wxyz_xyz=gripper_link_wxyz_xyz_1)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        # Transform from robot1's base frame to world frame
        gripper_world_1 = vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_1) @ gripper_robot_base_1_local

        # Transform from world frame to robot0's base frame
        gripper_robot_base_1 = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz_0).inverse() @ gripper_world_1
        )

        robosuite_obs["robot0_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base_0.translation(),
                gripper_robot_base_0.rotation().wxyz,
                [robosuite_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )

        # Now robot1_cartesian_pos is in robot0's base frame
        robosuite_obs["robot1_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base_1.translation(),
                gripper_robot_base_1.rotation().wxyz,
                [robosuite_obs["robot1_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )

        return robosuite_obs

    # ------------------------- Joint State Collection -------------------------

    def enable_joint_state_collection(self, enabled: bool = True, *, clear: bool = True, freq: int = 1) -> None:
        """Enable or disable joint state collection.
        
        Args:
            enabled: If True, enable joint state collection
            clear: If True, clear existing joint state buffer
            freq: Collection frequency in simulation steps (collect every N steps)
        """
        self._collect_joint_states = enabled
        self._joint_state_collect_freq = max(1, int(freq))
        if clear:
            self._joint_state_buffer.clear()
        if enabled:
            # Record initial state
            self._record_joint_state()

    def get_collected_joint_states(self, *, clear: bool = False) -> list[dict[str, np.ndarray]]:
        """Get collected joint states.
        
        Args:
            clear: If True, clear the buffer after retrieving states
            
        Returns:
            List of dictionaries containing joint state data for each collected timestep.
            Each dictionary has keys: 'robot0_joint_pos', 'robot0_joint_vel', 'robot0_gripper_qpos',
            'robot1_joint_pos', 'robot1_joint_vel', 'robot1_gripper_qpos'
        """
        states = [state.copy() for state in self._joint_state_buffer]
        if clear:
            self._joint_state_buffer.clear()
        return states

    def _record_joint_state(self) -> None:
        """Record current joint state to buffer."""
        if not self._collect_joint_states:
            return
        
        robosuite_obs = self.robosuite_env._get_observations()
        state = {
            "robot0_joint_pos": np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64).copy(),
            "robot1_joint_pos": np.array(robosuite_obs["robot1_joint_pos"], dtype=np.float64).copy(),
        }
        
        # Add velocities if available
        if "robot0_joint_vel" in robosuite_obs:
            state["robot0_joint_vel"] = np.array(robosuite_obs["robot0_joint_vel"], dtype=np.float64).copy()
        if "robot1_joint_vel" in robosuite_obs:
            state["robot1_joint_vel"] = np.array(robosuite_obs["robot1_joint_vel"], dtype=np.float64).copy()
        
        # Add gripper positions if available
        if "robot0_gripper_qpos" in robosuite_obs:
            state["robot0_gripper_qpos"] = np.array(robosuite_obs["robot0_gripper_qpos"], dtype=np.float64).copy()
        if "robot1_gripper_qpos" in robosuite_obs:
            state["robot1_gripper_qpos"] = np.array(robosuite_obs["robot1_gripper_qpos"], dtype=np.float64).copy()
        
        self._joint_state_buffer.append(state)

    # ------------------------- Video Capture -------------------------

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True, freq: int | None = None) -> None:
        """Enable or disable video frame capture.

        Args:
            enabled: If True, enable capture.
            clear: If True, clear existing frame buffers.
            freq: Record a frame every N simulation steps (same semantics as joint state freq).
                  If None, keeps current _subsample_rate (default 1 = every step).
                  Use the same value as joint_state_collect_freq to align #image = #proprio.
        """
        self._record_frames = enabled
        if freq is not None:
            self._subsample_rate = max(1, int(freq))
        if clear:
            self._frame_buffer.clear()
            self._camera_frame_buffers.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_camera_frames(self, *, clear: bool = False) -> dict[str, list[np.ndarray]]:
        """Get collected video frames organized by camera name.
        
        Args:
            clear: If True, clear the buffers after retrieving frames
            
        Returns:
            Dictionary mapping camera names to lists of frames (each frame is a numpy array)
        """
        camera_frames = {
            cam_name: [frame.copy() for frame in frames]
            for cam_name, frames in self._camera_frame_buffers.items()
        }
        if clear:
            self._camera_frame_buffers.clear()
        return camera_frames

    def _record_frame(self) -> None:
        if not self._record_frames:
            return

        # Record concatenated frame (for backward compatibility)
        frame = self._render_frame()
        self._frame_buffer.append(frame)
        
        # Record individual camera frames
        self._record_camera_frames()

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        frames = []

        # Agentview
        frames.append(self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )[::-1])

        if self.use_wrist_cameras:
            # Try to find wrist cameras
            for i in range(2):
                for suffix in ["eye_in_hand", "camera_d405"]:
                    cam_name = f"robot{i}_{suffix}"
                    try:
                        self.robosuite_env.sim.model.camera_name2id(cam_name)
                        frames.append(self.robosuite_env.sim.render(
                            camera_name=cam_name,
                            width=self._render_width,
                            height=self._render_height,
                            depth=False,
                        )[::-1])
                        break
                    except Exception:
                        pass

        return np.concatenate(frames, axis=1)

    def _record_camera_frames(self) -> None:
        """Record individual camera frames separately."""
        if not self._record_frames:
            return
        
        # Agentview camera
        agentview_frame = self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )[::-1]
        if "agentview" not in self._camera_frame_buffers:
            self._camera_frame_buffers["agentview"] = []
        self._camera_frame_buffers["agentview"].append(agentview_frame)
        
        if self.use_wrist_cameras:
            # Record wrist cameras
            for i in range(2):
                for suffix in ["eye_in_hand", "camera_d405"]:
                    cam_name = f"robot{i}_{suffix}"
                    try:
                        self.robosuite_env.sim.model.camera_name2id(cam_name)
                        wrist_frame = self.robosuite_env.sim.render(
                            camera_name=cam_name,
                            width=self._render_width,
                            height=self._render_height,
                            depth=False,
                        )[::-1]
                        if cam_name not in self._camera_frame_buffers:
                            self._camera_frame_buffers[cam_name] = []
                        self._camera_frame_buffers[cam_name].append(wrist_frame)
                        break
                    except Exception:
                        pass

    # Temporary viser debugging
    def _update_viser_server(
        self,
    ) -> None:
        obs = self.get_observation()
        if self.viser_server is not None:
            self._viser_init_check()

            # action_joint = action["arm"]["joint_pos"]
            # action_cartesian = action["arm"]["cartesian_pos"][:-1]

            # obs_joint = obs["robot_joint_pos"]
            obs_cartesian = obs["robot_cartesian_pos"][:-1]

            # action_joint_copy = action_joint.copy()
            # action_joint_copy[-1] /= self.gripper_metric_length

            # self.urdf_vis.update_cfg(action_joint_copy)
            # self.urdf_mj_vis.update_cfg(obs_joint)

            # self.pyroki_ee_frame_handle.position = action_cartesian[:3]
            # self.pyroki_ee_frame_handle.wxyz = action_cartesian[3:]

            self.mjcf_ee_frame_handle.position = obs_cartesian[:3]
            self.mjcf_ee_frame_handle.wxyz = obs_cartesian[3:]

            rbg_imgs = obs_get_rgb(obs)
            # if len(rbg_imgs.keys()) > 0:
            for image_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[image_key]

                if "pose" in obs[image_key]:
                    self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                    self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                    self.image_frustum_handle.image = rbg_imgs[image_key]
                else:
                    self.image_frustum_handle.visible = False

            # Temporary hardcode to visualise some stuff for debugging
            if "depth" in obs["agentview"]["images"]:
                points, colors = depth_color_to_pointcloud(
                    obs["agentview"]["images"]["depth"][:, :, 0],
                    rbg_imgs["agentview"],
                    obs["agentview"]["intrinsics"],
                )
                self.viser_server.scene.add_point_cloud(
                    "agentview/point_cloud",
                    points,
                    colors,
                    point_size=0.001,
                    point_shape="square",
                )

    def update_viser_image(self, frame: np.ndarray) -> None:
        if self.viser_server is None:
            return
        self._viser_init_check()
        if self.viser_img_handle is not None:
            self.viser_img_handle.image = frame

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle is None:
            self.mjcf_ee_frame_handle = self.viser_server.scene.add_frame(
                "/panda_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Mujoco render")

        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="agentview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=self._render_width / self._render_height,
                scale=0.05,
            )


__all__ = ["RobosuiteHandoverEnv"]