#!/usr/bin/env python3
"""Interactive waypoint editor for dual-arm Franka tape handover.

Provides a viser-based GUI for:
  1. Dragging a 3D gizmo to set target poses interactively.
  2. Recording waypoints (position + orientation + arm + gripper action).
  3. Auto-generating executable action code from recorded waypoints.
  4. Playing back the trajectory in the MuJoCo simulation.

Usage:
  python test_scripts/waypoint_editor.py [--yellow_offset x,y,z] [--duct_offset x,y,z]

Then open the printed viser URL in a browser.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import viser
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as SciRotation

# ---------------------------------------------------------------------------
# Path / import bootstrap (mirrors existing test scripts)
# ---------------------------------------------------------------------------
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"

os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = (
    "/usr/local/cuda-12.9/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.9"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.base_api import register_api
from api.franka_priviledged_api import FrankaControlTapeHandoverPrivilegedApi
from robosuite.environments.custom.control.base_executor import (
    CodeExecutionEnvBase,
    CodeExecEnvConfig,
)
from robosuite.environments.custom.franka_robosuite_tape_handover import (
    FrankaRobosuiteTapeHandover,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════


class GripperAction(Enum):
    NONE = "none"
    OPEN = "open"
    CLOSE = "close"


class ArmID(Enum):
    ARM0 = 0
    ARM1 = 1


@dataclass
class Waypoint:
    """A single recorded waypoint."""

    arm: ArmID
    position: np.ndarray          # (3,) in robot0 base frame
    orientation_wxyz: np.ndarray  # (4,) wxyz quaternion
    gripper_action: GripperAction = GripperAction.NONE
    z_approach: float = 0.0

    def label(self, index: int) -> str:
        grip = "" if self.gripper_action == GripperAction.NONE else f" | grip={self.gripper_action.value}"
        return f"WP{index} arm{self.arm.value}{grip}"


# ═══════════════════════════════════════════════════════════════════════════
# Waypoint Editor
# ═══════════════════════════════════════════════════════════════════════════


class WaypointEditor:
    """Manages interactive waypoint editing, code generation, and playback.

    All MuJoCo-touching work is dispatched to the main thread via a command
    queue (``cmd_queue``) so that viser callbacks never call into the
    simulator directly — avoiding segfaults from MuJoCo's thread-unsafety.
    """

    def __init__(
        self,
        viser_server: viser.ViserServer,
        low_level_env: FrankaRobosuiteTapeHandover,
        exec_env: CodeExecutionEnvBase,
    ) -> None:
        self.server = viser_server
        self.low_level_env = low_level_env
        self.exec_env = exec_env

        self.waypoints: list[Waypoint] = []
        self._waypoint_handles: list[Any] = []
        self._generated_code: str = ""
        self._playing = False

        # Thread-safe queue: viser callbacks post work here,
        # the main thread drains it.
        self.cmd_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        # Live-follow state (position tracking for change detection)
        self._last_follow_pos: np.ndarray | None = None
        self._last_follow_wxyz: np.ndarray | None = None

        # Grab the API instance for IK solving during live follow
        self._api: FrankaControlTapeHandoverPrivilegedApi = list(
            exec_env._apis.values()
        )[0]

        self._build_gui()

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_gui(self) -> None:
        srv = self.server

        # --- Pose gizmo (initialised at arm0 gripper) ---
        arm0_pos, arm0_quat = self._get_gripper_pose_world(ArmID.ARM0)

        self._gizmo = srv.scene.add_transform_controls(
            "/waypoint_gizmo",
            scale=0.14,
            position=tuple(arm0_pos),
            wxyz=tuple(arm0_quat),
        )

        # --- Sidebar ---
        with srv.gui.add_folder("Waypoint Editor", expand_by_default=True):
            self._arm_select = srv.gui.add_dropdown(
                "Arm", options=["arm0", "arm1"], initial_value="arm0"
            )
            self._live_follow_cb = srv.gui.add_checkbox(
                "Live follow (arm tracks gizmo)", initial_value=False,
            )
            self._follow_steps_slider = srv.gui.add_slider(
                "Follow sim steps", min=5, max=60, step=5, initial_value=15,
                hint="Sim steps per tick while live-following (more = faster tracking, less responsive UI)",
            )
            self._gripper_select = srv.gui.add_dropdown(
                "Gripper action",
                options=["none", "open", "close"],
                initial_value="none",
            )
            self._z_approach_slider = srv.gui.add_slider(
                "z_approach", min=0.0, max=0.30, step=0.005, initial_value=0.0,
                hint="IK approach height offset along tool Z before final pose",
            )
            self._snap_btn = srv.gui.add_button("Snap gizmo to gripper", color="blue")
            self._record_btn = srv.gui.add_button("Record waypoint", color="green")
            self._undo_btn = srv.gui.add_button("Undo last waypoint")
            self._clear_btn = srv.gui.add_button("Clear all waypoints")

            srv.gui.add_markdown("---")
            self._wp_count_label = srv.gui.add_markdown("**Waypoints: 0**")

        with srv.gui.add_folder("Code Editor & Playback", expand_by_default=True):
            self._gen_btn = srv.gui.add_button("Generate from waypoints", color="blue")
            self._code_input = srv.gui.add_text(
                "Action code",
                initial_value="# Paste or generate code here\n",
                multiline=True,
            )
            self._save_btn = srv.gui.add_button("Save code", color="blue")
            self._play_btn = srv.gui.add_button("Run code", color="green")
            self._play_status = srv.gui.add_markdown("*Ready*")

        with srv.gui.add_folder("Go Home / Reset"):
            self._home_arm0_btn = srv.gui.add_button("Arm 0 -> home")
            self._home_arm1_btn = srv.gui.add_button("Arm 1 -> home")
            self._reset_btn = srv.gui.add_button("Reset environment", color="red")

        # --- Callbacks (all just enqueue; never touch MuJoCo) ---
        self._live_follow_cb.on_update(self._on_live_follow_toggle)
        self._snap_btn.on_click(self._on_snap)
        self._record_btn.on_click(self._on_record)
        self._undo_btn.on_click(self._on_undo)
        self._clear_btn.on_click(self._on_clear)
        self._gen_btn.on_click(self._on_generate)
        self._save_btn.on_click(self._on_save)
        self._play_btn.on_click(self._on_play)
        self._home_arm0_btn.on_click(lambda _: self.cmd_queue.put(("home", ArmID.ARM0)))
        self._home_arm1_btn.on_click(lambda _: self.cmd_queue.put(("home", ArmID.ARM1)))
        self._reset_btn.on_click(lambda _: self.cmd_queue.put(("reset", None)))

    # ------------------------------------------------------------------
    # Helpers: coordinate transforms
    # ------------------------------------------------------------------

    def _world_to_robot0(self, world_pos: np.ndarray) -> np.ndarray:
        base0 = vtf.SE3(wxyz_xyz=self.low_level_env.base_link_wxyz_xyz_0)
        return np.asarray(
            (base0.inverse() @ vtf.SE3.from_translation(world_pos)).translation(),
            dtype=np.float64,
        )

    def _robot0_to_world(self, robot0_pos: np.ndarray) -> np.ndarray:
        base0 = vtf.SE3(wxyz_xyz=self.low_level_env.base_link_wxyz_xyz_0)
        return np.asarray(
            (base0 @ vtf.SE3.from_translation(robot0_pos)).translation(),
            dtype=np.float64,
        )

    def _quat_world_to_robot0(self, world_wxyz: np.ndarray) -> np.ndarray:
        base0 = vtf.SE3(wxyz_xyz=self.low_level_env.base_link_wxyz_xyz_0)
        return np.asarray(
            (base0.inverse().rotation() @ vtf.SO3(wxyz=world_wxyz)).wxyz,
            dtype=np.float64,
        )

    def _quat_robot0_to_world(self, robot0_wxyz: np.ndarray) -> np.ndarray:
        base0 = vtf.SE3(wxyz_xyz=self.low_level_env.base_link_wxyz_xyz_0)
        return np.asarray(
            (base0.rotation() @ vtf.SO3(wxyz=robot0_wxyz)).wxyz,
            dtype=np.float64,
        )

    def _get_gripper_pose_world(self, arm: ArmID) -> tuple[np.ndarray, np.ndarray]:
        """Return gripper (position_world, wxyz_world)."""
        obs = self.low_level_env.get_observation()
        key = f"robot{arm.value}_cartesian_pos"
        pos_r0 = obs[key][:3]
        quat_r0 = obs[key][3:7]
        return self._robot0_to_world(pos_r0), self._quat_robot0_to_world(quat_r0)

    def _selected_arm(self) -> ArmID:
        return ArmID.ARM0 if self._arm_select.value == "arm0" else ArmID.ARM1

    def _selected_gripper_action(self) -> GripperAction:
        return GripperAction(self._gripper_select.value)

    # ------------------------------------------------------------------
    # Callbacks — run on the viser thread, so they must NOT touch MuJoCo.
    # Pure-viser ops (reading gizmo, updating labels) are fine; anything
    # that reaches the simulator goes through cmd_queue.
    # ------------------------------------------------------------------

    def _on_live_follow_toggle(self, event: Any) -> None:
        if event.target.value:
            # Reset tracking so the next main-thread tick snaps the gizmo
            self._last_follow_pos = None
            self._last_follow_wxyz = None
            print(f"[editor] Live follow ON (arm {self._arm_select.value})")
        else:
            print("[editor] Live follow OFF")

    def _on_snap(self, _event: Any) -> None:
        self.cmd_queue.put(("snap", None))

    def _on_record(self, _event: Any) -> None:
        world_pos = np.array(self._gizmo.position)
        world_wxyz = np.array(self._gizmo.wxyz)
        r0_pos = self._world_to_robot0(world_pos)
        r0_wxyz = self._quat_world_to_robot0(world_wxyz)

        wp = Waypoint(
            arm=self._selected_arm(),
            position=r0_pos.copy(),
            orientation_wxyz=r0_wxyz.copy(),
            gripper_action=self._selected_gripper_action(),
            z_approach=float(self._z_approach_slider.value),
        )
        self.waypoints.append(wp)
        self._draw_waypoint(len(self.waypoints) - 1, wp)
        self._update_wp_label()
        print(f"[editor] Recorded {wp.label(len(self.waypoints) - 1)}: pos={r0_pos.round(4)}")

    def _on_undo(self, _event: Any) -> None:
        if not self.waypoints:
            return
        self.waypoints.pop()
        if self._waypoint_handles:
            handle = self._waypoint_handles.pop()
            handle.remove()
        self._update_wp_label()

    def _on_clear(self, _event: Any) -> None:
        self.waypoints.clear()
        for h in self._waypoint_handles:
            h.remove()
        self._waypoint_handles.clear()
        self._update_wp_label()

    def _on_generate(self, _event: Any) -> None:
        code = self.generate_code()
        self._generated_code = code
        self._code_input.value = code
        print(f"[editor] Generated action code ({len(self.waypoints)} waypoints)")

    def _on_save(self, _event: Any) -> None:
        self._generated_code = self._code_input.value
        self._play_status.content = "**Code saved.**"
        print(f"[editor] Code saved ({len(self._generated_code)} chars)")

    def _on_play(self, _event: Any) -> None:
        # Always read the latest text from the editor
        code = self._code_input.value.strip()
        if not code:
            self._play_status.content = "**No code to run -- paste or generate code first.**"
            return
        if self._playing:
            self._play_status.content = "*Playback already running...*"
            return
        # Save before running so _generated_code stays in sync
        self._generated_code = code
        self.cmd_queue.put(("play", code))

    # ------------------------------------------------------------------
    # Main-thread command handlers (safe to touch MuJoCo here)
    # ------------------------------------------------------------------

    def process_commands(self) -> None:
        """Drain the command queue. Call this from the main thread."""
        while not self.cmd_queue.empty():
            try:
                cmd, payload = self.cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "snap":
                self._handle_snap()
            elif cmd == "play":
                self._handle_play(payload)
            elif cmd == "home":
                self._handle_home(payload)
            elif cmd == "reset":
                self._handle_reset()

    def _handle_snap(self) -> None:
        arm = self._selected_arm()
        pos_w, quat_w = self._get_gripper_pose_world(arm)
        self._gizmo.position = tuple(pos_w)
        self._gizmo.wxyz = tuple(quat_w)

    def _handle_play(self, code: str) -> None:
        self._playing = True
        self._play_status.content = "*Playing trajectory...*"
        self._play_btn.disabled = True
        try:
            obs, reward, terminated, truncated, info = self.exec_env.step(code)
            rc = info.get("sandbox_rc", -1)
            if rc == 0:
                self._play_status.content = "**Playback complete.**"
            else:
                stderr = info.get("stderr", "")
                self._play_status.content = (
                    f"**Execution error (rc={rc})**\n```\n{stderr[:500]}\n```"
                )
            print(f"[editor] Playback finished: rc={rc}")
        except Exception as exc:
            self._play_status.content = f"**Error:** {exc}"
            print(f"[editor] Playback exception: {exc}")
        finally:
            self._play_btn.disabled = False
            self._playing = False

    def _handle_home(self, arm: ArmID) -> None:
        code = f"goto_home_joint_position_arm{arm.value}()"
        try:
            self.exec_env.step(code)
            print(f"[editor] Arm {arm.value} sent home.")
        except Exception as exc:
            print(f"[editor] Go-home error: {exc}")

    def _handle_reset(self) -> None:
        """Reset the simulation, IK caches, and re-snap the gizmo."""
        try:
            self.exec_env.reset()
            self._api.cfg = None
            self._api.cfg_1 = None
            self._last_follow_pos = None
            self._last_follow_wxyz = None

            # Force viser scene refresh
            self.low_level_env._viser_step = 15
            self.low_level_env._update_viser_server()

            # Re-snap gizmo to arm0 gripper
            pos_w, quat_w = self._get_gripper_pose_world(ArmID.ARM0)
            self._gizmo.position = tuple(pos_w)
            self._gizmo.wxyz = tuple(quat_w)

            self._play_status.content = "**Environment reset.**"
            print("[editor] Environment reset.")
        except Exception as exc:
            self._play_status.content = f"**Reset error:** {exc}"
            print(f"[editor] Reset error: {exc}")

    # ------------------------------------------------------------------
    # Live follow — arm tracks gizmo in real time
    # ------------------------------------------------------------------

    _GIZMO_MOVE_THRESHOLD = 1e-4  # metres / quat-norm

    def live_follow_tick(self) -> None:
        """Called every main-loop iteration. If live-follow is on and the
        gizmo has moved, solve IK and step the sim a few times toward the
        target. Skipped while a playback is in progress.

        Reads the checkbox value directly (no flag) so there is no
        cross-thread race with the viser callback.
        """
        if not self._live_follow_cb.value or self._playing:
            return

        # First tick after enabling: snap the gizmo to the current gripper
        # on the main thread (safe to call get_observation here).
        if self._last_follow_pos is None:
            arm = self._selected_arm()
            pos_w, quat_w = self._get_gripper_pose_world(arm)
            self._gizmo.position = tuple(pos_w)
            self._gizmo.wxyz = tuple(quat_w)
            self._last_follow_pos = pos_w.copy()
            self._last_follow_wxyz = quat_w.copy()
            print(f"[editor] Live follow: gizmo snapped to arm{arm.value}")
            return

        gizmo_pos = np.array(self._gizmo.position, dtype=np.float64)
        gizmo_wxyz = np.array(self._gizmo.wxyz, dtype=np.float64)

        pos_delta = np.linalg.norm(gizmo_pos - self._last_follow_pos)
        quat_delta = np.linalg.norm(gizmo_wxyz - self._last_follow_wxyz)
        if pos_delta < self._GIZMO_MOVE_THRESHOLD and quat_delta < self._GIZMO_MOVE_THRESHOLD:
            return

        self._last_follow_pos = gizmo_pos.copy()
        self._last_follow_wxyz = gizmo_wxyz.copy()

        arm = self._selected_arm()
        r0_pos = self._world_to_robot0(gizmo_pos)
        r0_wxyz = self._quat_world_to_robot0(gizmo_wxyz)

        try:
            target_joints = self._solve_ik(arm, r0_pos, r0_wxyz)
        except Exception as exc:
            print(f"[editor] Live-follow IK failed: {exc}")
            return

        n_steps = int(self._follow_steps_slider.value)
        self._step_sim_toward(target_joints, arm, n_steps)

    def _solve_ik(
        self, arm: ArmID, pos_r0: np.ndarray, quat_wxyz_r0: np.ndarray
    ) -> np.ndarray:
        """Solve IK for the given arm and return target joint angles (7,).

        Mirrors the TCP-offset and frame-transform logic in the API's
        ``goto_pose_arm{0,1}`` but without calling ``move_to_joints_blocking``.
        """
        api = self._api
        tcp_offset = api._TCP_OFFSET

        if arm == ArmID.ARM0:
            quat_xyzw = np.array([quat_wxyz_r0[1], quat_wxyz_r0[2],
                                  quat_wxyz_r0[3], quat_wxyz_r0[0]])
            rot = SciRotation.from_quat(quat_xyzw)
            ik_pos = pos_r0 + rot.apply(tcp_offset)
            ik_wxyz = quat_wxyz_r0

            api.cfg = self._run_ik(ik_pos, ik_wxyz, api.cfg, "robot0_joint_pos")
            return np.asarray(api.cfg[:-1], dtype=np.float64).reshape(7)

        # Arm 1: convert robot0-frame pose -> robot1-frame, then IK
        pose_r0 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=quat_wxyz_r0), translation=pos_r0,
        )
        base0 = vtf.SE3(wxyz_xyz=self.low_level_env.base_link_wxyz_xyz_0)
        base1 = vtf.SE3(wxyz_xyz=self.low_level_env.base_link_wxyz_xyz_1)
        pose_r1 = base1.inverse() @ base0 @ pose_r0

        pos1 = np.asarray(pose_r1.translation(), dtype=np.float64)
        wxyz1 = np.asarray(pose_r1.rotation().wxyz, dtype=np.float64)
        quat_xyzw1 = np.array([wxyz1[1], wxyz1[2], wxyz1[3], wxyz1[0]])
        rot1 = SciRotation.from_quat(quat_xyzw1)
        ik_pos = pos1 + rot1.apply(tcp_offset)

        api.cfg_1 = self._run_ik(ik_pos, wxyz1, api.cfg_1, "robot1_joint_pos")
        return np.asarray(api.cfg_1[:-1], dtype=np.float64).reshape(7)

    def _run_ik(
        self,
        ik_pos: np.ndarray,
        ik_wxyz: np.ndarray,
        prev_cfg: Any,
        joint_obs_key: str,
    ) -> Any:
        """Solve IK with fallback: prev_cfg -> current joints -> no seed."""
        api = self._api
        pks, robot, link = api._pks, api._robot, api._target_link_name

        if prev_cfg is not None:
            return pks.solve_ik_vel_cost(
                robot=robot, target_link_name=link,
                target_position=ik_pos, target_wxyz=ik_wxyz, prev_cfg=prev_cfg,
            )

        obs = self.low_level_env.get_observation()
        init = np.asarray(obs[joint_obs_key][:7], dtype=np.float64)
        try:
            return pks.solve_ik(
                robot=robot, target_link_name=link,
                target_position=ik_pos, target_wxyz=ik_wxyz, initial_cfg=init,
            )
        except Exception:
            return pks.solve_ik(
                robot=robot, target_link_name=link,
                target_position=ik_pos, target_wxyz=ik_wxyz, initial_cfg=None,
            )

    def _step_sim_toward(
        self, target_joints: np.ndarray, arm: ArmID, n_steps: int
    ) -> None:
        """Run *n_steps* simulation steps driving *arm* toward *target_joints*
        while holding the other arm in place. Non-blocking — returns after
        exactly *n_steps* regardless of convergence."""
        env = self.low_level_env
        for _ in range(n_steps):
            obs = env.robosuite_env._get_observations()
            r0_joints = np.array(obs["robot0_joint_pos"], dtype=np.float64)
            r1_joints = np.array(obs["robot1_joint_pos"], dtype=np.float64)

            if arm == ArmID.ARM0:
                r0_action = np.concatenate([target_joints,
                                            [1.0 - env._gripper_fraction_0 * 2.0]])
                r1_action = np.concatenate([r1_joints,
                                            [1.0 - env._gripper_fraction_1 * 2.0]])
            else:
                r0_action = np.concatenate([r0_joints,
                                            [1.0 - env._gripper_fraction_0 * 2.0]])
                r1_action = np.concatenate([target_joints,
                                            [1.0 - env._gripper_fraction_1 * 2.0]])

            env.robosuite_env.step(np.concatenate([r0_action, r1_action]))
            env._sim_step_count += 1

            if hasattr(env, "viser_server"):
                env._update_viser_server()

    # ------------------------------------------------------------------
    # Waypoint visualisation
    # ------------------------------------------------------------------

    def _draw_waypoint(self, idx: int, wp: Waypoint) -> None:
        world_pos = self._robot0_to_world(wp.position)
        world_wxyz = self._quat_robot0_to_world(wp.orientation_wxyz)

        frame = self.server.scene.add_frame(
            f"/waypoints/wp_{idx}",
            position=tuple(world_pos),
            wxyz=tuple(world_wxyz),
            axes_length=0.06,
            axes_radius=0.003,
        )
        self.server.scene.add_label(
            f"/waypoints/wp_{idx}/label",
            text=wp.label(idx),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.04),
        )
        self._waypoint_handles.append(frame)

    def _redraw_all_waypoints(self) -> None:
        for h in self._waypoint_handles:
            h.remove()
        self._waypoint_handles.clear()
        for i, wp in enumerate(self.waypoints):
            self._draw_waypoint(i, wp)

    def _update_wp_label(self) -> None:
        self._wp_count_label.content = f"**Waypoints: {len(self.waypoints)}**"

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def generate_code(self) -> str:
        """Produce an executable action code string from the recorded waypoints."""
        lines: list[str] = [
            "import numpy as np",
            "import viser.transforms as vtf",
            "",
        ]

        for i, wp in enumerate(self.waypoints):
            arm = wp.arm.value
            pos_str = np.array2string(wp.position, separator=", ", precision=6)
            quat_str = np.array2string(wp.orientation_wxyz, separator=", ", precision=6)

            if wp.gripper_action == GripperAction.OPEN:
                lines.append(f"open_gripper_arm{arm}()")
            elif wp.gripper_action == GripperAction.CLOSE:
                lines.append(f"close_gripper_arm{arm}()")

            z_arg = f", z_approach={wp.z_approach}" if wp.z_approach > 0 else ""
            lines.append(
                f"goto_pose_arm{arm}("
                f"np.array({pos_str}), "
                f"np.array({quat_str})"
                f"{z_arg})"
            )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# CLI helpers
# ═══════════════════════════════════════════════════════════════════════════


def parse_offset_list(offset_str: str) -> np.ndarray:
    try:
        values = [float(x.strip()) for x in offset_str.split(",")]
        if len(values) != 3:
            raise ValueError("Offset must contain exactly 3 values (x, y, z)")
        return np.array(values, dtype=float)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid offset format '{offset_str}': {e}. "
            "Expected format: 'x,y,z' (e.g., '0.0,-0.7,0.0')"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive viser waypoint editor for dual-arm Franka tape handover.",
    )
    parser.add_argument(
        "--yellow_offset",
        type=parse_offset_list,
        default="-0.28,0.065,0.0",
    )
    parser.add_argument(
        "--duct_offset",
        type=parse_offset_list,
        default="-0.1,-0.065,0.0",
    )
    args = parser.parse_args()

    yellow_offset = np.array(args.yellow_offset, dtype=float)
    duct_offset = np.array(args.duct_offset, dtype=float)
    print(f"Yellow tape offset: {yellow_offset}")
    print(f"Duct tape offset:   {duct_offset}")

    register_api(
        "franka-handover-privileged",
        lambda env: FrankaControlTapeHandoverPrivilegedApi(env),
    )

    controller_cfg_path = os.path.join(
        ROOT_DIR,
        "robosuite", "environments", "custom", "configs", "panda_joint_ctrl_slow.json",
    )

    print("Initializing low-level environment...")
    low_level_env = FrankaRobosuiteTapeHandover(
        controller_cfg=controller_cfg_path,
        viser_debug=True,
        privileged=True,
        enable_render=False,
        use_wrist_cameras=False,
        yellow_tape_offset=yellow_offset,
        duct_tape_offset=duct_offset,
    )

    cfg = CodeExecEnvConfig(
        low_level=low_level_env,
        apis=["franka-handover-privileged"],
        prompt="Interactive waypoint editing session.",
    )

    print("Initializing code execution environment...")
    exec_env = CodeExecutionEnvBase(cfg)
    print("Resetting environment...")
    exec_env.reset()

    # Force the viser scene to initialise (URDFs, table, objects, camera
    # image) right now, instead of waiting for the first sim step.
    # _viser_init_check is gated by _viser_scene_init; _update_viser_server
    # is gated by _viser_step%16. We bypass both by calling them directly.
    print("Populating viser scene...")
    low_level_env._viser_step = 15  # next increment hits %16==0
    low_level_env._update_viser_server()

    srv = low_level_env.viser_server

    print("Building waypoint editor UI...")
    editor = WaypointEditor(srv, low_level_env, exec_env)

    print(f"\n{'='*60}")
    print(f"  Waypoint Editor ready -- open viser in your browser")
    print(f"  (default: http://localhost:8080)")
    print(f"{'='*60}\n")

    # Main-thread event loop: drain the command queue and run live-follow
    # so that all MuJoCo work happens on this thread (not thread-safe).
    try:
        while True:
            editor.process_commands()
            editor.live_follow_tick()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
