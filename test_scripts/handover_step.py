import os
import sys
import argparse
from datetime import datetime

import imageio
import numpy as np
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up"
macros.IMAGE_CONVENTION = "opencv"

# Configure CUDA paths for XLA / GPU execution.
os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.9/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.9"

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# Now we can import the classes
from robosuite.environments.custom.franka_robosuite_tape_handover import FrankaRobosuiteTapeHandover
from robosuite.environments.custom.control.base_executor import CodeExecutionEnvBase, CodeExecEnvConfig
from api.franka_priviledged_api import FrankaControlTapeHandoverPrivilegedApi
from api.base_api import register_api

def parse_offset_list(offset_str):
    """
    Parse a comma-separated string of three floats into a numpy array.
    """
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


def build_handover_action_code() -> str:
    """Construct the handover action code.

    Handover geometry (x_shift, y_shift, angle_shift) is read dynamically via
    get_handover_params() right before the handover sequence so that viser slider
    adjustments made while paused in step-mode take effect immediately.
    """
    return """import numpy as np
import viser.transforms as vtf

def _handover_geometry(center):
    p = get_handover_params()
    ang = p["angle_shift"]
    Rz_q = np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)])
    Rz = vtf.SO3(wxyz=Rz_q).as_matrix()
    h_pos = center + Rz @ np.array([-0.15 - p["x_shift"], 0.1 + p["y_shift"], 0.0])
    a0_pos = h_pos + Rz @ np.array([0.035, -0.1025, 0.0])
    gs_quat = (vtf.SO3(wxyz=Rz_q) @ vtf.SO3(wxyz=[0.707, 0, -0.707, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])).wxyz
    gr_quat = (vtf.SO3(wxyz=Rz_q) @ vtf.SO3(wxyz=[0.707, 0, 0.707, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])).wxyz
    return h_pos, a0_pos, gs_quat, gr_quat

yellow_tape_pos, _ = get_object_pose("yellow tape")
duct_tape_pos, _ = get_object_pose("duct tape")

arm1_pos, _ = get_arm1_gripper_pose()
arm0_pos, _ = get_arm0_gripper_pose()
center = (arm1_pos + arm0_pos) / 2

gripper_down_quat = np.array([0, 1, 0, 0])

# Initial orientations (for early gotos that need gripper_rotated_side_quat)
_, _, _, gripper_rotated_side_quat = _handover_geometry(center)

# Arm1: pick up yellow tape
open_gripper_arm1()
goto_pose_arm1(yellow_tape_pos + np.array([-0.01, 0.05, -0.02]), gripper_down_quat, z_approach=0.15)
close_gripper_arm1()
lifted = yellow_tape_pos.copy()
lifted[2] = 0.15
goto_pose_arm1(lifted, gripper_down_quat)

above_pickup = lifted.copy()
above_pickup[2] = get_arm_base_midpoint_z()
goto_pose_arm1(above_pickup, gripper_rotated_side_quat)

# Re-read slider values here — any adjustments made while paused at above gotos take effect now
handover_pos, arm0_handover_pos, gripper_side_quat, gripper_rotated_side_quat = _handover_geometry(center)

# Lateral waypoint: sampled from a sphere, constrained to left/right of the approach path
# Sphere centers are offset so the two spheres just barely touch (center-to-center = 2r)
_p = get_handover_params()
_perturb_r = _p.get("perturb_radius", 0.0)
if _perturb_r > 0:
    _sep = handover_pos - arm0_handover_pos
    _sep_d = np.linalg.norm(_sep)
    _sep_hat = _sep / _sep_d if _sep_d > 1e-6 else np.array([1.0, 0.0, 0.0])
    _mid = (handover_pos + arm0_handover_pos) / 2
    _arm1_sph = _mid + _perturb_r * _sep_hat
    _arm0_sph = _mid - _perturb_r * _sep_hat

if _perturb_r > 0:
    _arm1_now, _ = get_arm1_gripper_pose()
    _approach = handover_pos - _arm1_now
    _approach_hat = _approach / np.linalg.norm(_approach)
    _up = np.array([0.0, 0.0, 1.0])
    _lateral = np.cross(_approach_hat, _up)
    _ln = np.linalg.norm(_lateral)
    _lateral = _lateral / _ln if _ln > 1e-6 else np.array([1.0, 0.0, 0.0])
    _vert = np.cross(_lateral, _approach_hat)
    _side = np.random.choice([-1.0, 1.0])
    _theta = np.random.uniform(-np.pi / 6, np.pi / 6)
    _dir = _side * np.cos(_theta) * _lateral + np.sin(_theta) * _vert
    _dir = _dir / np.linalg.norm(_dir)
    _waypoint = _arm1_sph + _perturb_r * _dir
    goto_pose_arm1(_waypoint, gripper_rotated_side_quat)

# Arm1: move to handover
goto_pose_arm1(handover_pos, gripper_rotated_side_quat)

# Arm0: approach with lateral waypoint
arm0_quat = gripper_side_quat
open_gripper_arm0()
if _perturb_r > 0:
    _arm0_now, _ = get_arm0_gripper_pose()
    _approach0 = arm0_handover_pos - _arm0_now
    _approach0_hat = _approach0 / np.linalg.norm(_approach0)
    _up0 = np.array([0.0, 0.0, 1.0])
    _lateral0 = np.cross(_approach0_hat, _up0)
    _ln0 = np.linalg.norm(_lateral0)
    _lateral0 = _lateral0 / _ln0 if _ln0 > 1e-6 else np.array([1.0, 0.0, 0.0])
    _vert0 = np.cross(_lateral0, _approach0_hat)
    _side0 = np.random.choice([-1.0, 1.0])
    _theta0 = np.random.uniform(-np.pi / 6, np.pi / 6)
    _dir0 = _side0 * np.cos(_theta0) * _lateral0 + np.sin(_theta0) * _vert0
    _dir0 = _dir0 / np.linalg.norm(_dir0)
    _waypoint0 = _arm0_sph + _perturb_r * _dir0
    # goto_pose_arm0(_waypoint0, arm0_quat)
goto_pose_arm0(arm0_handover_pos + np.array([-.15, -0.15, 0.0]), arm0_quat)
goto_pose_arm0(arm0_handover_pos, arm0_quat, z_approach=0.10)
close_gripper_arm0()

# Arm1: release and retract
open_gripper_arm1()
shifted_handover = handover_pos + vtf.SO3(wxyz=gripper_rotated_side_quat).as_matrix() @ np.array([0, 0, -0.1])
shifted_arm0 = arm0_handover_pos + vtf.SO3(wxyz=arm0_quat).as_matrix() @ np.array([0, 0, -0.1])
goto_pose_arm0(shifted_arm0, arm0_quat)
goto_pose_arm1(shifted_handover, gripper_rotated_side_quat)
goto_home_joint_position_arm1()

# Arm0: place at duct tape
goto_pose_arm0(duct_tape_pos + np.array([0.0, -0.03, 0.05]), gripper_down_quat, z_approach=0.15)
open_gripper_arm0()
goto_pose_arm0(duct_tape_pos + np.array([0, -0.05, 0.2]), gripper_down_quat)
goto_home_joint_position_arm0()
""".rstrip()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run a single Franka tape handover rollout and save video/joint-state data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_handover_step.py --yellow_offset 0.0,-0.7,0.0 --duct_offset 0.0,0.7,0.0
  python test_handover_step.py --yellow_offset 0.1,-0.6,0.0 --duct_offset -0.1,0.6,0.0
        """
    )
    parser.add_argument(
        '--yellow_offset',
        type=parse_offset_list,
        default='-0.28,0.065,0.0',
        help='Yellow tape offset as comma-separated x,y,z values (default: -0.28,0.065,0.0)'
    )
    parser.add_argument(
        '--duct_offset',
        type=parse_offset_list,
        default='-0.1,-0.065,0.0',
        help='Duct tape offset as comma-separated x,y,z values (default: -0.1,-0.065,0.0)'
    )
    parser.add_argument(
        '--joint_state_fps',
        type=float,
        default=30.0,
        help='Sampling rate for joint state and video capture in fps; use same for both so #image = #proprio (default: 30.0). Increase to 60 or 120 for more frames.'
    )
    parser.add_argument(
        '--x_shift',
        type=float,
        default=0.0,
        help='X shift for the handover position (default: 0.0)'
    )
    parser.add_argument(
        '--y_shift',
        type=float,
        default=0.0,
        help='Y shift for the handover position (default: 0.0)'
    )
    parser.add_argument(
        '--angle_shift',
        type=float,
        default=0.0,
        help='Angle (z-axis) shift for the handover position in radians (default: 0.0)'
    )
    parser.add_argument(
        '--perturb_radius',
        type=float,
        default=0.05,
        help='Radius (m) of sphere for sampling a lateral waypoint before handover; 0 to disable (default: 0.05)'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=None,
        help='Optional output directory. If set, files are written there directly (e.g. dataset/<wandb_run_name>).'
    )
    args = parser.parse_args()
    
    # Extract offsets as numpy arrays
    yellow_offset = np.array(args.yellow_offset, dtype=float)
    duct_offset = np.array(args.duct_offset, dtype=float)

    x_shift = float(args.x_shift)
    y_shift = float(args.y_shift)
    angle_shift = float(args.angle_shift)
    perturb_radius = float(args.perturb_radius)

    print(f"Yellow tape offset: {yellow_offset}")
    print(f"Duct tape offset: {duct_offset}")
    print(f"Handover position shifts: x_shift={x_shift}, y_shift={y_shift}, angle_shift={angle_shift}")
    print(f"Lateral waypoint sphere radius: {perturb_radius} m" + (" (disabled)" if perturb_radius <= 0 else ""))
    
    # Register the API so CodeExecutionEnvBase can find it
    # The name here is used by CodeExecutionEnvBase to look up the API
    register_api("franka-handover-privileged", lambda env: FrankaControlTapeHandoverPrivilegedApi(env))

    # 1. Instantiate the low-level environment
    controller_cfg_path = os.path.join(root_dir, "robosuite", "environments", "custom", "configs", "panda_joint_ctrl_slow.json")
    print("Initializing low-level FrankaRobosuiteTapeHandover environment...")
    low_level_env = FrankaRobosuiteTapeHandover(
        controller_cfg=controller_cfg_path,
        viser_debug=False,
        privileged=True,
        enable_render=False,
        use_wrist_cameras=True,  # Enable wrist cameras for data collection
        yellow_tape_offset=yellow_offset,
        duct_tape_offset=duct_offset,
    )

    # 2. Define the configuration for the high-level code execution environment
    # We specify the low-level env and the API we just registered
    cfg = CodeExecEnvConfig(
        low_level=low_level_env,
        apis=["franka-handover-privileged"],
        prompt="Pick up the yellow tape with Arm 1 and hand it over to Arm 0.",
    )

    # 3. Instantiate the high-level environment
    # This environment's step() method takes a string of Python code
    print("Initializing high-level CodeExecutionEnvBase...")
    exec_env = CodeExecutionEnvBase(cfg)

    # 5. Reset the environment
    print("Resetting environment...")
    obs, info = exec_env.reset()

    # 4. Enable video recording and joint state collection
    # Sample proprio (and video) every simulation frame so #image = #proprio at sim rate (~500 Hz).
    # To downsample instead: joint_state_freq_steps = max(1, int(SIMULATION_FPS / args.joint_state_fps))
    SIMULATION_FPS = 500.0  # From robosuite macros.SIMULATION_TIMESTEP = 0.002s
    joint_state_freq_steps = max(1, int(SIMULATION_FPS / args.joint_state_fps))
    effective_fps = SIMULATION_FPS / joint_state_freq_steps
    print(f"Enabling video and joint state every {joint_state_freq_steps} sim steps (~{effective_fps:.1f} Hz)...")
    exec_env.enable_video_capture(True, freq=joint_state_freq_steps)
    low_level_env.enable_joint_state_collection(True, clear=True, freq=joint_state_freq_steps)

    low_level_env.handover_params.update({
        "x_shift": x_shift,
        "y_shift": y_shift,
        "angle_shift": angle_shift,
        "perturb_radius": perturb_radius,
    })

    action_code = build_handover_action_code()

    # 7. Run the step with the scripted action
    print("\nExecuting hardcoded action via exec_env.step()...")
    # This call triggers exec(action_code, ...) inside the executor
    obs, reward, terminated, truncated, info = exec_env.step(action_code)
    
    # 9. Create output directory
    if args.dataset_dir is not None and str(args.dataset_dir).strip() != "":
        dataset_dir = args.dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Using dataset_dir override: {dataset_dir}")
    else:
        # Directory name encodes the yellow and duct tape offsets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_params = low_level_env.handover_params
        yo = yellow_offset
        do = duct_offset
        dir_name = (
            f"handover_yellow_{yo[0]}_{yo[1]}_{yo[2]}"
            f"_duct_{do[0]}_{do[1]}_{do[2]}"
            f"_x_{final_params['x_shift']}_y_{final_params['y_shift']}"
            f"_angle_{final_params['angle_shift']}_{timestamp}"
        ).replace(".", "_").replace("-", "neg")
        dataset_dir = os.path.join("dataset", dir_name)
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Created directory: {dataset_dir}")
    
    # Base filename for files (without the full path)
    base_filename = "handover"
    
    # 10. Save the recorded videos (separate videos for each camera)
    all_video_frames = low_level_env.get_camera_frames()
    
    if all_video_frames:
        # Save agentview camera video
        if "agentview" in all_video_frames and all_video_frames["agentview"]:
            agentview_frames = all_video_frames["agentview"]
            agentview_path = os.path.join(dataset_dir, f"{base_filename}_agentview.mp4")
            print(f"Saving agentview video with {len(agentview_frames)} frames to {agentview_path}...")
            imageio.mimsave(agentview_path, agentview_frames, fps=20)
            print(f"Agentview video saved to {agentview_path}")
        
        # Save robot0 wrist camera video
        if "robot0_eye_in_hand" in all_video_frames and all_video_frames["robot0_eye_in_hand"]:
            robot0_frames = all_video_frames["robot0_eye_in_hand"]
            robot0_path = os.path.join(dataset_dir, f"{base_filename}_robot0_wrist.mp4")
            print(f"Saving robot0 wrist camera video with {len(robot0_frames)} frames to {robot0_path}...")
            imageio.mimsave(robot0_path, robot0_frames, fps=20)
            print(f"Robot0 wrist camera video saved to {robot0_path}")
        
        # Save robot1 wrist camera video
        if "robot1_eye_in_hand" in all_video_frames and all_video_frames["robot1_eye_in_hand"]:
            robot1_frames = all_video_frames["robot1_eye_in_hand"]
            robot1_path = os.path.join(dataset_dir, f"{base_filename}_robot1_wrist.mp4")
            print(f"Saving robot1 wrist camera video with {len(robot1_frames)} frames to {robot1_path}...")
            imageio.mimsave(robot1_path, robot1_frames, fps=20)
            print(f"Robot1 wrist camera video saved to {robot1_path}")
    else:
        print("No video frames were captured.")
    
    # 11. Save joint states to .npz files (separate files for each arm)
    joint_states = low_level_env.get_collected_joint_states(clear=False)
    if joint_states:
        print(f"\nSaving joint states to .npz files...")
        
        # Prepare data for robot0 (arm0)
        robot0_data = {}
        if joint_states and "robot0_joint_pos" in joint_states[0]:
            robot0_data["joint_positions"] = np.stack([state["robot0_joint_pos"] for state in joint_states])
        if joint_states and "robot0_joint_vel" in joint_states[0]:
            robot0_data["joint_velocities"] = np.stack([state["robot0_joint_vel"] for state in joint_states])
        if joint_states and "robot0_gripper_qpos" in joint_states[0]:
            robot0_data["gripper_positions"] = np.stack([state["robot0_gripper_qpos"] for state in joint_states])
        
        # Prepare data for robot1 (arm1)
        robot1_data = {}
        if joint_states and "robot1_joint_pos" in joint_states[0]:
            robot1_data["joint_positions"] = np.stack([state["robot1_joint_pos"] for state in joint_states])
        if joint_states and "robot1_joint_vel" in joint_states[0]:
            robot1_data["joint_velocities"] = np.stack([state["robot1_joint_vel"] for state in joint_states])
        if joint_states and "robot1_gripper_qpos" in joint_states[0]:
            robot1_data["gripper_positions"] = np.stack([state["robot1_gripper_qpos"] for state in joint_states])
        
        # Save robot0 joint states
        if robot0_data:
            robot0_filename = os.path.join(dataset_dir, f"{base_filename}_robot0_joints.npz")
            np.savez_compressed(robot0_filename, **robot0_data)
            print(f"Robot0 (arm0) joint states saved to {robot0_filename} ({len(joint_states)} samples)")
        
        # Save robot1 joint states
        if robot1_data:
            robot1_filename = os.path.join(dataset_dir, f"{base_filename}_robot1_joints.npz")
            np.savez_compressed(robot1_filename, **robot1_data)
            print(f"Robot1 (arm1) joint states saved to {robot1_filename} ({len(joint_states)} samples)")
    else:
        print("No joint states were collected.")

    # 9. Print execution results and logs
    print("\n" + "="*40)
    print("STEP EXECUTION RESULTS")
    print("="*40)
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    task_completed = info.get('task_completed', False)
    print(f"Task Completed: {task_completed}")
    
    # 12. Check for failure or excessive length
    max_frames = 1200 # 60 seconds at 20 fps
    num_frames = 0
    if all_video_frames and "agentview" in all_video_frames:
        num_frames = len(all_video_frames["agentview"])
    
    if num_frames > max_frames:
        print(f"\nFAILURE: Video length ({num_frames} frames) exceeds maximum allowed ({max_frames} frames, ~1 min).")
        sys.exit(2)
    
    if info.get('sandbox_rc') != 0:
        print(f"\nFAILURE: Code execution failed with an exception.")
        sys.exit(1)

    print("\n--- STDOUT FROM CODE EXECUTION ---")
    print(info['stdout'])
    
    if info['stderr']:
        print("\n--- STDERR FROM CODE EXECUTION ---")
        print(info['stderr'])
    
    print("\nDone.")

if __name__ == "__main__":
    main()