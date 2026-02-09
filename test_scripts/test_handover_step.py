import os
import sys
import argparse
import numpy as np
import imageio
from collections import OrderedDict
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up"
macros.IMAGE_CONVENTION = "opencv"

# fix cuda error
import os
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
    Parse a comma-separated string of floats into a numpy array.
    
    Args:
        offset_str: String like "0.2,-0.5,0.0" or "-0.3,0.5,0.0"
    
    Returns:
        numpy array of floats
    """
    try:
        values = [float(x.strip()) for x in offset_str.split(',')]
        if len(values) != 3:
            raise ValueError("Offset must contain exactly 3 values (x, y, z)")
        return np.array(values)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid offset format '{offset_str}': {e}. Expected format: 'x,y,z' (e.g., '0.0,-0.7,0.0')")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test handover step with configurable tape offsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_handover_step.py --yellow_offset 0.0,-0.7,0.0 --duct_offset 0.0,0.7,0.0
  python test_handover_step.py --yellow_offset 0.1,-0.6,0.0 --duct_offset -0.1,0.6,0.0

Data collection randomization:
  - Handover position: use --randomize_handover to vary where the handoff happens (blend between arms, x/y shifts, z).
  - Pickup/placement: vary --yellow_offset and --duct_offset per run (e.g. loop with random samples, or use sweep_handover_offsets.py for a grid).
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
    # Handover position randomization (for more diverse data collection)
    parser.add_argument(
        '--randomize_handover',
        action='store_true',
        help='Randomize handover position (blend between arms, x/y shifts, z jitter) for each rollout.'
    )
    parser.add_argument(
        '--handover_x_shift_range',
        type=str,
        default='-0.20,-0.10',
        help='Comma-separated min,max for handover x shift in m (default: -0.20,-0.10). Used only if --randomize_handover.'
    )
    parser.add_argument(
        '--handover_y_shift_range',
        type=str,
        default='0.05,0.15',
        help='Comma-separated min,max for handover y shift in m (default: 0.05,0.15). Used only if --randomize_handover.'
    )
    parser.add_argument(
        '--handover_alpha_range',
        type=str,
        default='0.35,0.65',
        help='Comma-separated min,max for blend alpha: handover_pos = (1-alpha)*arm1 + alpha*arm0 (default: 0.35,0.65). Used only if --randomize_handover.'
    )
    parser.add_argument(
        '--handover_z_jitter_range',
        type=str,
        default='-0.02,0.03',
        help='Comma-separated min,max for handover z jitter in m (default: -0.02,0.03). Used only if --randomize_handover.'
    )
    parser.add_argument(
        '--handover_seed',
        type=int,
        default=None,
        help='Seed for handover randomization (default: None). Used only if --randomize_handover.'
    )
    parser.add_argument(
        '--handover_angle_range',
        type=str,
        default='-0.35,0.35',
        help='Comma-separated min,max for handover yaw angle in radians (rotation around world Z; default ±0.35 rad ≈ ±20°). Used only if --randomize_handover.'
    )
    args = parser.parse_args()
    
    # Extract offsets as numpy arrays
    yellow_offset_args = args.yellow_offset
    duct_offset_args = args.duct_offset
    yellow_offset = np.array(yellow_offset_args)
    duct_offset = np.array(duct_offset_args)
    
    print(f"Yellow tape offset: {yellow_offset_args}")
    print(f"Duct tape offset: {duct_offset_args}")

    # Handover position randomization (for diverse data collection)
    def parse_range(s, name="range"):
        parts = [x.strip() for x in s.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{name} must be 'min,max', got {s}")
        return float(parts[0]), float(parts[1])

    use_handover_rand = getattr(args, "randomize_handover", False)
    if use_handover_rand:
        rng_handover = np.random.default_rng(args.handover_seed)
        x_lo, x_hi = parse_range(args.handover_x_shift_range, "handover_x_shift_range")
        y_lo, y_hi = parse_range(args.handover_y_shift_range, "handover_y_shift_range")
        a_lo, a_hi = parse_range(args.handover_alpha_range, "handover_alpha_range")
        z_lo, z_hi = parse_range(args.handover_z_jitter_range, "handover_z_jitter_range")
        handover_x_shift = rng_handover.uniform(x_lo, x_hi)
        handover_y_shift = rng_handover.uniform(y_lo, y_hi)
        handover_alpha = rng_handover.uniform(a_lo, a_hi)
        handover_z_jitter = rng_handover.uniform(z_lo, z_hi)
        angle_lo, angle_hi = parse_range(args.handover_angle_range, "handover_angle_range")
        handover_angle_rad = rng_handover.uniform(angle_lo, angle_hi)
        # arm0_handover offset from center (half gripper width 0.1025, small forward 0.035) - add small jitter
        arm0_y_offset = -0.1025 + rng_handover.uniform(-0.01, 0.01)
        arm0_x_offset = 0.035 + rng_handover.uniform(-0.01, 0.01)
        print(f"Handover randomization: x_shift={handover_x_shift:.3f}, y_shift={handover_y_shift:.3f}, alpha={handover_alpha:.3f}, z_jitter={handover_z_jitter:.3f}, angle_rad={handover_angle_rad:.3f}")
    else:
        handover_x_shift = -0.15
        handover_y_shift = 0.1
        handover_alpha = 0.5
        handover_z_jitter = 0.0
        handover_angle_rad = 0.0
        arm0_y_offset = -0.1025
        arm0_x_offset = 0.035
    
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
        yellow_tape_offset=yellow_offset_args,
        duct_tape_offset=duct_offset_args,
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
    joint_state_freq_steps = 1
    print(f"Enabling video and joint state (every simulation frame, ~{int(SIMULATION_FPS)} Hz)...")
    exec_env.enable_video_capture(True, freq=joint_state_freq_steps)
    low_level_env.enable_joint_state_collection(True, clear=True, freq=joint_state_freq_steps)

    # --- Manually set object position (relative offset) ---
    # sim = low_level_env.robosuite_env.sim

    # # 1. Get the table center
    # table_center = sim.data.site_xpos[sim.model.site_name2id("table0_top")]

    # total_offset_yellow_tape = np.array([0.0, 0.0, 0.0])
    # total_offset_duct_tape = np.array([0.0, 0.0, 0.0])

    # # 2. Get current yellow tape position
    # yellow_tape_joint = low_level_env.robosuite_env.yellow_tape.joints[0]
    # yellow_qpos = sim.data.get_joint_qpos(yellow_tape_joint).copy()
    # duct_tape_joint = low_level_env.robosuite_env.duct_tape.joints[0]
    # duct_qpos = sim.data.get_joint_qpos(duct_tape_joint).copy()
    
    # # 3. Calculate offset to center it (keeping original Z height)
    # yellow_offset = table_center - yellow_qpos[:3]
    # yellow_offset[2] = 0 # Optional: don't shift Z if you want it to stay on the surface
    # duct_offset = table_center - duct_qpos[:3]
    # duct_offset[2] = 0 # Optional: don't shift Z if you want it to stay on the surface
    # # print the offset
    # print(f"Yellow offset: {yellow_offset}")
    # print(f"Duct offset: {duct_offset}")
    # # Apply the offset (move the yellow tape to the table center)
    # yellow_qpos[:3] += yellow_offset
    # sim.data.set_joint_qpos(yellow_tape_joint, yellow_qpos)
    # total_offset_yellow_tape += yellow_offset
    # duct_qpos[:3] += duct_offset
    # sim.data.set_joint_qpos(duct_tape_joint, duct_qpos)
    # total_offset_duct_tape += duct_offset

    # # Define offsets [dx, dy, dz]
    # yellow_offset = yellow_offset_args
    # duct_offset = duct_offset_args
    # total_offset_yellow_tape += yellow_offset
    # total_offset_duct_tape += duct_offset

    
    # # Offsets are now passed in via command line arguments
    # # Yellow tape
    # yellow_tape_joint = low_level_env.robosuite_env.yellow_tape.joints[0]
    # yellow_qpos = sim.data.get_joint_qpos(yellow_tape_joint).copy()
    # yellow_qpos[:3] += yellow_offset
    # sim.data.set_joint_qpos(yellow_tape_joint, yellow_qpos)

    # # Duct tape
    # duct_tape_joint = low_level_env.robosuite_env.duct_tape.joints[0]
    # duct_qpos = sim.data.get_joint_qpos(duct_tape_joint).copy()
    # duct_qpos[:3] += duct_offset
    # sim.data.set_joint_qpos(duct_tape_joint, duct_qpos)

    # sim.forward()
    # print(f"Total offset yellow tape: {total_offset_yellow_tape}")
    # print(f"Total offset duct tape: {total_offset_duct_tape}")
    # ------------------------------------
    # Handover position: midpoint with blend alpha, then shifts (fixed or randomized via --randomize_handover)
    action_code = f"""import numpy as np
import viser.transforms as vtf

# --- Get poses ---
yellow_tape_pos, yellow_tape_quat = get_object_pose("yellow tape")
duct_tape_pos, duct_tape_quat = get_object_pose("duct tape")

arm1_pos, _ = get_arm1_gripper_pose()
arm0_pos, _ = get_arm0_gripper_pose()
# handover_pos: blend between arms (alpha=0.5 is midpoint), then apply x/y shifts and optional z jitter
handover_pos = (1.0 - {handover_alpha}) * arm1_pos + {handover_alpha} * arm0_pos
handover_pos[0] += {handover_x_shift}
handover_pos[1] += {handover_y_shift}
handover_pos[2] += {handover_z_jitter}
# arm0 receive position: offset from handover center, rotated with handover yaw so geometry stays consistent
arm0_offset_vec = np.array([{arm0_x_offset}, {arm0_y_offset}, 0.0])

# --- Pickup orientation ---
gripper_down_quat = np.array([0, 1, 0, 0])
# old one: gripper_side_matrix = vtf.SO3(wxyz=[0.707, 0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_side_matrix = vtf.SO3(wxyz=[0.707, 0, -0.707, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_side_quat = gripper_side_matrix.wxyz
# old one: gripper_rotated_side_matrix = vtf.SO3(wxyz=[0.707, -0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_rotated_side_matrix = vtf.SO3(wxyz=[0.707, 0, 0.707, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_rotated_side_quat = gripper_rotated_side_matrix.wxyz
# Optional random yaw around world Z for handover (angle_rad=0 when not randomizing)
handover_angle_rad = {handover_angle_rad}
R_z = vtf.SO3.from_rpy_radians(0.0, 0.0, handover_angle_rad)
# Rotate arm0 offset by same yaw so handover position + orientation stay consistent
arm0_handover_pos = handover_pos + (R_z.as_matrix() @ arm0_offset_vec)
arm1_handover_quat = (R_z @ vtf.SO3(wxyz=gripper_rotated_side_quat)).wxyz
arm0_handover_quat = (R_z @ vtf.SO3(wxyz=gripper_side_quat)).wxyz

# Arm1: pick up yellow tape
open_gripper_arm1()
# shift the pick -y by 2cm to grab the tape on one end of the radius
goto_pose_arm1((yellow_tape_pos+np.array([-0.01, 0.05, -0.02])), gripper_down_quat, z_approach=0.15)
close_gripper_arm1()
lifted = yellow_tape_pos.copy(); lifted[2] = 0.15
goto_pose_arm1(lifted, gripper_down_quat)
# goto_home_joint_position_arm1()

above_pickup_at_handover_height = lifted.copy()
above_pickup_at_handover_height[2] = get_arm_base_midpoint_z()

goto_pose_arm1(above_pickup_at_handover_height, arm1_handover_quat)


# Arm1: move to handover (shifted toward arm0)
goto_pose_arm1(handover_pos, arm1_handover_quat)

# Arm0 approach
open_gripper_arm0()
goto_pose_arm0(arm0_handover_pos, arm0_handover_quat, z_approach=0.10)
close_gripper_arm0()

# Arm1: release and retract
open_gripper_arm1()
shifted_handover_pos = handover_pos + vtf.SO3(wxyz=arm1_handover_quat).as_matrix() @ np.array([0, 0, -0.1])
shifted_arm0_pos = arm0_handover_pos + vtf.SO3(wxyz=arm0_handover_quat).as_matrix() @ np.array([0, 0, -0.1])
goto_pose_arm0(shifted_arm0_pos, arm0_handover_quat)
goto_pose_arm1(shifted_handover_pos, arm1_handover_quat)
goto_home_joint_position_arm1()
# goto_home_joint_position_arm0()

# Arm0: drop cube in bowl, shifted to the left because the tape is slightly off-center in the robot's grasp
goto_pose_arm0((duct_tape_pos+np.array([0.0, -0.03, 0.05])), gripper_down_quat, z_approach=0.15)
open_gripper_arm0()
goto_pose_arm0((duct_tape_pos+np.array([0, -0.05, 0.2])), gripper_down_quat)
goto_home_joint_position_arm0()
    """

#     action_code = f"""import numpy as np
# import viser.transforms as vtf

# # --- Get poses ---
# yellow_tape_pos, yellow_tape_quat = get_object_pose("yellow tape")
# duct_tape_pos, duct_tape_quat = get_object_pose("duct tape")

# arm1_pos, _ = get_arm1_gripper_pose()
# arm0_pos, _ = get_arm0_gripper_pose()
# handover_pos = (arm1_pos + arm0_pos) / 2
# arm0_handover_pos = handover_pos.copy()
# # Need a way to get the width of the yellow tape that isnt privileged
# # this is half the width of the franka gripper:
# arm0_handover_pos[2] += 0.1025

# # --- Pickup orientation ---
# gripper_down_quat = np.array([0, 1, 0, 0])
# gripper_side_matrix = vtf.SO3(wxyz=[0.707, 0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
# gripper_side_quat = gripper_side_matrix.wxyz
# gripper_rotated_side_matrix = vtf.SO3(wxyz=[0.707, -0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
# gripper_rotated_side_quat = gripper_rotated_side_matrix.wxyz

# # # Arm0 pick up duct tape
# # open_gripper_arm0()
# # goto_pose_arm0((duct_tape_pos+np.array([-0.01, -0.05, 0.0])), gripper_down_quat, z_approach=0.15)
# # close_gripper_arm0()
# # lifted = duct_tape_pos.copy(); lifted[2] = 0.15
# # goto_pose_arm0(lifted, gripper_down_quat)
# # goto_home_joint_position_arm0()

# # Arm1: pick up yellow tape
# open_gripper_arm1()
# # shift the pick -y by 2cm to grab the tape on one end of the radius
# goto_pose_arm1((yellow_tape_pos+np.array([-0.01, 0.05, -0.02])), gripper_down_quat, z_approach=0.15)
# close_gripper_arm1()
# lifted = yellow_tape_pos.copy(); lifted[2] = 0.15
# goto_pose_arm1(lifted, gripper_down_quat)
# goto_home_joint_position_arm1()

# # Arm1: move to handover (shifted toward arm0)
# goto_pose_arm1(handover_pos, gripper_rotated_side_quat)

# # Arm0 approach
# # arm0_quat = np.array([0.707, 0.707, 0, 0])
# arm0_quat = gripper_side_quat
# open_gripper_arm0()
# # goto_pose_arm0(arm0_handover_pos + np.array([0.1, 0, 0.12]), arm0_quat, z_approach=0.1)
# # goto_pose_arm0(arm0_handover_pos, arm0_quat, z_approach=0.12)
# goto_pose_arm0(arm0_handover_pos + np.array([0, 0.02, 0]), arm0_quat, z_approach=0.10)
# goto_pose_arm0(arm0_handover_pos + np.array([0, 0.02, 0]), arm0_quat, z_approach=0.01)
# close_gripper_arm0()

# # Arm1: release and retract
# open_gripper_arm1()
# shifted_handover_pos = handover_pos + vtf.SO3(wxyz=gripper_rotated_side_quat).as_matrix() @ np.array([0, 0, -0.1])
# shifted_arm0_pos = arm0_handover_pos + vtf.SO3(wxyz=arm0_quat).as_matrix() @ np.array([0, 0, -0.1])
# goto_pose_arm1(shifted_handover_pos, gripper_rotated_side_quat)
# goto_pose_arm0(shifted_arm0_pos, arm0_quat)
# goto_home_joint_position_arm1()
# goto_home_joint_position_arm0()

# # Arm0: drop yellow tape at duct tape position, shifted to the left because the tape is slightly off-center in the robot's grasp
# goto_pose_arm0((duct_tape_pos+np.array([-0.02, 0, 0.05])), gripper_down_quat, z_approach=0.15)
# open_gripper_arm0()
# goto_pose_arm0((duct_tape_pos+np.array([0, 0, 0.2])), gripper_down_quat)
# goto_home_joint_position_arm0()
#     """

    # 7. Run the step with the hardcoded action
    print("\nExecuting hardcoded action via exec_env.step()...")
    # This call triggers exec(action_code, ...) inside the executor
    obs, reward, terminated, truncated, info = exec_env.step(action_code)
    
    # 9. Create directory based on tape initialization information
    # Directory name encodes the yellow and duct tape offsets (and handover rand suffix if used)
    dir_name = f"handover_yellow_{yellow_offset_args[0]}_{yellow_offset_args[1]}_{yellow_offset_args[2]}_duct_{duct_offset_args[0]}_{duct_offset_args[1]}_{duct_offset_args[2]}".replace(".", "_").replace("-", "neg")
    if use_handover_rand:
        rand_suffix = rng_handover.integers(0, 1000000)
        dir_name = f"{dir_name}_handover_rand_{rand_suffix}"
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
