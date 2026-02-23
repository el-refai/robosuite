import os
import sys
import argparse
from datetime import datetime
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


def sample_funnel_preimage(
    cone_half_angle_rad: float,
    distances_from_bottleneck: list[float],
    n_radial_rings: int = 2,
    n_theta_per_ring: int = 8,
    n_circle_per_disk: int = 12,
) -> list[tuple[float, float, float]]:
    """
    Deterministic funnel preimage: disks working backward from bottleneck.
    Bottleneck = handover; axis points toward handover (approach direction).
    Disk at distance d has center at bottleneck - d*axis, radius = d * tan(cone_half_angle).
    Returns list of (d, r, theta) in meters and radians.
    """
    out = []
    for i, d in enumerate(distances_from_bottleneck):
        r_max = d * np.tan(cone_half_angle_rad)
        if r_max <= 0:
            continue
        if i == 0:
            # Small disk near bottleneck: radial sampling to avoid clumping
            out.append((d, 0.0, 0.0))  # center
            for ring in range(1, n_radial_rings + 1):
                r = r_max * ring / (n_radial_rings + 1)
                for k in range(n_theta_per_ring):
                    theta = 2.0 * np.pi * k / n_theta_per_ring
                    out.append((d, r, theta))
            # Edge of small disk
            for k in range(n_circle_per_disk):
                theta = 2.0 * np.pi * k / n_circle_per_disk
                out.append((d, r_max, theta))
        else:
            # Larger disks: sample evenly on circle only
            for k in range(n_circle_per_disk):
                theta = 2.0 * np.pi * k / n_circle_per_disk
                out.append((d, r_max, theta))
    return out


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test handover step with configurable tape offsets",
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
        default=0.02,
        help='Radius (m) of sphere for sampling perturbed waypoint before handover; 0 to disable (default: 0.02)'
    )
    parser.add_argument(
        '--perturb_sphere_offset',
        type=parse_offset_list,
        default='0.0,0.0,0.0',
        help='Translation offset (x,y,z) of sphere/cone base from handover_pos, in rotated frame (default: 0,0,0)'
    )
    parser.add_argument(
        '--cone_half_angle',
        type=float,
        default=0.15,
        help='Cone half-angle in radians for waypoint sampling; used when perturb_radius > 0 (default: 0.15 ~ 8.6 deg)'
    )
    parser.add_argument(
        '--no_skip_intermediate_yellow_arm',
        dest='skip_intermediate_yellow_arm',
        action='store_false',
        default=True,
        help='Include intermediate yellow arm waypoints (steps 2 and 3: lifted z=0.15 and above_pickup_at_handover_height). Default is to skip them.'
    )
    parser.add_argument(
        '--funnel',
        action='store_true',
        help='Use deterministic funnel preimage: disks backward from bottleneck, one demo per preimage point.'
    )
    parser.add_argument(
        '--funnel_distances',
        type=str,
        default='0.02,0.05,0.10',
        help='Comma-separated distances (m) from bottleneck for funnel disks (default: 0.02,0.05,0.10)'
    )
    parser.add_argument(
        '--funnel_n_radial',
        type=int,
        default=2,
        help='Number of radial rings on smallest disk (default: 2)'
    )
    parser.add_argument(
        '--funnel_n_theta_ring',
        type=int,
        default=8,
        help='Points per ring on smallest disk (default: 8)'
    )
    parser.add_argument(
        '--funnel_n_circle',
        type=int,
        default=12,
        help='Points on each disk circle/edge (default: 12)'
    )
    parser.add_argument(
        '--funnel_n_run',
        type=int,
        default=None,
        metavar='N',
        help='Max number of funnel points to run (default: all)'
    )
    args = parser.parse_args()
    
    # Extract offsets as numpy arrays
    yellow_offset_args = args.yellow_offset
    duct_offset_args = args.duct_offset
    yellow_offset = np.array(yellow_offset_args)
    duct_offset = np.array(duct_offset_args)

    x_shift = float(args.x_shift)
    y_shift = float(args.y_shift)
    angle_shift = float(args.angle_shift)
    perturb_radius = float(args.perturb_radius)
    perturb_sphere_offset = np.array(args.perturb_sphere_offset)
    cone_half_angle = float(args.cone_half_angle)
    skip_intermediate_yellow_arm = args.skip_intermediate_yellow_arm
    use_funnel = getattr(args, 'funnel', False)
    funnel_distances = [float(x) for x in args.funnel_distances.split(',')]
    funnel_n_radial = getattr(args, 'funnel_n_radial', 2)
    funnel_n_theta_ring = getattr(args, 'funnel_n_theta_ring', 8)
    funnel_n_circle = getattr(args, 'funnel_n_circle', 12)
    funnel_n_run = getattr(args, 'funnel_n_run', None)

    print(f"Yellow tape offset: {yellow_offset_args}")
    print(f"Duct tape offset: {duct_offset_args}")
    print(f"Handover position shifts: x_shift={x_shift}, y_shift={y_shift}, angle_shift={angle_shift}")
    print(f"Perturbation sphere radius: {perturb_radius} m" + (" (disabled)" if perturb_radius <= 0 else ""))
    print(f"Perturbation sphere center offset: {perturb_sphere_offset}")
    if perturb_radius > 0:
        print(f"Cone half-angle: {cone_half_angle} rad (~{np.degrees(cone_half_angle):.1f} deg)")
    
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
    funnel_d_val = None
    funnel_r_val = None
    funnel_theta_val = None

    if use_funnel:
        preimage = sample_funnel_preimage(
            cone_half_angle, funnel_distances,
            n_radial_rings=funnel_n_radial,
            n_theta_per_ring=funnel_n_theta_ring,
            n_circle_per_disk=funnel_n_circle,
        )
        runs = list(enumerate(preimage))
        if funnel_n_run is not None:
            runs = runs[:funnel_n_run]
            print(f"Running first {len(runs)} of {len(preimage)} funnel points (--funnel_n_run={funnel_n_run})")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"handover_funnel_yellow_{yellow_offset_args[0]}_{yellow_offset_args[1]}_{yellow_offset_args[2]}_duct_{duct_offset_args[0]}_{duct_offset_args[1]}_{duct_offset_args[2]}_x_{x_shift}_y_{y_shift}_angle_{angle_shift}_{timestamp}".replace(".", "_").replace("-", "neg")
        dataset_dir = os.path.join("dataset", dir_name)
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Funnel mode: {len(runs)} preimage points, output dir: {dataset_dir}")
    else:
        runs = [(0, (None, None, None))]

    for run_idx, (d, r, theta) in runs:
        funnel_d_val, funnel_r_val, funnel_theta_val = d, r, theta
        action_code = f"""import numpy as np
import viser.transforms as vtf

# --- Get poses ---
yellow_tape_pos, yellow_tape_quat = get_object_pose("yellow tape")
duct_tape_pos, duct_tape_quat = get_object_pose("duct tape")

arm1_pos, _ = get_arm1_gripper_pose()
arm0_pos, _ = get_arm0_gripper_pose()
center = (arm1_pos + arm0_pos) / 2
# Z-axis rotation for angle_shift (radians): handover position and offsets are rotated around center
angle_shift = {angle_shift}
Rz_quat = np.array([np.cos(angle_shift / 2), 0, 0, np.sin(angle_shift / 2)])  # wxyz
Rz = vtf.SO3(wxyz=Rz_quat).as_matrix()
# Translation offset from center (then rotated by angle_shift)
handover_offset = np.array([-0.15 - {x_shift}, 0.1 + {y_shift}, 0.0])
handover_pos = center + Rz @ handover_offset
# Arm0 handover offset (0.035, -0.1025, 0) applied in the rotated frame so it stays consistent
arm0_offset = np.array([0.035, -0.1025, 0.0])
arm0_handover_pos = handover_pos + Rz @ arm0_offset

# --- Pickup orientation ---
gripper_down_quat = np.array([0, 1, 0, 0])
# Base side orientations; then rotate by angle_shift around z so approach aligns with rotated handover
gripper_side_matrix = vtf.SO3(wxyz=[0.707, 0, -0.707, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_rotated_side_matrix = vtf.SO3(wxyz=[0.707, 0, 0.707, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_side_matrix = vtf.SO3(wxyz=Rz_quat) @ gripper_side_matrix
gripper_rotated_side_matrix = vtf.SO3(wxyz=Rz_quat) @ gripper_rotated_side_matrix
gripper_side_quat = gripper_side_matrix.wxyz
gripper_rotated_side_quat = gripper_rotated_side_matrix.wxyz

# Arm1: pick up yellow tape
open_gripper_arm1()
# shift the pick -y by 2cm to grab the tape on one end of the radius
goto_pose_arm1((yellow_tape_pos+np.array([-0.01, 0.05, -0.02])), gripper_down_quat, z_approach=0.15)
close_gripper_arm1()
skip_intermediate_yellow_arm = {str(skip_intermediate_yellow_arm)}
if not skip_intermediate_yellow_arm:
    lifted = yellow_tape_pos.copy(); lifted[2] = 0.15
    goto_pose_arm1(lifted, gripper_down_quat)
    above_pickup_at_handover_height = lifted.copy()
    above_pickup_at_handover_height[2] = get_arm_base_midpoint_z()
    goto_pose_arm1(above_pickup_at_handover_height, gripper_down_quat)

# Waypoint: deterministic funnel (d,r,theta) or random cone; gripper points waypoint->handover
funnel_d_injected = {repr(funnel_d_val)}
funnel_r_injected = {repr(funnel_r_val)}
funnel_theta_injected = {repr(funnel_theta_val)}
perturb_radius = {perturb_radius}
perturb_sphere_offset = np.array({perturb_sphere_offset.tolist()})
cone_half_angle = {cone_half_angle}
if perturb_radius > 0 or funnel_d_injected is not None:
    cone_base = handover_pos + Rz @ perturb_sphere_offset
    axis = handover_pos - cone_base
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        axis = Rz @ np.array([1.0, 0.0, 0.0])
        axis_norm = np.linalg.norm(axis)
    else:
        axis = axis / axis_norm
    if abs(axis[2]) < 0.9:
        v1 = np.array([0, 0, 1])
    else:
        v1 = np.array([1, 0, 0])
    v1 = v1 - np.dot(v1, axis) * axis
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = np.cross(axis, v1)
    if funnel_d_injected is not None:
        disk_center = handover_pos - funnel_d_injected * axis
        perturbed_waypoint = disk_center + funnel_r_injected * (np.cos(funnel_theta_injected) * v1 + np.sin(funnel_theta_injected) * v2)
    else:
        theta = cone_half_angle * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        direction = np.cos(theta) * axis + np.sin(theta) * (np.cos(phi) * v1 + np.sin(phi) * v2)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        perturbed_waypoint = cone_base + perturb_radius * direction
    approach_dir = handover_pos - perturbed_waypoint
    approach_norm = np.linalg.norm(approach_dir)
    if approach_norm < 1e-9:
        waypoint_gripper_quat = gripper_rotated_side_quat
    else:
        approach_dir = approach_dir / approach_norm
        R_nom = vtf.SO3(wxyz=gripper_rotated_side_quat).as_matrix()
        nominal_forward = R_nom @ np.array([0.0, 0.0, -1.0])
        a, b = nominal_forward, approach_dir
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.clip(np.dot(a, b), -1.0, 1.0)
        if s < 1e-6:
            R_align = np.eye(3) if c > 0 else vtf.SO3.from_z_radians(np.pi).as_matrix()
        else:
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            angle = np.arccos(c)
            R_align = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        R_waypoint = R_align @ R_nom
        waypoint_gripper_quat = vtf.SO3.from_matrix(R_waypoint).wxyz
    goto_pose_arm1(perturbed_waypoint, waypoint_gripper_quat)

# Arm1: move to handover (shifted toward arm0)
goto_pose_arm1(handover_pos, gripper_rotated_side_quat)

# Arm0 approach
# arm0_quat = np.array([0.707, 0.707, 0, 0])
arm0_quat = gripper_side_quat
open_gripper_arm0()
goto_pose_arm0(arm0_handover_pos, arm0_quat, z_approach=0.10)
close_gripper_arm0()

# Arm1: release and retract
open_gripper_arm1()
shifted_handover_pos = handover_pos + vtf.SO3(wxyz=gripper_rotated_side_quat).as_matrix() @ np.array([0, 0, -0.1])
shifted_arm0_pos = arm0_handover_pos + vtf.SO3(wxyz=arm0_quat).as_matrix() @ np.array([0, 0, -0.1])
goto_pose_arm0(shifted_arm0_pos, arm0_quat)
goto_pose_arm1(shifted_handover_pos, gripper_rotated_side_quat)
goto_home_joint_position_arm1()
# goto_home_joint_position_arm0()

# Arm0: drop cube in bowl, shifted to the left because the tape is slightly off-center in the robot's grasp
goto_pose_arm0((duct_tape_pos+np.array([0.0, -0.03, 0.05])), gripper_down_quat, z_approach=0.15)
open_gripper_arm0()
goto_pose_arm0((duct_tape_pos+np.array([0, -0.05, 0.2])), gripper_down_quat)
goto_home_joint_position_arm0()
    """

        if use_funnel or run_idx > 0:
            obs, info = exec_env.reset()
            exec_env.enable_video_capture(True, freq=joint_state_freq_steps)
            low_level_env.enable_joint_state_collection(True, clear=True, freq=joint_state_freq_steps)

        print("\nExecuting hardcoded action via exec_env.step()..." + (f" (funnel point {run_idx + 1}/{len(runs)})" if use_funnel else ""))
        obs, reward, terminated, truncated, info = exec_env.step(action_code)

        if use_funnel:
            base_filename = f"handover_{run_idx:04d}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"handover_yellow_{yellow_offset_args[0]}_{yellow_offset_args[1]}_{yellow_offset_args[2]}_duct_{duct_offset_args[0]}_{duct_offset_args[1]}_{duct_offset_args[2]}_x_{x_shift}_y_{y_shift}_angle_{angle_shift}_{timestamp}".replace(".", "_").replace("-", "neg")
            dataset_dir = os.path.join("dataset", dir_name)
            os.makedirs(dataset_dir, exist_ok=True)
            base_filename = "handover"
            print(f"Created directory: {dataset_dir}")

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

        # 10. Save the recorded videos (separate videos for each camera)
        all_video_frames = low_level_env.get_camera_frames()

        if all_video_frames:
            if "agentview" in all_video_frames and all_video_frames["agentview"]:
                agentview_frames = all_video_frames["agentview"]
                agentview_path = os.path.join(dataset_dir, f"{base_filename}_agentview.mp4")
                print(f"Saving agentview video with {len(agentview_frames)} frames to {agentview_path}...")
                imageio.mimsave(agentview_path, agentview_frames, fps=20)
                print(f"Agentview video saved to {agentview_path}")
            if "robot0_eye_in_hand" in all_video_frames and all_video_frames["robot0_eye_in_hand"]:
                robot0_frames = all_video_frames["robot0_eye_in_hand"]
                robot0_path = os.path.join(dataset_dir, f"{base_filename}_robot0_wrist.mp4")
                print(f"Saving robot0 wrist camera video with {len(robot0_frames)} frames to {robot0_path}...")
                imageio.mimsave(robot0_path, robot0_frames, fps=20)
                print(f"Robot0 wrist camera video saved to {robot0_path}")
            if "robot1_eye_in_hand" in all_video_frames and all_video_frames["robot1_eye_in_hand"]:
                robot1_frames = all_video_frames["robot1_eye_in_hand"]
                robot1_path = os.path.join(dataset_dir, f"{base_filename}_robot1_wrist.mp4")
                print(f"Saving robot1 wrist camera video with {len(robot1_frames)} frames to {robot1_path}...")
                imageio.mimsave(robot1_path, robot1_frames, fps=20)
                print(f"Robot1 wrist camera video saved to {robot1_path}")
        else:
            print("No video frames were captured.")

        joint_states = low_level_env.get_collected_joint_states(clear=False)
        if joint_states:
            print(f"\nSaving joint states to .npz files...")
            robot0_data = {}
            if joint_states and "robot0_joint_pos" in joint_states[0]:
                robot0_data["joint_positions"] = np.stack([state["robot0_joint_pos"] for state in joint_states])
            if joint_states and "robot0_joint_vel" in joint_states[0]:
                robot0_data["joint_velocities"] = np.stack([state["robot0_joint_vel"] for state in joint_states])
            if joint_states and "robot0_gripper_qpos" in joint_states[0]:
                robot0_data["gripper_positions"] = np.stack([state["robot0_gripper_qpos"] for state in joint_states])
            robot1_data = {}
            if joint_states and "robot1_joint_pos" in joint_states[0]:
                robot1_data["joint_positions"] = np.stack([state["robot1_joint_pos"] for state in joint_states])
            if joint_states and "robot1_joint_vel" in joint_states[0]:
                robot1_data["joint_velocities"] = np.stack([state["robot1_joint_vel"] for state in joint_states])
            if joint_states and "robot1_gripper_qpos" in joint_states[0]:
                robot1_data["gripper_positions"] = np.stack([state["robot1_gripper_qpos"] for state in joint_states])
            if robot0_data:
                robot0_filename = os.path.join(dataset_dir, f"{base_filename}_robot0_joints.npz")
                np.savez_compressed(robot0_filename, **robot0_data)
                print(f"Robot0 (arm0) joint states saved to {robot0_filename} ({len(joint_states)} samples)")
            if robot1_data:
                robot1_filename = os.path.join(dataset_dir, f"{base_filename}_robot1_joints.npz")
                np.savez_compressed(robot1_filename, **robot1_data)
                print(f"Robot1 (arm1) joint states saved to {robot1_filename} ({len(joint_states)} samples)")
        else:
            print("No joint states were collected.")

        print("\n" + "="*40)
        print("STEP EXECUTION RESULTS" + (f" (funnel point {run_idx + 1}/{len(runs)})" if use_funnel else ""))
        print("="*40)
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        task_completed = info.get('task_completed', False)
        print(f"Task Completed: {task_completed}")

        max_frames = 1200
        num_frames = len(all_video_frames["agentview"]) if all_video_frames and "agentview" in all_video_frames else 0
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