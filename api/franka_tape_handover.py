from robosuite.environments.custom.control.base_executor import CodeExecutionEnvBase

PROMPT = """
You are controlling a two-arm Franka Emika robot system with API described below.
Goal: Arm 1 should pick up the yellow tape, lift it, and hand it over to Arm 0. Arm 0 should then grasp the yellow tape. The yellow tape should be lifted above the table surface (height threshold ~0.1m) to count as a successful handover.

Guidance:
- You should initially grip the yellow tape with arm1 from above.
- When you are trying to move the arm to the handover position, you should use quaternion = [0, 1, 0, 0]
- The position at which you hand the yellow tape from arm1 to arm0 should be between the two arms.

Coordinate system:
- All pose functions accept positions in robot0's base frame.
- Z-axis: Up
- X-axis: Forward
- Y-axis: Left

You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""

# pos x right
# pos y forward
# z up
ORACLE_CODE = """
import numpy as np
import viser.transforms as vtf

# --- Get poses ---
yellow_tape_pos, yellow_tape_quat = get_object_pose("yellow tape")
duct_tape_pos, duct_tape_quat = get_object_pose("duct tape")

arm1_pos, _ = get_arm1_gripper_pose()
arm0_pos, _ = get_arm0_gripper_pose()
handover_pos = (arm1_pos + arm0_pos) / 2
handover_pos[0] -= 0.15 # shift the handover position back by 15cm to make it more reachable
handover_pos[1] += 0.1 # shift the handover position left by 10cm to make it more reachable
arm0_handover_pos = handover_pos.copy()
# Need a way to get the width of the yellow tape that isnt privileged
# this is half the width of the franka gripper:
arm0_handover_pos[1] -= 0.1025 # shift handover position right by half the width of the franka gripper
arm0_handover_pos[0] += 0.02 # shift handover position forward by 2cm

# --- Pickup orientation ---
gripper_down_quat = np.array([0, 1, 0, 0])
# old one: gripper_side_matrix = vtf.SO3(wxyz=[0.707, 0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_side_matrix = vtf.SO3(wxyz=[0.707, 0, -0.707, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_side_quat = gripper_side_matrix.wxyz
# old one: gripper_rotated_side_matrix = vtf.SO3(wxyz=[0.707, -0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_rotated_side_matrix = vtf.SO3(wxyz=[0.707, 0, 0.707, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_rotated_side_quat = gripper_rotated_side_matrix.wxyz

# Arm1: pick up yellow tape
open_gripper_arm1()
# shift the pick -y by 2cm to grab the tape on one end of the radius
goto_pose_arm1((yellow_tape_pos+np.array([-0.01, 0.05, -0.01])), gripper_down_quat, z_approach=0.15)
close_gripper_arm1()
lifted = yellow_tape_pos.copy(); lifted[2] = 0.15
goto_pose_arm1(lifted, gripper_down_quat)
goto_home_joint_position_arm1()

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
goto_home_joint_position_arm0()

# Arm0: drop cube in bowl, shifted to the left because the tape is slightly off-center in the robot's grasp
goto_pose_arm0((duct_tape_pos+np.array([0, -0.05, 0.05])), gripper_down_quat, z_approach=0.15)
open_gripper_arm0()
goto_pose_arm0((duct_tape_pos+np.array([0, -0.05, 0.2])), gripper_down_quat)
goto_home_joint_position_arm0()
"""

ORACLE_CODE_ORIG = """
import numpy as np
import viser.transforms as vtf

# --- Get poses ---
yellow_tape_pos, yellow_tape_quat = get_object_pose("yellow tape")
duct_tape_pos, duct_tape_quat = get_object_pose("duct tape")

arm1_pos, _ = get_arm1_gripper_pose()
arm0_pos, _ = get_arm0_gripper_pose()
handover_pos = (arm1_pos + arm0_pos) / 2
arm0_handover_pos = handover_pos.copy()
# Need a way to get the width of the yellow tape that isnt privileged
# this is half the width of the franka gripper:
arm0_handover_pos[2] += 0.1025
arm0_handover_pos[1] += 0.01

# --- Pickup orientation ---
gripper_down_quat = np.array([0, 1, 0, 0])
gripper_side_matrix = vtf.SO3(wxyz=[0.707, 0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_side_quat = gripper_side_matrix.wxyz
gripper_rotated_side_matrix = vtf.SO3(wxyz=[0.707, -0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_rotated_side_quat = gripper_rotated_side_matrix.wxyz

# Arm1: pick up yellow tape
open_gripper_arm1()
# shift the pick y by 2cm to grab the tape on the left side of the radius
goto_pose_arm1((yellow_tape_pos+np.array([-0.01, 0.05, -0.01])), gripper_down_quat, z_approach=0.15)
close_gripper_arm1()
lifted = yellow_tape_pos.copy(); lifted[2] = 0.15
goto_pose_arm1(lifted, gripper_down_quat)
goto_home_joint_position_arm1()

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
goto_pose_arm1(shifted_handover_pos, gripper_rotated_side_quat)
goto_pose_arm0(shifted_arm0_pos, arm0_quat)
goto_home_joint_position_arm1()
goto_home_joint_position_arm0()

# Arm0: drop cube in bowl, shifted to the left because the tape is slightly off-center in the robot's grasp
goto_pose_arm0((duct_tape_pos+np.array([0, 0.05, 0.05])), gripper_down_quat, z_approach=0.15)
open_gripper_arm0()
goto_pose_arm0((duct_tape_pos+np.array([0, 0.05, 0.2])), gripper_down_quat)
goto_home_joint_position_arm0()
"""


# ---------------------------- High-level Env -----------------------------
class FrankaTapeHandoverCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for Franka tape handover task."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE
    # oracle_code = ORACLE_CODE_ORIG

    def compute_reward(self) -> float:
        # Delegate to low-level environment for reward computation
        return self.low_level_env.compute_reward()


__all__ = [
    "FrankaTapeHandoverCodeEnv",
]