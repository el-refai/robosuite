import os
import sys
import numpy as np
import imageio
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up"
macros.IMAGE_CONVENTION = "opencv"

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# Now we can import the classes
from robosuite.environments.custom.franka_robosuite_tape_handover import FrankaRobosuiteTapeHandover
from robosuite.environments.custom.control.base_executor import CodeExecEnvConfig
from api.franka_priviledged_api import FrankaControlTapeHandoverPrivilegedApi
from api.base_api import register_api
from api.franka_tape_handover import FrankaTapeHandoverCodeEnv

def main():
    # Register the API so CodeExecutionEnvBase can find it
    # The name here is used by CodeExecutionEnvBase to look up the API
    register_api("franka-handover-privileged", lambda env: FrankaControlTapeHandoverPrivilegedApi(env))

    # 1. Instantiate the low-level environment
    print("Initializing low-level FrankaRobosuiteTapeHandover environment...")
    low_level_env = FrankaRobosuiteTapeHandover(
        viser_debug=False,
        privileged=True,
        enable_render=False,
    )

    # 2. Define the configuration for the high-level code execution environment
    # We specify the low-level env and the API we just registered
    cfg = CodeExecEnvConfig(
        low_level=low_level_env,
        apis=["franka-handover-privileged"],
    )

    # 3. Instantiate the high-level environment
    # This environment subclasses CodeExecutionEnvBase and defines its own prompt and oracle_code
    print("Initializing high-level FrankaTapeHandoverCodeEnv...")
    exec_env = FrankaTapeHandoverCodeEnv(cfg)

    # 4. Reset the environment
    print("Resetting environment...")
    obs, info = exec_env.reset()

    # 5. Enable video recording
    print("Enabling video capture...")
    exec_env.enable_video_capture(True)

    # 6. Execute the oracle code
    print("\nExecuting oracle action via exec_env.step()...")
    oracle_code = exec_env.oracle_code
    obs, reward, terminated, truncated, info = exec_env.step(oracle_code)

    # 7. Save the recorded video
    video_frames = exec_env.get_video_frames()
    if video_frames:
        video_path = "franka_tape_handover_oracle.mp4"
        print(f"Saving video with {len(video_frames)} frames to {video_path}...")
        imageio.mimsave(video_path, video_frames, fps=20)
        print(f"Video saved to {video_path}")
    else:
        print("No video frames were captured.")

    # 8. Print execution results and logs
    print("\n" + "="*40)
    print("STEP EXECUTION RESULTS")
    print("="*40)
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Task Completed: {info.get('task_completed')}")
    
    print("\n--- STDOUT FROM CODE EXECUTION ---")
    print(info['stdout'])
    
    if info['stderr']:
        print("\n--- STDERR FROM CODE EXECUTION ---")
        print(info['stderr'])
    
    print("\nDone.")

if __name__ == "__main__":
    main()
