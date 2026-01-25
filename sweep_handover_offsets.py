import subprocess
import os
import numpy as np
import wandb
import imageio
import sys
from datetime import datetime

def calc_steps(min_val, max_val, num):
    if num <= 1:
        return 0.0
    return (max_val - min_val) / (num - 1)

def get_dir_name(yellow_offset_str, duct_offset_str):
    """Replicate the directory naming logic in test_handover_step.py"""
    y_vals = [float(x.strip()) for x in yellow_offset_str.split(',')]
    d_vals = [float(x.strip()) for x in duct_offset_str.split(',')]
    
    dir_name = f"handover_yellow_{y_vals[0]}_{y_vals[1]}_{y_vals[2]}_duct_{d_vals[0]}_{d_vals[1]}_{d_vals[2]}"
    return dir_name.replace(".", "_").replace("-", "neg")

def main():
    YELLOW_BASE_X = 0.0
    YELLOW_BASE_Y = 0.0
    DUCT_BASE_X = 0.0
    DUCT_BASE_Y = 0.0

    # Range definitions from sweep_handover_offsets.sh
    YELLOW_X_MIN, YELLOW_X_MAX = -0.2, 0.1
    YELLOW_Y_MIN, YELLOW_Y_MAX = 0.25, 0.5
    DUCT_X_MIN, DUCT_X_MAX = -0.2, 0.1
    DUCT_Y_MIN, DUCT_Y_MAX = -0.5, -0.25

    NUM_X = 4
    NUM_Y = 2
    
    YELLOW_X_STEP = calc_steps(YELLOW_X_MIN, YELLOW_X_MAX, NUM_X)
    YELLOW_Y_STEP = calc_steps(YELLOW_Y_MIN, YELLOW_Y_MAX, NUM_Y)
    DUCT_X_STEP = calc_steps(DUCT_X_MIN, DUCT_X_MAX, NUM_X)
    DUCT_Y_STEP = calc_steps(DUCT_Y_MIN, DUCT_Y_MAX, NUM_Y)

    # Generate yellow positions
    yellow_positions = []
    for i in range(NUM_X):
        x_offset = YELLOW_X_MIN + i * YELLOW_X_STEP
        x = YELLOW_BASE_X + x_offset
        for j in range(NUM_Y):
            y_offset = YELLOW_Y_MIN + j * YELLOW_Y_STEP
            y = YELLOW_BASE_Y + y_offset
            yellow_positions.append(f"{x:.6f},{y:.6f},0.0")

    # Generate duct positions
    duct_positions = []
    for i in range(NUM_X):
        x_offset = DUCT_X_MIN + i * DUCT_X_STEP
        x = DUCT_BASE_X + x_offset
        for j in range(NUM_Y):
            y_offset = DUCT_Y_MIN + j * DUCT_Y_STEP
            y = DUCT_BASE_Y + y_offset
            duct_positions.append(f"{x:.6f},{y:.6f},0.0")

    total_combos = len(yellow_positions) * len(duct_positions)
    current_combo = 0
    MAX_RETRIES = 3
    
    # Project name
    project_name = f"robosuite-handover-sweep-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("==========================================")
    print("Handover Offset Sweep (Python version)")
    print("==========================================")
    print(f"Total combinations: {total_combos}")
    print("==========================================")

    for yellow_offset in yellow_positions:
        for duct_offset in duct_positions:
            current_combo += 1
            
            # Unique ID for this offset combination to allow resuming/updating on retry
            # We use the directory name as a basis for the run ID to ensure it's unique and consistent
            combo_id = get_dir_name(yellow_offset, duct_offset)
            run_id = f"sweep_{combo_id}"
            
            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                prefix = f"[{current_combo}/{total_combos}]"
                if attempt > 1:
                    print(f"{prefix} Attempt {attempt}/{MAX_RETRIES} for:")
                else:
                    print(f"{prefix} Running with:")
                print(f"  Yellow tape offset: [{yellow_offset}]")
                print(f"  Duct tape offset:   [{duct_offset}]")

                # Initialize or resume wandb run
                # We finish the run after each attempt to ensure data is synced, 
                # but use the same run_id to "update" the same run on retries.
                run = wandb.init(
                    project=project_name,
                    id=run_id,
                    resume="allow",
                    config={
                        "yellow_offset": yellow_offset,
                        "duct_offset": duct_offset,
                        "combo_idx": current_combo,
                        "yellow_x": float(yellow_offset.split(',')[0]),
                        "yellow_y": float(yellow_offset.split(',')[1]),
                        "duct_x": float(duct_offset.split(',')[0]),
                        "duct_y": float(duct_offset.split(',')[1]),
                    },
                    name=combo_id,
                    group="sweep_v1"
                )

                # Run the simulation using the venv's python
                # Use --flag=value format to handle negative numbers correctly
                cmd = [
                    "venv/bin/python", "test_scripts/test_handover_step.py",
                    f"--yellow_offset={yellow_offset}",
                    f"--duct_offset={duct_offset}"
                ]
                
                # Execute command and wait
                process = subprocess.run(cmd)
                exit_code = process.returncode

                if exit_code == 0:
                    print("  ✓ Completed successfully")
                    success = True
                elif exit_code == 2:
                    # Note: argparse also returns 2 on argument errors
                    print(f"  ✗ Failed: Video length > 1 min or Argument Error (exit code {exit_code})")
                else:
                    print(f"  ✗ Failed with exit code {exit_code}")

                # Path to the saved video
                dataset_dir = os.path.join("dataset", combo_id)
                video_path = os.path.join(dataset_dir, "handover_agentview.mp4")

                log_data = {
                    "exit_code": exit_code,
                    "attempt": attempt,
                    "success": success,
                }

                # Extract last frame and video length if video exists
                if os.path.exists(video_path):
                    try:
                        # Iterate through frames to robustly get the last one and the count
                        # This avoids "Reached end of video" errors with some ffmpeg versions
                        reader = imageio.get_reader(video_path)
                        last_frame = None
                        num_frames = 0
                        for frame in reader:
                            last_frame = frame
                            num_frames += 1
                        reader.close()
                        
                        if last_frame is not None:
                            log_data["video_length_frames"] = num_frames
                            log_data["last_frame"] = wandb.Image(last_frame, caption=f"Last frame - Attempt {attempt}")
                        
                        # Also upload the video to wandb
                        log_data["video"] = wandb.Video(video_path, fps=20, format="mp4")
                    except Exception as e:
                        print(f"  ! Warning: Could not read video for wandb logging: {e}")

                # Log everything to wandb
                run.log(log_data)
                
                # We always finish the run. If we retry, wandb.init(resume="allow") with same ID
                # will effectively "update" the same run in the wandb dashboard.
                run.finish()

                if success:
                    break
                
                if attempt < MAX_RETRIES:
                    print(f"  Retrying... ({attempt}/{MAX_RETRIES})\n")
            
            if not success:
                print(f"  !!! Giving up after {MAX_RETRIES} attempts")
            print("")

    print("==========================================")
    print("Sweep completed!")
    print(f"Total combinations run: {current_combo}")
    print("==========================================")

if __name__ == "__main__":
    main()
