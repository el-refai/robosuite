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
    # Grid specification:
    # - Each cell: 0.18m tall (x-direction) x 0.13m wide (y-direction)
    # - Grid: 3 rows tall x 6 columns wide
    # - Left 3 columns: yellow tape, Right 3 columns: duct tape
    # - Grid centered on table (y-direction), bottom edge flush with table bottom (x-direction)
    
    CELL_HEIGHT = 0.18  # x-direction (depth from camera view)
    CELL_WIDTH = 0.13   # y-direction (left-right from camera view)
    NUM_ROWS = 3        # rows in x-direction
    NUM_COLS = 6        # total columns in y-direction (3 yellow + 3 duct)
    
    # Table dimensions: 0.74m (x) x 1.19m (y)
    TABLE_X = 0.74
    TABLE_Y = 1.19
    
    # Grid dimensions
    GRID_HEIGHT = NUM_ROWS * CELL_HEIGHT  # 0.54m
    GRID_WIDTH = NUM_COLS * CELL_WIDTH    # 0.78m
    
    # Grid positioning:
    # - Centered on table in y-direction
    # - Bottom edge flush with table bottom in x-direction (x = -TABLE_X/2)
    GRID_X_MIN = -TABLE_X / 2  # -0.37 (bottom of table)
    GRID_Y_MIN = -GRID_WIDTH / 2  # -0.39 (centered on table)
    
    # Calculate row centers (x-direction)
    row_centers = []
    for row in range(NUM_ROWS):
        x = GRID_X_MIN + (row + 0.5) * CELL_HEIGHT
        row_centers.append(x)
    
    # Calculate column centers (y-direction)
    col_centers = []
    for col in range(NUM_COLS):
        y = GRID_Y_MIN + (col + 0.5) * CELL_WIDTH
        col_centers.append(y)
    
    # Camera is at -x looking toward +x, so positive y = LEFT in image, negative y = RIGHT
    # Yellow tape: left 3 columns in image (positive y, cols 3, 4, 5)
    yellow_cols = col_centers[3:]  # y = +0.065, +0.195, +0.325
    
    # Duct tape: right 3 columns in image (negative y, cols 0, 1, 2)
    duct_cols = col_centers[:3]    # y = -0.325, -0.195, -0.065
    
    # Generate yellow positions (3 cols x 3 rows = 9 positions)
    yellow_positions = []
    for x in row_centers:
        for y in yellow_cols:
            yellow_positions.append(f"{x:.6f},{y:.6f},0.0")

    # Generate duct positions (3 cols x 3 rows = 9 positions)
    duct_positions = []
    for x in row_centers:
        for y in duct_cols:
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
