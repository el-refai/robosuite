import subprocess
import os
import wandb
import imageio
import sys
import argparse
from datetime import datetime

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
ROW_CENTERS = []
for row in range(NUM_ROWS):
    x = GRID_X_MIN + (row + 0.5) * CELL_HEIGHT
    ROW_CENTERS.append(round(x, 6))

# Calculate column centers (y-direction)
COL_CENTERS = []
for col in range(NUM_COLS):
    y = GRID_Y_MIN + (col + 0.5) * CELL_WIDTH
    COL_CENTERS.append(round(y, 6))

# Camera is at -x looking toward +x, so positive y = LEFT in image, negative y = RIGHT
# Yellow tape: left 3 columns in image (positive y, cols 3, 4, 5)
YELLOW_COLS = COL_CENTERS[3:]  # y = +0.065, +0.195, +0.325

# Duct tape: right 3 columns in image (negative y, cols 0, 1, 2)
DUCT_COLS = COL_CENTERS[:3]    # y = -0.325, -0.195, -0.065

# Excluded corner positions (these cause issues)
# Yellow at (-0.28, 0.325) - extreme corner
EXCLUDED_YELLOW_POSITION = (ROW_CENTERS[0], YELLOW_COLS[2])  # (-0.28, 0.325)
# Duct at (-0.28, -0.325) - extreme corner  
EXCLUDED_DUCT_POSITION = (ROW_CENTERS[0], DUCT_COLS[0])  # (-0.28, -0.325)

def run_name_number_first(name):
    """Convert wandb run name from 'legendary-sweep-64' to '64-legendary-sweep'."""
    if not name:
        return name
    name = str(name).strip()
    parts = name.split("-")
    if not parts or not parts[-1].isdigit():
        return name
    num = parts[-1]
    rest = "-".join(parts[:-1])
    return f"{num}-{rest}"


def get_dir_name(yellow_offset_str, duct_offset_str):
    """Replicate the directory naming logic for combo identification."""
    y_vals = [float(x.strip()) for x in yellow_offset_str.split(',')]
    d_vals = [float(x.strip()) for x in duct_offset_str.split(',')]
    
    base = f"handover_yellow_{y_vals[0]}_{y_vals[1]}_{y_vals[2]}_duct_{d_vals[0]}_{d_vals[1]}_{d_vals[2]}"
    return base.replace(".", "_").replace("-", "neg")


def generate_valid_combinations():
    """Generate all valid position combinations, excluding corner positions."""
    combinations = []
    
    for x_yellow in ROW_CENTERS:
        for y_yellow in YELLOW_COLS:
            # Skip excluded yellow position
            if (x_yellow, y_yellow) == EXCLUDED_YELLOW_POSITION:
                continue
                
            for x_duct in ROW_CENTERS:
                for y_duct in DUCT_COLS:
                    # Skip excluded duct position
                    if (x_duct, y_duct) == EXCLUDED_DUCT_POSITION:
                        continue
                    
                    yellow_offset = f"{x_yellow:.6f},{y_yellow:.6f},0.0"
                    duct_offset = f"{x_duct:.6f},{y_duct:.6f},0.0"
                    combo_id = get_dir_name(yellow_offset, duct_offset)
                    
                    combinations.append({
                        "yellow_offset": yellow_offset,
                        "duct_offset": duct_offset,
                        "combo_id": combo_id,
                        "yellow_x": x_yellow,
                        "yellow_y": y_yellow,
                        "duct_x": x_duct,
                        "duct_y": y_duct,
                    })
    
    return combinations


def create_sweep_config(project_name):
    """Create wandb sweep configuration."""
    combinations = generate_valid_combinations()
    
    # Create parameter values as lists for the sweep
    sweep_config = {
        "name": "handover-position-sweep",
        "method": "grid",  # Run all combinations
        "metric": {
            "name": "success",
            "goal": "maximize"
        },
        "parameters": {
            "combo_idx": {
                "values": list(range(len(combinations)))
            }
        }
    }
    
    return sweep_config, combinations


def run_single_combination(combo_idx, combinations, script_path):
    """Run a single combination using the handover executor script."""
    combo = combinations[combo_idx]
    combo_id = combo["combo_id"]
    print(f"Running combination {combo_idx + 1}/{len(combinations)}: {combo_id}")

    # Run the simulation
    run = wandb.run
    run_name = None
    if run is not None:
        run_name = run.name or run.id
        # Ensure name is safe as a single directory component
        safe = str(run_name).replace("/", "_").replace("\\", "_")
        if safe != run_name:
            run.name = safe
            run_name = safe
            try:
                run.save()
            except Exception:
                pass
        # Use number-first format for dir: legendary-sweep-64 -> 64-legendary-sweep
        run_name = run_name_number_first(run_name)
        if run_name != (run.name or run.id):
            run.name = run_name
            try:
                run.save()
            except Exception:
                pass

    timestamp = "dataset_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join("dataset", timestamp, str(run_name)) if run_name else None

    script_path = os.path.abspath(script_path)
    cmd = [sys.executable, script_path]
    if dataset_dir is not None:
        cmd.append(f"--dataset_dir={dataset_dir}")

    process = subprocess.run(cmd)
    exit_code = process.returncode
    success = exit_code == 0
    
    if success:
        print("  ✓ Completed successfully")
    elif exit_code == 2:
        print(f"  ✗ Failed: Video length > 1 min or Argument Error (exit code {exit_code})")
    else:
        print(f"  ✗ Failed with exit code {exit_code}")
    
    # Prepare log data
    log_data = {
        "exit_code": exit_code,
        "success": success,
        "yellow_x": combo["yellow_x"],
        "yellow_y": combo["yellow_y"],
        "duct_x": combo["duct_x"],
        "duct_y": combo["duct_y"],
        "combo_id": combo_id,
    }

    if dataset_dir is not None:
        log_data["dataset_dir"] = dataset_dir

    def _log_video_and_last_frame(key_prefix: str, path: str):
        if not os.path.exists(path):
            return
        try:
            reader = imageio.get_reader(path)
            try:
                n = reader.count_frames()
                last_frame = reader.get_data(n - 1) if n and n > 0 else None
                num_frames = n
            except Exception:
                last_frame = None
                num_frames = 0
                for frame in reader:
                    last_frame = frame
                    num_frames += 1
            finally:
                reader.close()

            if num_frames:
                log_data[f"{key_prefix}_length_frames"] = int(num_frames)
            if last_frame is not None:
                log_data[f"{key_prefix}_last_frame"] = wandb.Image(last_frame, caption=f"{key_prefix} last frame")
            log_data[f"{key_prefix}_video"] = wandb.Video(path, fps=20, format="mp4")
        except Exception as e:
            print(f"  ! Warning: Could not read {key_prefix} video for wandb logging: {e}")

    if dataset_dir is not None:
        _log_video_and_last_frame("agentview", os.path.join(dataset_dir, "handover_agentview.mp4"))
        # _log_video_and_last_frame("robot0_wrist", os.path.join(dataset_dir, "handover_robot0_wrist.mp4"))
        # _log_video_and_last_frame("robot1_wrist", os.path.join(dataset_dir, "handover_robot1_wrist.mp4"))
    
    return log_data


def sweep_agent_fn(script_path):
    """Function called by each wandb sweep agent."""
    # Generate combinations (same for all agents)
    combinations = generate_valid_combinations()
    
    # Initialize wandb run (sweep will set the config)
    run = wandb.init()
    
    # Get the combo_idx assigned by the sweep
    combo_idx = wandb.config.combo_idx
    
    # Run the combination
    log_data = run_single_combination(combo_idx, combinations, script_path)
    
    # Log results
    wandb.log(log_data)
    
    run.finish()


def create_sweep(project_name):
    """Create a new wandb sweep and return the sweep ID."""
    sweep_config, combinations = create_sweep_config(project_name)
    
    print("==========================================")
    print("Creating WandB Sweep")
    print("==========================================")
    print(f"Project: {project_name}")
    print(f"Total valid combinations: {len(combinations)}")
    print(f"Excluded positions:")
    print(f"  - Yellow at ({EXCLUDED_YELLOW_POSITION[0]}, {EXCLUDED_YELLOW_POSITION[1]})")
    print(f"  - Duct at ({EXCLUDED_DUCT_POSITION[0]}, {EXCLUDED_DUCT_POSITION[1]})")
    print("==========================================")
    
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    print(f"\nSweep created with ID: {sweep_id}")
    print(f"\nTo run agents in parallel, open multiple terminals and run:")
    print(f"  python sweep_handover_wandb.py --run-agent --sweep-id {sweep_id} --project {project_name}")
    print(f"  (optional: add --script <path> to specify handover executor script)")
    print(f"\nOr run a single agent now with --count to limit runs:")
    print(f"  python sweep_handover_wandb.py --run-agent --sweep-id {sweep_id} --project {project_name} --count 10")
    
    return sweep_id


def run_agent(sweep_id, project_name, script_path, count=None):
    """Run a wandb sweep agent."""
    print("==========================================")
    print("Starting WandB Sweep Agent")
    print("==========================================")
    print(f"Sweep ID: {sweep_id}")
    print(f"Project: {project_name}")
    print(f"Script: {script_path}")
    if count:
        print(f"Max runs: {count}")
    print("==========================================")
    
    def fn():
        return sweep_agent_fn(script_path)
    
    wandb.agent(sweep_id, function=fn, project=project_name, count=count)


def main():
    parser = argparse.ArgumentParser(description="WandB Sweep for Handover Position Combinations")
    parser.add_argument("--create-sweep", action="store_true", 
                        help="Create a new sweep")
    parser.add_argument("--run-agent", action="store_true",
                        help="Run a sweep agent")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Sweep ID to run agent for")
    parser.add_argument("--project", type=str, 
                        default=None,
                        help="WandB project name (default: robosuite-handover-sweep-YYYYMMDD_HHMMSS)")
    parser.add_argument("--count", type=int, default=None,
                        help="Maximum number of runs for this agent")
    parser.add_argument("--list-combinations", action="store_true",
                        help="List all valid combinations and exit")
    parser.add_argument("--script", type=str, default="data_gen_scripts/handover_executor.py",
                        help="Path to the handover executor script to run")

    args = parser.parse_args()
    
    # Generate project name with timestamp if not specified
    if args.project is None:
        args.project = f"robosuite-handover-sweep-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if args.list_combinations:
        combinations = generate_valid_combinations()
        print(f"Total valid combinations: {len(combinations)}")
        print(f"\nExcluded positions:")
        print(f"  - Yellow at ({EXCLUDED_YELLOW_POSITION[0]}, {EXCLUDED_YELLOW_POSITION[1]})")
        print(f"  - Duct at ({EXCLUDED_DUCT_POSITION[0]}, {EXCLUDED_DUCT_POSITION[1]})")
        print(f"\nValid combinations:")
        for i, combo in enumerate(combinations):
            print(f"  {i}: {combo['combo_id']}")
        return
    
    if args.create_sweep:
        create_sweep(args.project)
    elif args.run_agent:
        if not args.sweep_id:
            print("Error: --sweep-id is required when running an agent")
            print("First create a sweep with --create-sweep, then use the returned sweep ID")
            sys.exit(1)
        run_agent(args.sweep_id, args.project, args.script, args.count)
    else:
        # Default: create sweep and start one agent
        sweep_id = create_sweep(args.project)
        print("\n" + "=" * 42)
        print("Starting local agent...")
        print("=" * 42 + "\n")
        run_agent(sweep_id, args.project, args.script, args.count)


if __name__ == "__main__":
    main()
