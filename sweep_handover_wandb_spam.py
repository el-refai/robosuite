"""
WandB sweep that runs the same handover command 500 times.
Sweep parameter run_idx is 1..500 and is only used to get 500 runs; values are otherwise meaningless.
"""
import subprocess
import os
import wandb
import imageio
import sys
import argparse
from datetime import datetime

# Fixed command args (same every run)
YELLOW_OFFSET = "-0.28,0.065,0.0"
DUCT_OFFSET = "-0.1,-0.065,0.0"
X_SHIFT = "0.0"
Y_SHIFT = "0.0"
ANGLE_SHIFT = "0.0"
PERTURB_SPHERE_OFFSET = "0.0,.02,0"
PERTURB_RADIUS = "0.02"

NUM_RUNS = 500


def get_dir_name(yellow_offset_str, duct_offset_str, x_shift=None, y_shift=None, angle_shift=None):
    """Replicate the directory naming logic in test_handover_step.py."""
    y_vals = [float(x.strip()) for x in yellow_offset_str.split(',')]
    d_vals = [float(x.strip()) for x in duct_offset_str.split(',')]
    base = f"handover_yellow_{y_vals[0]}_{y_vals[1]}_{y_vals[2]}_duct_{d_vals[0]}_{d_vals[1]}_{d_vals[2]}"
    if x_shift is not None and y_shift is not None and angle_shift is not None:
        base = f"{base}_x_{x_shift}_y_{y_shift}_angle_{angle_shift}"
    return base.replace(".", "_").replace("-", "neg")


def create_sweep_config(project_name):
    """Create wandb sweep configuration: 500 runs, run_idx 1..500 (values meaningless)."""
    sweep_config = {
        "name": "handover-spam-500",
        "method": "grid",
        "metric": {
            "name": "success",
            "goal": "maximize"
        },
        "parameters": {
            "run_idx": {
                "values": list(range(1, NUM_RUNS + 1))
            }
        }
    }
    return sweep_config


def run_single(run_idx):
    """Run the fixed handover command once."""
    print(f"Running {run_idx}/{NUM_RUNS}")

    cmd = [
        "venv/bin/python", "test_scripts/test_handover_step.py",
        f"--yellow_offset={YELLOW_OFFSET}",
        f"--duct_offset={DUCT_OFFSET}",
        f"--x_shift={X_SHIFT}",
        f"--y_shift={Y_SHIFT}",
        f"--angle_shift={ANGLE_SHIFT}",
        "--perturb_sphere_offset", PERTURB_SPHERE_OFFSET,
        "--perturb_radius", PERTURB_RADIUS,
    ]

    process = subprocess.run(cmd)
    exit_code = process.returncode
    success = exit_code == 0

    if success:
        print("  ✓ Completed successfully")
    elif exit_code == 2:
        print(f"  ✗ Failed: Video length > 1 min or Argument Error (exit code {exit_code})")
    else:
        print(f"  ✗ Failed with exit code {exit_code}")

    log_data = {
        "exit_code": exit_code,
        "success": success,
        "run_idx": run_idx,
    }

    combo_id_with_shifts = get_dir_name(YELLOW_OFFSET, DUCT_OFFSET, X_SHIFT, Y_SHIFT, ANGLE_SHIFT)
    dataset_dir = os.path.join("dataset", combo_id_with_shifts)
    video_path = os.path.join(dataset_dir, "handover_agentview.mp4")

    if os.path.exists(video_path):
        try:
            reader = imageio.get_reader(video_path)
            last_frame = None
            num_frames = 0
            for frame in reader:
                last_frame = frame
                num_frames += 1
            reader.close()
            if last_frame is not None:
                log_data["video_length_frames"] = num_frames
                log_data["last_frame"] = wandb.Image(last_frame, caption="Last frame")
            log_data["video"] = wandb.Video(video_path, fps=20, format="mp4")
        except Exception as e:
            print(f"  ! Warning: Could not read video for wandb logging: {e}")

    return log_data


def sweep_agent_fn():
    """Function called by each wandb sweep agent."""
    run = wandb.init()
    run_idx = wandb.config.run_idx
    log_data = run_single(run_idx)
    wandb.log(log_data)
    run.finish()


def create_sweep(project_name):
    """Create a new wandb sweep and return the sweep ID."""
    sweep_config = create_sweep_config(project_name)

    print("==========================================")
    print("Creating WandB Sweep (500 runs, same command)")
    print("==========================================")
    print(f"Project: {project_name}")
    print(f"Runs: {NUM_RUNS} (run_idx 1..500)")
    print("==========================================")

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    print(f"\nSweep created with ID: {sweep_id}")
    print(f"\nTo run agents in parallel, open multiple terminals and run:")
    print(f"  python sweep_handover_wandb_spam.py --run-agent --sweep-id {sweep_id} --project {project_name}")
    print(f"\nOr run a single agent with --count to limit runs:")
    print(f"  python sweep_handover_wandb_spam.py --run-agent --sweep-id {sweep_id} --project {project_name} --count 10")

    return sweep_id


def run_agent(sweep_id, project_name, count=None):
    """Run a wandb sweep agent."""
    print("==========================================")
    print("Starting WandB Sweep Agent")
    print("==========================================")
    print(f"Sweep ID: {sweep_id}")
    print(f"Project: {project_name}")
    if count:
        print(f"Max runs: {count}")
    print("==========================================")

    wandb.agent(sweep_id, function=sweep_agent_fn, project=project_name, count=count)


def main():
    parser = argparse.ArgumentParser(description="WandB Sweep: run same handover command 500 times")
    parser.add_argument("--create-sweep", action="store_true", help="Create a new sweep")
    parser.add_argument("--run-agent", action="store_true", help="Run a sweep agent")
    parser.add_argument("--sweep-id", type=str, default=None, help="Sweep ID to run agent for")
    parser.add_argument("--project", type=str, default=None,
                        help="WandB project name (default: robosuite-handover-spam-YYYYMMDD_HHMMSS)")
    parser.add_argument("--count", type=int, default=None, help="Maximum number of runs for this agent")

    args = parser.parse_args()

    if args.project is None:
        args.project = f"robosuite-handover-spam-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.create_sweep:
        create_sweep(args.project)
    elif args.run_agent:
        if not args.sweep_id:
            print("Error: --sweep-id is required when running an agent")
            print("First create a sweep with --create-sweep, then use the returned sweep ID")
            sys.exit(1)
        run_agent(args.sweep_id, args.project, args.count)
    else:
        sweep_id = create_sweep(args.project)
        print("\n" + "=" * 42)
        print("Starting local agent...")
        print("=" * 42 + "\n")
        run_agent(sweep_id, args.project, args.count)


if __name__ == "__main__":
    main()
