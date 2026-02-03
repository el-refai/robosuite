"""Visualize the sweep grid by placing markers at each cell center.

This script saves individual images for each possible start position
(yellow tape positions x duct tape positions) with both tapes visible.
These images can be used to create a video of all possible start states.

Uses the same offset ranges as sweep_handover_offsets.sh:
- Yellow tape: X in [-0.2, 0.1] (4 positions), Y in [0.25, 0.5] (2 positions) = 8 positions
- Duct tape: X in [-0.2, 0.1] (4 positions), Y in [-0.5, -0.25] (2 positions) = 8 positions
- Total: 8 x 8 = 64 combinations
"""

import os
import sys
import numpy as np
import imageio

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config


def generate_positions(x_min, x_max, y_min, y_max, num_x, num_y):
    """Generate grid positions matching sweep_handover_offsets.sh logic."""
    positions = []
    
    # Calculate step sizes
    x_step = (x_max - x_min) / (num_x - 1) if num_x > 1 else 0.0
    y_step = (y_max - y_min) / (num_y - 1) if num_y > 1 else 0.0
    
    # Generate positions (x varies in outer loop, y in inner loop - matches bash script)
    for i in range(num_x):
        x = x_min + i * x_step
        for j in range(num_y):
            y = y_min + j * y_step
            positions.append({
                'x_idx': i,
                'y_idx': j,
                'x': x,
                'y': y,
            })
    
    return positions


def main():
    # Grid specification (same as sweep_handover_offsets.sh)
    # Yellow tape ranges
    YELLOW_X_MIN = -0.2
    YELLOW_X_MAX = 0.1
    YELLOW_Y_MIN = 0.25
    YELLOW_Y_MAX = 0.5
    
    # Duct tape ranges
    DUCT_X_MIN = -0.2
    DUCT_X_MAX = 0.1
    DUCT_Y_MIN = -0.5
    DUCT_Y_MAX = -0.25
    
    # Grid dimensions (4x2 = 8 positions per tape)
    NUM_X = 4
    NUM_Y = 2
    
    # Table height
    TABLE_Z_OFFSET = 0.8
    
    # Generate positions
    yellow_positions = generate_positions(
        YELLOW_X_MIN, YELLOW_X_MAX, YELLOW_Y_MIN, YELLOW_Y_MAX, NUM_X, NUM_Y
    )
    duct_positions = generate_positions(
        DUCT_X_MIN, DUCT_X_MAX, DUCT_Y_MIN, DUCT_Y_MAX, NUM_X, NUM_Y
    )
    
    print("Grid Configuration (matching sweep_handover_offsets.sh):")
    print(f"  Yellow tape X range: [{YELLOW_X_MIN}, {YELLOW_X_MAX}] ({NUM_X} positions)")
    print(f"  Yellow tape Y range: [{YELLOW_Y_MIN}, {YELLOW_Y_MAX}] ({NUM_Y} positions)")
    print(f"  Duct tape X range: [{DUCT_X_MIN}, {DUCT_X_MAX}] ({NUM_X} positions)")
    print(f"  Duct tape Y range: [{DUCT_Y_MIN}, {DUCT_Y_MAX}] ({NUM_Y} positions)")
    print(f"  Yellow tape positions: {len(yellow_positions)}")
    print(f"  Duct tape positions: {len(duct_positions)}")
    print(f"  Total combinations: {len(yellow_positions) * len(duct_positions)}")
    
    # Create output directory for individual frames
    frames_dir = os.path.join(root_dir, "start_state_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"\nOutput directory: {frames_dir}")
    
    # Create environment
    print("\nInitializing environment...")
    controller_config = load_composite_controller_config(controller="envs/configs/panda_joint_ctrl_slow.json")
    
    env = suite.environments.manipulation.two_arm_tape_handover.TwoArmTapeHandover(
        robots=["Panda", "Panda"],
        env_configuration="parallel",
        has_renderer=True,
        has_offscreen_renderer=True,
        camera_names=["agentview"],
        renderer="mujoco",
        camera_heights=512,
        camera_widths=512,
        controller_configs=[controller_config, controller_config],
        horizon=100,
        reward_shaping=False,
        use_object_obs=True,
        use_camera_obs=True,
        yellow_tape_offset=np.array([0.0, -0.3, 0.0]),
        duct_tape_offset=np.array([0.0, 0.3, 0.0]),
    )
    
    env.reset()
    
    # Access the simulation
    sim = env.sim
    
    # Get tape joint names
    yellow_tape_joint = env.yellow_tape.joints[0]
    duct_tape_joint = env.duct_tape.joints[0]
    
    total_combinations = len(yellow_positions) * len(duct_positions)
    print(f"\nRendering all {total_combinations} start state combinations...")
    
    # Image settings
    frame_size = 512
    all_frames = []  # Store all frames for video
    frame_idx = 0
    
    for yellow_idx, yellow_pos in enumerate(yellow_positions):
        for duct_idx, duct_pos in enumerate(duct_positions):
            # Position both tapes
            z = TABLE_Z_OFFSET + 0.02  # Slightly above table
            quat = np.array([1, 0, 0, 0])
            
            yellow_qpos = np.concatenate([[yellow_pos['x'], yellow_pos['y'], z], quat])
            duct_qpos = np.concatenate([[duct_pos['x'], duct_pos['y'], z], quat])
            
            sim.data.set_joint_qpos(yellow_tape_joint, yellow_qpos)
            sim.data.set_joint_qpos(duct_tape_joint, duct_qpos)
            sim.forward()
            
            # Render frame
            frame = sim.render(
                camera_name="agentview",
                width=frame_size,
                height=frame_size,
                depth=False,
            )[::-1].copy()
            
            # Save individual frame
            frame_filename = f"frame_{frame_idx:03d}_yellow_x{yellow_pos['x_idx']}_y{yellow_pos['y_idx']}_duct_x{duct_pos['x_idx']}_y{duct_pos['y_idx']}.png"
            frame_path = os.path.join(frames_dir, frame_filename)
            imageio.imwrite(frame_path, frame)
            
            # Store frame for video
            all_frames.append(frame)
            
            print(f"  [{frame_idx+1:3d}/{total_combinations}] Yellow: x{yellow_pos['x_idx']}y{yellow_pos['y_idx']} @ ({yellow_pos['x']:.3f}, {yellow_pos['y']:.3f}), "
                  f"Duct: x{duct_pos['x_idx']}y{duct_pos['y_idx']} @ ({duct_pos['x']:.3f}, {duct_pos['y']:.3f})")
            
            frame_idx += 1
    
    # Save video of all frames
    video_path = os.path.join(root_dir, "start_states.mp4")
    print(f"\nSaving video to: {video_path}")
    imageio.mimwrite(video_path, all_frames, fps=2)  # 2 FPS so each state is visible for 0.5 seconds
    
    # Also create a grid montage of all combinations
    # Arrange as grid (yellow positions as rows, duct positions as columns)
    num_yellow = len(yellow_positions)
    num_duct = len(duct_positions)
    print(f"\nCreating {num_yellow}x{num_duct} grid montage...")
    montage_frame_size = 128
    montage_rows = []
    
    frame_idx = 0
    for yellow_idx in range(num_yellow):
        row_frames = []
        for duct_idx in range(num_duct):
            # Resize frame for montage
            frame = all_frames[frame_idx]
            # Simple downscale by taking every Nth pixel
            scale = frame_size // montage_frame_size
            small_frame = frame[::scale, ::scale]
            row_frames.append(small_frame)
            frame_idx += 1
        montage_rows.append(np.concatenate(row_frames, axis=1))
    
    montage = np.concatenate(montage_rows, axis=0)
    montage_path = os.path.join(root_dir, "start_states_montage.png")
    imageio.imwrite(montage_path, montage)
    print(f"Saved montage to: {montage_path}")
    print(f"  Montage size: {montage.shape[1]}x{montage.shape[0]}")
    
    # Create composite image showing ALL tape positions at once
    # Since we only have one of each tape object, we composite multiple renders
    print("\nCreating composite image with all tape positions...")
    
    composite_size = 1024
    hidden_qpos = np.array([0, 0, -10, 1, 0, 0, 0])
    z = TABLE_Z_OFFSET + 0.02
    quat = np.array([1, 0, 0, 0])
    
    # First, render a base frame with no tapes (both hidden)
    sim.data.set_joint_qpos(yellow_tape_joint, hidden_qpos)
    sim.data.set_joint_qpos(duct_tape_joint, hidden_qpos)
    sim.forward()
    
    base_frame = sim.render(
        camera_name="agentview",
        width=composite_size,
        height=composite_size,
        depth=False,
    )[::-1].copy().astype(np.float32)
    
    # Render each yellow tape position and extract just the tape
    yellow_frames = []
    for pos in yellow_positions:
        yellow_qpos = np.concatenate([[pos['x'], pos['y'], z], quat])
        sim.data.set_joint_qpos(yellow_tape_joint, yellow_qpos)
        sim.data.set_joint_qpos(duct_tape_joint, hidden_qpos)
        sim.forward()
        
        frame = sim.render(
            camera_name="agentview",
            width=composite_size,
            height=composite_size,
            depth=False,
        )[::-1].copy()
        yellow_frames.append(frame)
    
    # Render each duct tape position
    duct_frames = []
    for pos in duct_positions:
        sim.data.set_joint_qpos(yellow_tape_joint, hidden_qpos)
        duct_qpos = np.concatenate([[pos['x'], pos['y'], z], quat])
        sim.data.set_joint_qpos(duct_tape_joint, duct_qpos)
        sim.forward()
        
        frame = sim.render(
            camera_name="agentview",
            width=composite_size,
            height=composite_size,
            depth=False,
        )[::-1].copy()
        duct_frames.append(frame)
    
    # Composite: find pixels that differ from base and overlay them
    composite = base_frame.copy()
    
    # Add yellow tapes (find where they differ from base)
    for frame in yellow_frames:
        diff = np.abs(frame.astype(np.float32) - base_frame)
        mask = np.max(diff, axis=2) > 10  # Threshold for detecting tape pixels
        composite[mask] = frame[mask]
    
    # Add duct tapes
    for frame in duct_frames:
        diff = np.abs(frame.astype(np.float32) - base_frame)
        mask = np.max(diff, axis=2) > 10
        composite[mask] = frame[mask]
    
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    
    all_positions_path = os.path.join(root_dir, "all_grid_positions.png")
    imageio.imwrite(all_positions_path, composite)
    print(f"Saved all positions composite to: {all_positions_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("OUTPUT SUMMARY")
    print("="*60)
    print(f"\nIndividual frames: {frames_dir}/")
    print(f"  - {total_combinations} PNG files (frame_XXX_yellow_xI_yJ_duct_xI_yJ.png)")
    print(f"\nVideo: {video_path}")
    print(f"  - {total_combinations} frames at 2 FPS (~{total_combinations // 2} seconds)")
    print(f"\nMontage: {montage_path}")
    print(f"  - {num_yellow}x{num_duct} grid showing all combinations")
    print(f"  - Rows = yellow tape positions ({num_yellow})")
    print(f"  - Columns = duct tape positions ({num_duct})")
    print(f"\nAll positions composite: {all_positions_path}")
    print(f"  - Shows all {num_yellow} yellow + {num_duct} duct tape positions at once")
    print("="*60)
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
