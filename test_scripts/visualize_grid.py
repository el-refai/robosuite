"""Visualize the sweep grid by placing markers at each cell center."""

import os
import sys
import numpy as np
import imageio

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config


def main():
    # Grid specification (same as sweep_handover_offsets.py)
    CELL_HEIGHT = 0.18  # x-direction
    CELL_WIDTH = 0.13   # y-direction
    NUM_ROWS = 3
    NUM_COLS = 6
    
    # Table dimensions
    TABLE_X = 0.74
    TABLE_Y = 1.19
    TABLE_Z_OFFSET = 0.8  # Table height
    
    # Grid positioning
    GRID_X_MIN = -TABLE_X / 2  # -0.37
    GRID_Y_MIN = -(NUM_COLS * CELL_WIDTH) / 2  # -0.39
    
    # Calculate cell centers
    row_centers = [GRID_X_MIN + (row + 0.5) * CELL_HEIGHT for row in range(NUM_ROWS)]
    col_centers = [GRID_Y_MIN + (col + 0.5) * CELL_WIDTH for col in range(NUM_COLS)]
    
    print("Grid Configuration:")
    print(f"  Cell size: {CELL_HEIGHT}m (x) x {CELL_WIDTH}m (y)")
    print(f"  Grid: {NUM_ROWS} rows x {NUM_COLS} columns")
    print(f"  Row centers (x): {[f'{x:.3f}' for x in row_centers]}")
    print(f"  Column centers (y): {[f'{y:.3f}' for y in col_centers]}")
    # Camera is at -x looking toward +x, so positive y = LEFT in image, negative y = RIGHT
    print(f"  Yellow tape columns (left in image, positive y): y = {[f'{col_centers[i]:.3f}' for i in range(3, 6)]}")
    print(f"  Duct tape columns (right in image, negative y): y = {[f'{col_centers[i]:.3f}' for i in range(3)]}")
    
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
    
    # Position to hide a tape (far below)
    hidden_qpos = np.array([0, 0, -10, 1, 0, 0, 0])
    
    print("\nRendering grid positions...")
    
    # Store all frames in a grid layout
    # We'll create a 3x6 grid of images (rows x cols)
    frame_size = 256
    all_frames = []
    
    for row_idx, x in enumerate(row_centers):
        row_frames = []
        for col_idx, y in enumerate(col_centers):
            z = TABLE_Z_OFFSET + 0.02  # Slightly above table
            quat = np.array([1, 0, 0, 0])
            qpos = np.concatenate([[x, y, z], quat])
            
            # Camera at -x looking toward +x: positive y = LEFT, negative y = RIGHT
            # Yellow tape: cols 3-5 (positive y, left in image)
            # Duct tape: cols 0-2 (negative y, right in image)
            if col_idx >= 3:
                sim.data.set_joint_qpos(yellow_tape_joint, qpos)
                sim.data.set_joint_qpos(duct_tape_joint, hidden_qpos)
                label = f"Y[{row_idx},{col_idx}]"
            else:
                sim.data.set_joint_qpos(yellow_tape_joint, hidden_qpos)
                sim.data.set_joint_qpos(duct_tape_joint, qpos)
                label = f"D[{row_idx},{col_idx}]"
            
            sim.forward()
            
            # Render frame
            frame = sim.render(
                camera_name="agentview",
                width=frame_size,
                height=frame_size,
                depth=False,
            )[::-1].copy()
            
            # Add label text overlay (simple approach - add colored border)
            if col_idx >= 3:
                # Yellow border for yellow tape positions (left in image, positive y)
                frame[:5, :] = [255, 255, 0]  # Top
                frame[-5:, :] = [255, 255, 0]  # Bottom
                frame[:, :5] = [255, 255, 0]  # Left
                frame[:, -5:] = [255, 255, 0]  # Right
            else:
                # Gray border for duct tape positions (right in image, negative y)
                frame[:5, :] = [128, 128, 128]
                frame[-5:, :] = [128, 128, 128]
                frame[:, :5] = [128, 128, 128]
                frame[:, -5:] = [128, 128, 128]
            
            row_frames.append(frame)
            print(f"  Rendered position ({row_idx}, {col_idx}): x={x:.3f}, y={y:.3f}")
        
        # Stack row horizontally
        row_image = np.concatenate(row_frames, axis=1)
        all_frames.append(row_image)
    
    # Stack all rows vertically
    grid_image = np.concatenate(all_frames, axis=0)
    
    # Save grid montage
    output_path = os.path.join(root_dir, "grid_visualization.png")
    imageio.imwrite(output_path, grid_image)
    print(f"\nSaved grid montage to: {output_path}")
    print(f"  Image size: {grid_image.shape[1]}x{grid_image.shape[0]}")
    
    # Also render a single high-res image with BOTH tapes visible at corner positions
    # to show the overall layout
    print("\nRendering overview with both tapes...")
    
    # Place yellow tape on left side (positive y), duct tape on right side (negative y)
    yellow_pos = np.array([row_centers[0], col_centers[4], TABLE_Z_OFFSET + 0.02])  # positive y = left
    duct_pos = np.array([row_centers[0], col_centers[1], TABLE_Z_OFFSET + 0.02])    # negative y = right
    
    sim.data.set_joint_qpos(yellow_tape_joint, np.concatenate([yellow_pos, [1, 0, 0, 0]]))
    sim.data.set_joint_qpos(duct_tape_joint, np.concatenate([duct_pos, [1, 0, 0, 0]]))
    sim.forward()
    
    overview_frame = sim.render(
        camera_name="agentview",
        width=1024,
        height=1024,
        depth=False,
    )[::-1].copy()
    
    overview_path = os.path.join(root_dir, "grid_overview.png")
    imageio.imwrite(overview_path, overview_frame)
    print(f"Saved overview to: {overview_path}")
    
    # Print position summary
    print("\n" + "="*60)
    print("GRID POSITION SUMMARY")
    print("="*60)
    print("\nYellow tape positions (LEFT in image, positive y, cols 3-5, yellow border):")
    for row_idx, x in enumerate(row_centers):
        for col_idx in range(3, 6):
            y = col_centers[col_idx]
            print(f"  Row {row_idx}, Col {col_idx}: x={x:.3f}, y={y:.3f}")
    
    print("\nDuct tape positions (RIGHT in image, negative y, cols 0-2, gray border):")
    for row_idx, x in enumerate(row_centers):
        for col_idx in range(3):
            y = col_centers[col_idx]
            print(f"  Row {row_idx}, Col {col_idx}: x={x:.3f}, y={y:.3f}")
    
    print("\n" + "="*60)
    print(f"Total yellow positions: 9")
    print(f"Total duct positions: 9")
    print(f"Total sweep combinations: 81")
    print("="*60)
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
