import json
import argparse
import os
from pathlib import Path

# Grid specification (copied from sweep script)
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

# Grid positioning
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

# Yellow tape: left 3 columns in image (positive y, cols 3, 4, 5)
YELLOW_COLS = COL_CENTERS[3:]  # y = +0.065, +0.195, +0.325

# Duct tape: right 3 columns in image (negative y, cols 0, 1, 2)
DUCT_COLS = COL_CENTERS[:3]    # y = -0.325, -0.195, -0.065

# Excluded corner positions
EXCLUDED_YELLOW_POSITION = (ROW_CENTERS[0], YELLOW_COLS[2])  # (-0.28, 0.325)
EXCLUDED_DUCT_POSITION = (ROW_CENTERS[0], DUCT_COLS[0])  # (-0.28, -0.325)


def get_dir_name(yellow_offset_str, duct_offset_str):
    """Replicate the directory naming logic in test_handover_step.py"""
    y_vals = [float(x.strip()) for x in yellow_offset_str.split(',')]
    d_vals = [float(x.strip()) for x in duct_offset_str.split(',')]
    
    dir_name = f"handover_yellow_{y_vals[0]}_{y_vals[1]}_{y_vals[2]}_duct_{d_vals[0]}_{d_vals[1]}_{d_vals[2]}"
    return dir_name.replace(".", "_").replace("-", "neg")


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
                        "trajectory_id": combo_id,
                        "yellow_offset": yellow_offset,
                        "duct_offset": duct_offset,
                        "yellow_x": x_yellow,
                        "yellow_y": y_yellow,
                        "duct_x": x_duct,
                        "duct_y": y_duct,
                    })
    
    return combinations


def save_mapping(output_dir):
    """
    Save trajectory ID to tape offset mapping as JSON.
    
    Args:
        output_dir: Directory to save the JSON file
    """
    combinations = generate_valid_combinations()
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping: trajectory_index -> {yellow_offset: [x,y,z], duct_offset: [x,y,z]}
    mapping = {}
    for idx, combo in enumerate(combinations):
        mapping[idx] = {
            "yellow_offset": [combo["yellow_x"], combo["yellow_y"], 0.0],
            "duct_offset": [combo["duct_x"], combo["duct_y"], 0.0]
        }
    
    output_file = output_dir / "trajectory_to_offsets.json"
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Saved {len(mapping)} trajectory mappings to {output_file}")
    print(f"Excluded positions:")
    print(f"  - Yellow at ({EXCLUDED_YELLOW_POSITION[0]}, {EXCLUDED_YELLOW_POSITION[1]})")
    print(f"  - Duct at ({EXCLUDED_DUCT_POSITION[0]}, {EXCLUDED_DUCT_POSITION[1]})")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Save trajectory ID to tape offset mapping as JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the JSON file (default: current directory)"
    )
    
    args = parser.parse_args()
    
    save_mapping(args.output_dir)


if __name__ == "__main__":
    main()