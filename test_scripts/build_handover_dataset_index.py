#!/usr/bin/env python3
"""
Build a JSON index mapping numerical ids to each handover in the dataset folder.
Each entry includes the path and the x,y coordinates of the yellow and duct tape
(parsed from the directory name: handover_yellow_{x}_{y}_{z}_duct_{x}_{y}_{z}
 with . -> _ and - -> neg).
"""

import json
import os


def parse_num(lead: str, rest: str) -> float:
    """Parse a number from directory name parts, e.g. '0','08' -> 0.08; 'neg0','065' -> -0.065."""
    decimal = rest if rest else "0"
    if lead == "neg0":
        return -float("0." + decimal)
    if lead == "0":
        return float("0." + decimal)
    return float(lead + "." + rest)


def parse_handover_dirname(name: str) -> dict | None:
    """
    Parse directory name like:
      handover_yellow_0_08_0_065_0_0_duct_0_08_neg0_065_0_0
    into yellow (x,y) and duct (x,y). Returns None if format doesn't match.
    """
    if not name.startswith("handover_yellow_") or "_duct_" not in name:
        return None
    rest = name.replace("handover_yellow_", "", 1)
    parts = rest.split("_duct_")
    if len(parts) != 2:
        return None
    yellow_part, duct_part = parts[0], parts[1]
    y_tokens = yellow_part.split("_")
    d_tokens = duct_part.split("_")
    # Each position is (lead, rest) e.g. (0, 08), (neg0, 065)
    if len(y_tokens) < 6 or len(d_tokens) < 6:
        return None
    try:
        yellow_x = parse_num(y_tokens[0], y_tokens[1])
        yellow_y = parse_num(y_tokens[2], y_tokens[3])
        duct_x = parse_num(d_tokens[0], d_tokens[1])
        duct_y = parse_num(d_tokens[2], d_tokens[3])
    except (ValueError, IndexError):
        return None
    return {"yellow_x": yellow_x, "yellow_y": yellow_y, "duct_x": duct_x, "duct_y": duct_y}


# Excluded positions (must match sweep_handover_offsets.py)
EXCLUDED_YELLOW = (-0.28, 0.325)
EXCLUDED_DUCT = (-0.28, -0.325)


def main():
    dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")
    out_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "handover_index.json")

    entries = []
    dirs = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("handover_")])

    for dir_name in dirs:
        coords = parse_handover_dirname(dir_name)
        if coords is None:
            print(f"Warning: could not parse dir name: {dir_name}")
            coords = {}
        # Skip excluded combos (sweep only has 64: 8 yellow x 8 duct positions)
        yellow_xy = (coords.get("yellow_x"), coords.get("yellow_y"))
        duct_xy = (coords.get("duct_x"), coords.get("duct_y"))
        if yellow_xy == EXCLUDED_YELLOW or duct_xy == EXCLUDED_DUCT:
            continue
        rel_path = os.path.join("dataset", dir_name)
        abs_path = os.path.join(dataset_root, dir_name)
        entries.append({
            "path": rel_path,
            "path_absolute": os.path.abspath(abs_path),
            "dir_name": dir_name,
            "yellow_x": coords.get("yellow_x"),
            "yellow_y": coords.get("yellow_y"),
            "duct_x": coords.get("duct_x"),
            "duct_y": coords.get("duct_y"),
        })

    for idx, e in enumerate(entries):
        e["id"] = idx

    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
