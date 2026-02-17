#!/usr/bin/env python3
"""
Post-process handover dataset: remove still frames from videos and align joint data.

Uses the agentview video to detect still frames (consecutive frames with change below
a threshold) and removes those frames from agentview, robot0_wrist, and robot1_wrist
videos. robot0_joints and robot1_joints are adjusted to match the new frame count
(interpolated to video length if needed, then same indices removed).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import imageio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove still frames from handover dataset videos and align joint data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        type=Path,
        help="One or more dataset directories (each with agentview, robot0_wrist, robot1_wrist mp4s and robot0/robot1_joints npz).",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="If a path is a directory, expand to all its immediate subdirectories (process all trajectories in that folder).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=1.0,
        help="Frame difference threshold: frames with mean absolute difference from previous frame below this are removed.",
    )
    parser.add_argument(
        "--base-filename",
        default="handover",
        help="Base filename (without camera/joints suffix) for mp4 and npz files.",
    )
    parser.add_argument(
        "--diff-metric",
        choices=("mean", "max", "l2"),
        default="mean",
        help="Metric for frame difference: mean absolute, max absolute, or L2 (RMSE) per pixel.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("filtered_dataset"),
        help="Directory to write filtered datasets (one subdir per input dataset). Default: filtered_dataset/",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original dataset directories instead of writing to --output-dir.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="When using --in-place, back up original files as .bak before overwriting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which frames would be removed and counts; do not write files.",
    )
    return parser.parse_args()


def frame_difference(prev: np.ndarray, curr: np.ndarray, metric: str) -> float:
    """Compute difference between two frames. Both are uint8 HxWxC."""
    diff = np.abs(curr.astype(np.float64) - prev.astype(np.float64))
    if metric == "mean":
        return float(np.mean(diff))
    if metric == "max":
        return float(np.max(diff))
    if metric == "l2":
        return float(np.sqrt(np.mean(diff**2)))
    raise ValueError(f"Unknown metric: {metric}")


def compute_keep_mask_from_video(
    video_path: Path, threshold: float, diff_metric: str
) -> tuple[int, np.ndarray]:
    """
    Compute boolean mask of frames to keep by streaming the video (low memory).
    Returns (n_frames, keep_mask). Frame 0 is always kept.
    Frame i is removed if diff(frames[i], frames[i-1]) < threshold.
    """
    keep_list: list[bool] = []
    prev: np.ndarray | None = None
    n = 0
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        for curr in reader:
            keep_list.append(True)  # will set frame 0 to True, then maybe False for i>0
            if prev is not None:
                d = frame_difference(prev, curr, diff_metric)
                if d < threshold:
                    keep_list[-1] = False
            prev = curr
            n += 1
    finally:
        reader.close()
    keep = np.array(keep_list, dtype=bool) if keep_list else np.ones(0, dtype=bool)
    return n, keep


def write_filtered_video(
    src_path: Path, out_path: Path, keep_mask: np.ndarray, fps: int = 20
) -> int:
    """Stream src video and write only frames where keep_mask[i] is True. Returns number written."""
    reader = imageio.get_reader(str(src_path), "ffmpeg")
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", format="FFMPEG")
    written = 0
    try:
        for i, frame in enumerate(reader):
            if i >= len(keep_mask):
                break
            if keep_mask[i]:
                writer.append_data(frame)
                written += 1
    finally:
        reader.close()
        writer.close()
    return written


def interpolate_joints_to_length(
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray | None,
    gripper_positions: np.ndarray | None,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Resample joint arrays from current length to target_length using linear interpolation.
    joint_positions shape (T, D). Returns arrays of shape (target_length, D).
    """
    T, D = joint_positions.shape
    old_indices = np.arange(T)
    new_indices = np.linspace(0, T - 1, target_length)

    new_joint_positions = np.zeros((target_length, D), dtype=joint_positions.dtype)
    for col in range(D):
        new_joint_positions[:, col] = np.interp(
            new_indices, old_indices, joint_positions[:, col]
        )

    new_joint_velocities = None
    if joint_velocities is not None:
        new_joint_velocities = np.zeros(
            (target_length, joint_velocities.shape[1]), dtype=joint_velocities.dtype
        )
        for col in range(joint_velocities.shape[1]):
            new_joint_velocities[:, col] = np.interp(
                new_indices, old_indices, joint_velocities[:, col]
            )

    new_gripper_positions = None
    if gripper_positions is not None:
        G = gripper_positions.shape[1] if gripper_positions.ndim > 1 else 1
        if gripper_positions.ndim == 1:
            gripper_positions = gripper_positions[:, None]
        new_gripper_positions = np.zeros((target_length, G), dtype=gripper_positions.dtype)
        for col in range(G):
            new_gripper_positions[:, col] = np.interp(
                new_indices, old_indices, gripper_positions[:, col]
            )
        if G == 1:
            new_gripper_positions = new_gripper_positions.squeeze(-1)

    return new_joint_positions, new_joint_velocities, new_gripper_positions


def process_dataset(
    dataset_dir: Path,
    base_filename: str,
    threshold: float,
    diff_metric: str,
    output_dir: Path | None,
    in_place: bool,
    backup: bool,
    dry_run: bool,
    output_subdir: Path | None = None,
) -> bool:
    """Process a single dataset directory. Returns True on success."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        print(f"Skip (not a directory): {dataset_dir}", file=sys.stderr)
        return False

    # Output: same dir when in-place, else output_dir / (output_subdir or dataset_dir.name)
    if in_place:
        out_dir = dataset_dir
    else:
        name = output_subdir if output_subdir is not None else dataset_dir.name
        out_dir = output_dir.resolve() / name
        out_dir.mkdir(parents=True, exist_ok=True)

    agentview_path = dataset_dir / f"{base_filename}_agentview.mp4"
    robot0_wrist_path = dataset_dir / f"{base_filename}_robot0_wrist.mp4"
    robot1_wrist_path = dataset_dir / f"{base_filename}_robot1_wrist.mp4"
    robot0_joints_path = dataset_dir / f"{base_filename}_robot0_joints.npz"
    robot1_joints_path = dataset_dir / f"{base_filename}_robot1_joints.npz"

    if not agentview_path.is_file():
        print(f"Skip (missing agentview): {dataset_dir}", file=sys.stderr)
        return False

    # Compute keep mask by streaming agentview (low memory)
    n_frames, keep_mask = compute_keep_mask_from_video(agentview_path, threshold, diff_metric)
    n_keep = int(np.sum(keep_mask))
    n_remove = n_frames - n_keep

    print(f"\n{dataset_dir.name}")
    print(f"  Agentview frames: {n_frames} -> keep {n_keep}, remove {n_remove}")
    if not in_place:
        print(f"  Output: {out_dir}")

    if n_remove == 0 and not dry_run:
        # No still frames: copy videos and joint files (no decode/encode)
        fps = 20
        for src in [agentview_path, robot0_wrist_path, robot1_wrist_path]:
            if src.is_file():
                dst = out_dir / src.name
                shutil.copy2(src, dst)
                print(f"  Copied {src.name} (no still frames removed)")
        for joint_path in [robot0_joints_path, robot1_joints_path]:
            if joint_path.is_file():
                shutil.copy2(joint_path, out_dir / joint_path.name)
                print(f"  Copied {joint_path.name}")
        return True

    if dry_run:
        print("  [dry-run] Would remove frames at indices:", np.where(~keep_mask)[0].tolist()[:20], "..." if n_remove > 20 else "")
        return True

    if not robot0_wrist_path.is_file() or not robot1_wrist_path.is_file():
        print(f"  Skip: missing robot wrist videos", file=sys.stderr)
        return False

    # Stream-write filtered videos (one frame in memory at a time)
    fps = 20
    for src_path, out_name in [
        (agentview_path, f"{base_filename}_agentview.mp4"),
        (robot0_wrist_path, f"{base_filename}_robot0_wrist.mp4"),
        (robot1_wrist_path, f"{base_filename}_robot1_wrist.mp4"),
    ]:
        out_path = out_dir / out_name
        write_path = out_path
        if in_place and src_path.resolve() == out_path.resolve():
            # Avoid reading and writing the same file: write to temp then move
            write_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if in_place and backup and out_path.is_file():
            bak_path = out_path.with_suffix(out_path.suffix + ".bak")
            shutil.copy2(out_path, bak_path)
            print(f"  Backed up {out_name} -> {out_name}.bak")
        written = write_filtered_video(src_path, write_path, keep_mask, fps=fps)
        if write_path != out_path:
            shutil.move(str(write_path), str(out_path))
        print(f"  Saved {out_name} with {written} frames")

    # Load and adjust joint npz files; write to out_dir
    for joint_path in [robot0_joints_path, robot1_joints_path]:
        if not joint_path.is_file():
            print(f"  Warning: missing {joint_path}; skipping joint adjustment.", file=sys.stderr)
            continue
        data = dict(np.load(joint_path, allow_pickle=True))
        joint_positions = data["joint_positions"]
        joint_velocities = data.get("joint_velocities")
        gripper_positions = data.get("gripper_positions")

        T_joint = joint_positions.shape[0]
        if T_joint != n_frames:
            joint_positions, joint_velocities, gripper_positions = interpolate_joints_to_length(
                joint_positions, joint_velocities, gripper_positions, n_frames
            )
        joint_positions = joint_positions[keep_mask]
        if joint_velocities is not None:
            joint_velocities = joint_velocities[keep_mask]
        if gripper_positions is not None:
            gripper_positions = gripper_positions[keep_mask]

        out_data = {"joint_positions": joint_positions}
        if joint_velocities is not None:
            out_data["joint_velocities"] = joint_velocities
        if gripper_positions is not None:
            out_data["gripper_positions"] = gripper_positions

        out_joint_path = out_dir / joint_path.name
        if in_place and backup:
            bak_path = joint_path.with_suffix(joint_path.suffix + ".bak")
            shutil.copy2(joint_path, bak_path)
            print(f"  Backed up {joint_path.name} -> {joint_path.name}.bak")
        np.savez_compressed(out_joint_path, **out_data)
        print(f"  Saved {out_joint_path.name} with {len(joint_positions)} samples")

    return True


def expand_dirs(
    dirs: list[Path], expand_all: bool
) -> list[tuple[Path, Path | None]]:
    """
    If expand_all, replace directory args with their subdirectories.
    Returns list of (dataset_path, output_subdir_override).
    output_subdir_override: when set, used instead of dataset_dir.name for output path
    (e.g. parent_name / subdir_name to preserve structure).
    """
    if not expand_all:
        return [(p.resolve(), None) for p in dirs]
    out: list[tuple[Path, Path | None]] = []
    for p in dirs:
        p = p.resolve()
        if p.is_dir():
            subdirs = sorted(d for d in p.iterdir() if d.is_dir())
            if not subdirs:
                out.append((p, None))
            else:
                for sub in subdirs:
                    # Preserve structure: output_dir/parent_name/subdir_name
                    out.append((sub, Path(p.name) / sub.name))
        else:
            out.append((p, None))
    return out


def main():
    args = parse_args()
    expanded = expand_dirs(args.dataset_dirs, args.all)
    if args.all and len(expanded) > len(args.dataset_dirs):
        print(f"Processing {len(expanded)} trajectories.\n")
    output_dir = None if args.in_place else args.output_dir
    for dataset_path, out_subdir_override in expanded:
        process_dataset(
            dataset_path,
            base_filename=args.base_filename,
            threshold=args.threshold,
            diff_metric=args.diff_metric,
            output_dir=output_dir,
            in_place=args.in_place,
            backup=args.backup,
            dry_run=args.dry_run,
            output_subdir=out_subdir_override,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
