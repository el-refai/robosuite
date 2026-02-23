#!/usr/bin/env python3
"""
Clip all demonstrations in a dataset folder by time range.

For each subfolder under dataset_dir that contains the expected files:
- Clips agent view and two wrist camera videos from start_sec to end_sec (default 13s–31s).
- Trims robot0 and robot1 joint .npz files to the same time range (by index using joint_fps).

Expected per-episode files (base filename "handover" by default):
  {base}_agentview.mp4, {base}_robot0_wrist.mp4, {base}_robot1_wrist.mp4
  {base}_robot0_joints.npz, {base}_robot1_joints.npz
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Clip dataset demonstrations by time range (videos + joint npz).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "dataset_dir",
        type=Path,
        help="Root directory containing one subfolder per demonstration.",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Write clipped data here (one subdir per episode). If not set, overwrite in place.",
    )
    p.add_argument(
        "--start",
        type=float,
        default=13.0,
        help="Start time in seconds.",
    )
    p.add_argument(
        "--end",
        type=float,
        default=31.0,
        help="End time in seconds.",
    )
    p.add_argument(
        "--base-filename",
        default="handover",
        help="Base name for mp4 and npz files (e.g. handover_agentview.mp4).",
    )
    p.add_argument(
        "--joint-fps",
        type=float,
        default=20.0,
        help="Assumed joint/log rate (Hz) to convert start/end time to array indices.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list episodes and time range; do not write.",
    )
    return p.parse_args()


def get_episode_folders(dataset_dir: Path, base: str) -> list[Path]:
    required = [
        f"{base}_agentview.mp4",
        f"{base}_robot0_wrist.mp4",
        f"{base}_robot1_wrist.mp4",
        f"{base}_robot0_joints.npz",
        f"{base}_robot1_joints.npz",
    ]
    episodes = []
    for item in sorted(dataset_dir.iterdir()):
        if not item.is_dir():
            continue
        if all((item / f).exists() for f in required):
            episodes.append(item)
    return episodes


def clip_video_ffmpeg(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    duration = end_sec - start_sec
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-c",
            "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def trim_npz_by_time(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    joint_fps: float,
) -> None:
    start_idx = int(start_sec * joint_fps)
    end_idx = int(end_sec * joint_fps)
    data = dict(np.load(input_path, allow_pickle=True))
    for key in data:
        arr = data[key]
        if hasattr(arr, "shape") and len(arr.shape) >= 1:
            data[key] = arr[start_idx:end_idx].copy()
    np.savez_compressed(output_path, **data)


def process_episode(
    episode_dir: Path,
    out_dir: Path,
    base: str,
    start_sec: float,
    end_sec: float,
    joint_fps: float,
    in_place: bool,
    dry_run: bool,
) -> bool:
    if dry_run:
        print(f"  [dry-run] Would clip: {episode_dir.name}")
        return True
    if not in_place:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = episode_dir

    videos = [
        f"{base}_agentview.mp4",
        f"{base}_robot0_wrist.mp4",
        f"{base}_robot1_wrist.mp4",
    ]
    for name in videos:
        src = episode_dir / name
        dst = out_dir / name
        if in_place:
            # Write to temp then move so we don't read overwritten file
            dst = out_dir / (name + ".tmp")
        try:
            clip_video_ffmpeg(src, dst, start_sec, end_sec)
        except subprocess.CalledProcessError as e:
            print(f"  ffmpeg failed for {name}: {e}", file=sys.stderr)
            if dst.exists():
                dst.unlink()
            return False
        if in_place:
            (out_dir / (name + ".tmp")).replace(out_dir / name)

    for name in [f"{base}_robot0_joints.npz", f"{base}_robot1_joints.npz"]:
        src = episode_dir / name
        dst = out_dir / name
        if in_place:
            dst = out_dir / (name + ".tmp")
        trim_npz_by_time(src, dst, start_sec, end_sec, joint_fps)
        if in_place:
            (out_dir / (name + ".tmp")).replace(out_dir / name)

    return True


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        print(f"Not a directory: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    episodes = get_episode_folders(dataset_dir, args.base_filename)
    if not episodes:
        print(f"No episode folders found under {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    in_place = args.output_dir is None
    if in_place:
        print(f"Clipping in place: {dataset_dir} ({len(episodes)} episodes)")
    else:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Clipping to {args.output_dir}: {dataset_dir} ({len(episodes)} episodes)")
    print(f"  Time range: {args.start}s – {args.end}s (joint_fps={args.joint_fps})")
    if args.dry_run:
        print("  [dry-run] No files will be modified.")

    ok = 0
    for ep in episodes:
        out_dir = (args.output_dir / ep.name) if args.output_dir else ep
        if process_episode(
            ep,
            out_dir,
            args.base_filename,
            args.start,
            args.end,
            args.joint_fps,
            in_place,
            args.dry_run,
        ):
            ok += 1
        else:
            print(f"  Failed: {ep.name}", file=sys.stderr)

    print(f"Done: {ok}/{len(episodes)} episodes.")
    if ok < len(episodes) and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
