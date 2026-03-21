#!/usr/bin/env python3

import argparse
import re
import shutil
import zipfile
from pathlib import Path
from typing import List


SEQ_PATTERN = re.compile(r"(\d+)")


def parse_split_file(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    seq_ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SEQ_PATTERN.search(line)
        if match:
            seq_ids.append(int(match.group(1)))
    if not seq_ids:
        raise ValueError(f"No sequence IDs found in {path}")
    return seq_ids


def ensure_unzipped(seq_id: int, root: Path) -> Path:
    seq_dir = root / f"seq-{seq_id:02d}"
    zip_path = root / f"seq-{seq_id:02d}.zip"
    if seq_dir.exists():
        return seq_dir
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing sequence folder and zip: {seq_dir} / {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    if not seq_dir.exists():
        raise FileNotFoundError(f"Zip extracted but sequence folder not found: {seq_dir}")
    return seq_dir


def copy_sequence(seq_dir: Path, target_root: Path) -> None:
    rgb_dir = target_root / "rgb"
    pose_dir = target_root / "poses"
    depth_dir = target_root / "depth"
    calib_dir = target_root / "calibration"
    for directory in (rgb_dir, pose_dir, depth_dir, calib_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(seq_dir.glob("frame-*.color.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB frames found in {seq_dir}")

    for rgb_file in rgb_files:
        stem = rgb_file.name.replace(".color.png", "")
        pose_file = seq_dir / f"{stem}.pose.txt"
        depth_file = seq_dir / f"{stem}.depth.png"
        if not pose_file.exists():
            raise FileNotFoundError(f"Missing pose file for {rgb_file}: {pose_file}")
        target_prefix = f"{seq_dir.name}-{stem}"
        shutil.copy2(rgb_file, rgb_dir / f"{target_prefix}.color.png")
        shutil.copy2(pose_file, pose_dir / f"{target_prefix}.pose.txt")
        if depth_file.exists():
            shutil.copy2(depth_file, depth_dir / f"{target_prefix}.depth.png")
        (calib_dir / f"{target_prefix}.txt").write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare 7Scenes chess dataset into train/test folders for SS-DVR."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Path to 7scenes/chess directory (default: current working directory).",
    )
    args = parser.parse_args()
    root = args.root

    train_split = parse_split_file(root / "TrainSplit.txt")
    test_split = parse_split_file(root / "TestSplit.txt")

    train_root = root / "train"
    test_root = root / "test"
    for directory in (train_root, test_root):
        directory.mkdir(parents=True, exist_ok=True)

    for seq_id in train_split:
        seq_dir = ensure_unzipped(seq_id, root)
        copy_sequence(seq_dir, train_root)

    for seq_id in test_split:
        seq_dir = ensure_unzipped(seq_id, root)
        copy_sequence(seq_dir, test_root)


if __name__ == "__main__":
    main()
