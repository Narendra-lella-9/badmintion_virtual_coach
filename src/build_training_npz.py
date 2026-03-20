from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from chunk_points import build_model_features, extract_frame_features


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def list_video_files(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    return sorted(
        file_path
        for file_path in root_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_features_for_video(video_path: Path) -> tuple[np.ndarray, int]:
    features, _, frame_count, _, _ = extract_frame_features(video_path)
    X = build_model_features(features)
    return X, frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training NPZ for rally(1)/break(0) classification from labels and clips."
    )
    parser.add_argument(
        "--items-json",
        required=False,
        default=None,
        type=Path,
        help="Optional JSON list. Each item: {video: path, points: [{start_sec,end_sec}, ...]}",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output .npz")
    parser.add_argument(
        "--rally-clips-dir",
        action="append",
        default=[],
        type=Path,
        help="Optional folder of rally clips (all frames labeled 1). Can be passed multiple times.",
    )
    parser.add_argument(
        "--break-clips-dir",
        action="append",
        default=[],
        type=Path,
        help="Optional folder of break/non-rally clips (all frames labeled 0). Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = []
    if args.items_json is not None:
        items = json.loads(args.items_json.read_text(encoding="utf-8"))

    if not items and not args.rally_clips_dir and not args.break_clips_dir:
        raise ValueError(
            "No training sources provided. Pass --items-json and/or --rally-clips-dir/--break-clips-dir."
        )

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    full_video_samples = 0
    rally_clip_samples = 0
    break_clip_samples = 0

    for item in items:
        video_path = Path(item["video"])
        points = item.get("points", [])

        features, fps, frame_count, _, _ = extract_frame_features(video_path)
        X = build_model_features(features)

        y = np.zeros(frame_count, dtype=np.int32)
        for point in points:
            start_frame = max(0, int(round(float(point["start_sec"]) * fps)))
            end_frame = min(frame_count - 1, int(round(float(point["end_sec"]) * fps)))
            if end_frame >= start_frame:
                y[start_frame : end_frame + 1] = 1

        X_all.append(X)
        y_all.append(y)
        full_video_samples += len(y)

    rally_files: list[Path] = []
    for rally_dir in args.rally_clips_dir:
        rally_files.extend(list_video_files(rally_dir))

    for rally_file in rally_files:
        X, frame_count = build_features_for_video(rally_file)
        y = np.ones(frame_count, dtype=np.int32)
        X_all.append(X)
        y_all.append(y)
        rally_clip_samples += len(y)

    break_files: list[Path] = []
    for break_dir in args.break_clips_dir:
        break_files.extend(list_video_files(break_dir))

    for break_file in break_files:
        X, frame_count = build_features_for_video(break_file)
        y = np.zeros(frame_count, dtype=np.int32)
        X_all.append(X)
        y_all.append(y)
        break_clip_samples += len(y)

    X_cat = np.concatenate(X_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X_cat, y=y_cat)
    print(f"Saved dataset: {args.output}")
    print(f"Samples: {len(y_cat)}")
    print(f"Active ratio: {float(np.mean(y_cat)):.4f}")
    print(f"Feature dimension: {X_cat.shape[1]}")
    print(f"Samples from full videos: {full_video_samples}")
    print(f"Samples from rally clips: {rally_clip_samples}")
    print(f"Samples from break clips: {break_clip_samples}")


if __name__ == "__main__":
    main()
