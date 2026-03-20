from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis import (
    build_match_summary,
    build_point_analysis,
    calibrate_court_from_clip,
    create_court_heatmap,
    detect_events_for_clip,
    map_positions_for_metrics,
    track_bottom_player,
    write_analysis_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 scaffold for post-chunk badminton game analysis.")
    parser.add_argument("--chunks-dir", required=True, type=Path, help="Directory containing point_XX clip folders.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis outputs. Defaults to <chunks-dir>/analysis.",
    )
    parser.add_argument("--debug", action="store_true", help="Include extra notes for development/debugging.")
    return parser.parse_args()


def load_point_metadata(chunks_dir: Path) -> dict[str, dict[str, float]]:
    segments_path = chunks_dir / "segments.json"
    if not segments_path.exists():
        return {}

    payload = json.loads(segments_path.read_text(encoding="utf-8"))
    metadata: dict[str, dict[str, float]] = {}
    for point in payload.get("points", []):
        name = point.get("point")
        if isinstance(name, str):
            metadata[name] = point
    return metadata


def find_point_clips(chunks_dir: Path) -> list[Path]:
    return sorted(chunks_dir.glob("point_*/point_*.mp4"))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.chunks_dir / "analysis"

    point_clips = find_point_clips(args.chunks_dir)
    if not point_clips:
        raise FileNotFoundError(f"No rally clips found in {args.chunks_dir}")

    point_metadata = load_point_metadata(args.chunks_dir)
    point_results = []
    all_trajectories = []

    for clip_path in point_clips:
        point_name = clip_path.stem
        calibration = calibrate_court_from_clip(clip_path, debug=args.debug)
        tracking = track_bottom_player(clip_path, debug=args.debug)
        events = detect_events_for_clip(tracking, clip_path=clip_path, debug=args.debug)
        result = build_point_analysis(
            point_name=point_name,
            clip_path=clip_path,
            point_metadata=point_metadata.get(point_name),
            calibration=calibration,
            tracking=tracking,
            events=events,
        )
        all_trajectories.append(map_positions_for_metrics(tracking.positions_norm, calibration))
        point_results.append(result)

    match_summary = build_match_summary(point_results)
    write_analysis_outputs(output_dir, point_results, match_summary)

    heatmap_path = output_dir / "court_heatmap.png"
    if create_court_heatmap(all_trajectories, heatmap_path):
        match_summary.heatmap_image = str(heatmap_path)
        write_analysis_outputs(output_dir, point_results, match_summary)

    print(f"Analyzed rally clips: {len(point_results)}")
    print(f"Output: {output_dir}")
    if match_summary.analysis_warnings:
        print("Warnings:")
        for warning in match_summary.analysis_warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()