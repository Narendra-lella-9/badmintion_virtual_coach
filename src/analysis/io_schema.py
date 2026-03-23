from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PointAnalysisResult:
    point_name: str
    clip_path: str
    start_sec: float | None
    end_sec: float | None
    duration_sec: float
    fps: float
    frame_count: int
    shots_est: int | None
    area_covered_norm: float | None
    path_length_norm: float | None
    speed_avg_norm_per_sec: float | None
    speed_max_norm_per_sec: float | None
    smash_count_est: int | None
    metric_confidence: float
    quality_flags: list[str] = field(default_factory=list)
    court_calibration_status: str = "pending"
    tracking_status: str = "pending"
    notes: list[str] = field(default_factory=list)
    speed_avg_mps: float | None = None
    speed_max_mps: float | None = None


@dataclass
class MatchAnalysisSummary:
    total_rallies: int
    analyzed_rallies: int
    avg_shots_per_rally: float | None
    avg_area_covered_norm: float | None
    avg_speed_norm_per_sec: float | None
    max_speed_norm_per_sec: float | None
    avg_smashes_per_rally: float | None
    total_smashes_est: int | None
    heatmap_image: str | None = None
    avg_speed_mps: float | None = None
    max_speed_mps: float | None = None
    analysis_warnings: list[str] = field(default_factory=list)


def write_analysis_outputs(
    output_dir: Path,
    point_results: list[PointAnalysisResult],
    match_summary: MatchAnalysisSummary,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "analysis_per_point.json").open("w", encoding="utf-8") as file_obj:
        json.dump([asdict(result) for result in point_results], file_obj, indent=2)

    with (output_dir / "analysis_match_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(asdict(match_summary), file_obj, indent=2)

    fieldnames = [
        "point_name",
        "clip_path",
        "start_sec",
        "end_sec",
        "duration_sec",
        "fps",
        "frame_count",
        "shots_est",
        "area_covered_norm",
        "path_length_norm",
        "speed_avg_norm_per_sec",
        "speed_max_norm_per_sec",
        "smash_count_est",
        "speed_avg_mps",
        "speed_max_mps",
        "metric_confidence",
        "court_calibration_status",
        "tracking_status",
        "quality_flags",
        "notes",
    ]
    with (output_dir / "analysis_per_point.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for result in point_results:
            row = asdict(result)
            row["quality_flags"] = "|".join(result.quality_flags)
            row["notes"] = "|".join(result.notes)
            writer.writerow(row)
