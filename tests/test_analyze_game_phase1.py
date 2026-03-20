from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZE_GAME = REPO_ROOT / "src" / "analyze_game.py"


def write_test_video(video_path: Path, frame_count: int = 30, fps: float = 30.0) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 64),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create synthetic video: {video_path}")

    for frame_index in range(frame_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        top = 32 + min(frame_index, 15)
        left = 10 + min(frame_index, 20)
        frame[top : top + 10, left : left + 10] = (255, 255, 255)
        writer.write(frame)

    writer.release()


class AnalyzeGamePhase1Tests(unittest.TestCase):
    def test_cli_generates_expected_outputs_with_default_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            chunks_dir = Path(temp_dir_str) / "chunks"
            write_test_video(chunks_dir / "point_01" / "point_01.mp4", frame_count=45)
            write_test_video(chunks_dir / "point_02" / "point_02.mp4", frame_count=60)

            segments_payload = {
                "points": [
                    {"point": "point_01", "start_sec": 1.0, "end_sec": 2.5},
                    {"point": "point_02", "start_sec": 3.0, "end_sec": 5.0},
                ]
            }
            (chunks_dir / "segments.json").write_text(json.dumps(segments_payload), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(ANALYZE_GAME), "--chunks-dir", str(chunks_dir)],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Analyzed rally clips: 2", result.stdout)

            analysis_dir = chunks_dir / "analysis"
            per_point_json = analysis_dir / "analysis_per_point.json"
            summary_json = analysis_dir / "analysis_match_summary.json"
            per_point_csv = analysis_dir / "analysis_per_point.csv"

            self.assertTrue(per_point_json.exists())
            self.assertTrue(summary_json.exists())
            self.assertTrue(per_point_csv.exists())

            points = json.loads(per_point_json.read_text(encoding="utf-8"))
            self.assertEqual(len(points), 2)
            self.assertEqual(points[0]["point_name"], "point_01")
            self.assertEqual(points[0]["start_sec"], 1.0)
            self.assertEqual(points[1]["point_name"], "point_02")
            self.assertEqual(points[1]["end_sec"], 5.0)
            self.assertIn("image_normalized_metrics", points[0]["quality_flags"])
            self.assertIsNotNone(points[0]["shots_est"])
            self.assertIsNotNone(points[0]["area_covered_norm"])
            self.assertIsNotNone(points[0]["path_length_norm"])
            self.assertIsNotNone(points[0]["speed_avg_norm_per_sec"])
            self.assertIsNotNone(points[0]["speed_max_norm_per_sec"])

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["total_rallies"], 2)
            self.assertEqual(summary["analyzed_rallies"], 2)
            self.assertTrue(summary["analysis_warnings"])
            self.assertIsNotNone(summary["avg_shots_per_rally"])
            self.assertIsNotNone(summary["avg_area_covered_norm"])
            self.assertIsNotNone(summary["avg_speed_norm_per_sec"])

            with per_point_csv.open("r", encoding="utf-8", newline="") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["point_name"], "point_01")

    def test_cli_supports_custom_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            chunks_dir = temp_dir / "chunks"
            custom_output_dir = temp_dir / "custom_analysis"
            write_test_video(chunks_dir / "point_01" / "point_01.mp4", frame_count=30)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZE_GAME),
                    "--chunks-dir",
                    str(chunks_dir),
                    "--output-dir",
                    str(custom_output_dir),
                    "--debug",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn(str(custom_output_dir), result.stdout)
            self.assertTrue((custom_output_dir / "analysis_per_point.json").exists())
            self.assertTrue((custom_output_dir / "analysis_match_summary.json").exists())
            self.assertTrue((custom_output_dir / "analysis_per_point.csv").exists())


if __name__ == "__main__":
    unittest.main()