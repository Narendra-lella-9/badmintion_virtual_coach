from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chunk_points import (  # noqa: E402
    DetectorConfig,
    decode_segments,
    decode_segments_from_smooth,
    moving_average,
)


class ChunkPointsOptimizationTests(unittest.TestCase):
    def test_decode_from_smooth_matches_decode_segments(self) -> None:
        rng = np.random.default_rng(7)
        activity = rng.random(1200, dtype=np.float32)
        activity[200:320] += 0.5
        activity[700:860] += 0.45
        activity = np.clip(activity, 0.0, 1.0)

        config = DetectorConfig(
            smooth_window_frames=9,
            start_confirm_sec=0.5,
            end_confirm_sec=0.9,
            min_rally_sec=1.8,
            min_gap_sec=0.8,
            threshold=0.52,
        )
        fps = 30.0

        baseline = decode_segments(activity, fps, config)
        smooth = moving_average(activity, config.smooth_window_frames)
        optimized = decode_segments_from_smooth(smooth, fps, config)

        self.assertEqual(
            [(segment.start_frame, segment.end_frame) for segment in baseline],
            [(segment.start_frame, segment.end_frame) for segment in optimized],
        )

    def test_decode_from_smooth_handles_empty_signal(self) -> None:
        config = DetectorConfig()
        empty = np.array([], dtype=np.float32)
        segments = decode_segments_from_smooth(empty, 30.0, config)
        self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()
