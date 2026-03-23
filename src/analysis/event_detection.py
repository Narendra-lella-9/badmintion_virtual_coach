from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .player_tracking import PlayerTrackingResult


@dataclass
class EventDetectionResult:
    status: str
    shot_count_estimate: int | None
    smash_count_estimate: int | None
    confidence: float
    notes: list[str] = field(default_factory=list)


def _estimate_fps(clip_path: Path | None, fallback_fps: float = 30.0) -> float:
    if clip_path is None:
        return fallback_fps
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return fallback_fps
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fallback_fps)
    cap.release()
    return fps if fps > 0 else fallback_fps


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def detect_events_for_clip(
    tracking: PlayerTrackingResult,
    clip_path: Path | None = None,
    fps: float | None = None,
    debug: bool = False,
) -> EventDetectionResult:
    use_fps = fps if fps is not None and fps > 0 else _estimate_fps(clip_path)

    speeds = np.zeros(len(tracking.positions_norm), dtype=np.float32)
    valid_speed = np.zeros(len(tracking.positions_norm), dtype=np.uint8)
    prev = None
    for index, position in enumerate(tracking.positions_norm):
        if position is None:
            prev = None
            continue
        if prev is not None:
            dx = position[0] - prev[0]
            dy = position[1] - prev[1]
            speeds[index] = float(np.sqrt(dx * dx + dy * dy) * use_fps)
            valid_speed[index] = 1
        prev = position

    if int(np.sum(valid_speed)) < 6:
        return EventDetectionResult(
            status="low_confidence",
            shot_count_estimate=0,
            smash_count_estimate=None,
            confidence=0.15,
            notes=["Insufficient tracked motion samples for reliable shot counting."],
        )

    smooth_speed = _smooth(speeds, 5)
    valid_values = smooth_speed[valid_speed == 1]
    base = float(np.percentile(valid_values, 60))
    spread = float(np.std(valid_values))
    threshold = max(base + 0.55 * spread, 0.45)
    min_gap_frames = max(4, int(round(0.28 * use_fps)))

    peaks = 0
    peak_speeds_list: list[float] = []
    last_peak = -10**9
    for i in range(1, len(smooth_speed) - 1):
        if valid_speed[i] == 0:
            continue
        is_peak = smooth_speed[i] > smooth_speed[i - 1] and smooth_speed[i] >= smooth_speed[i + 1]
        strong = smooth_speed[i] >= threshold
        far_enough = (i - last_peak) >= min_gap_frames
        if is_peak and strong and far_enough:
            peaks += 1
            peak_speeds_list.append(float(smooth_speed[i]))
            last_peak = i

    # Smash estimation: peaks significantly above the average motion baseline.
    smash_threshold = max(base + 2.5 * spread, threshold * 1.8, 0.80)
    smash_count = sum(1 for s in peak_speeds_list if s >= smash_threshold)

    shot_count = max(1, peaks) if int(np.sum(valid_speed)) >= 10 else peaks
    confidence = min(1.0, 0.35 + 0.45 * tracking.visible_ratio + 0.20 * min(1.0, peaks / 8.0))

    notes = [
        f"Shot-count from motion peaks above dynamic threshold ({threshold:.3f}).",
        f"Smash estimate from high-speed peaks above threshold ({smash_threshold:.3f}).",
    ]
    if debug:
        notes.append(
            f"fps={use_fps:.2f}, min_gap_frames={min_gap_frames}, "
            f"detected_peaks={peaks}, smash_count={smash_count}."
        )

    return EventDetectionResult(
        status="ready",
        shot_count_estimate=int(shot_count),
        smash_count_estimate=int(smash_count),
        confidence=float(confidence),
        notes=notes,
    )
