from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CourtCalibrationResult:
    status: str
    method: str
    confidence: float
    frame_width: int
    frame_height: int
    coordinate_system: str
    homography_img_to_norm: Optional[np.ndarray] = None
    court_corners_img: Optional[list[tuple[float, float]]] = None
    notes: list[str] = field(default_factory=list)


def _line_from_segment(segment: np.ndarray) -> tuple[float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in segment]
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    return a, b, c


def _intersection(line1: tuple[float, float, float], line2: tuple[float, float, float]) -> Optional[tuple[float, float]]:
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return float(x), float(y)


def calibrate_court_from_clip(clip_path: Path, debug: bool = False) -> CourtCalibrationResult:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return CourtCalibrationResult(
            status="unavailable",
            method="phase1_stub",
            confidence=0.0,
            frame_width=0,
            frame_height=0,
            coordinate_system="unknown",
            homography_img_to_norm=None,
            court_corners_img=None,
            notes=[f"Unable to open clip: {clip_path.name}"],
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return CourtCalibrationResult(
            status="unavailable",
            method="hough_lines_v1",
            confidence=0.0,
            frame_width=width,
            frame_height=height,
            coordinate_system="image_normalized",
            homography_img_to_norm=None,
            court_corners_img=None,
            notes=[f"Unable to read a frame from clip: {clip_path.name}"],
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=60,
        minLineLength=max(20, width // 8),
        maxLineGap=20,
    )

    horizontal_segments: list[np.ndarray] = []
    vertical_segments: list[np.ndarray] = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in line]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= 15 or angle >= 165:
                horizontal_segments.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            elif 75 <= angle <= 105:
                vertical_segments.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    horizontal_lines = len(horizontal_segments)
    vertical_lines = len(vertical_segments)

    homography = None
    corners_img = None
    if horizontal_lines >= 2 and vertical_lines >= 2:
        horizontal_sorted = sorted(horizontal_segments, key=lambda seg: float((seg[1] + seg[3]) * 0.5))
        vertical_sorted = sorted(vertical_segments, key=lambda seg: float((seg[0] + seg[2]) * 0.5))

        top_line = _line_from_segment(horizontal_sorted[0])
        bottom_line = _line_from_segment(horizontal_sorted[-1])
        left_line = _line_from_segment(vertical_sorted[0])
        right_line = _line_from_segment(vertical_sorted[-1])

        top_left = _intersection(top_line, left_line)
        top_right = _intersection(top_line, right_line)
        bottom_left = _intersection(bottom_line, left_line)
        bottom_right = _intersection(bottom_line, right_line)

        if all(point is not None for point in [top_left, top_right, bottom_left, bottom_right]):
            corners_img = [top_left, top_right, bottom_left, bottom_right]  # type: ignore[list-item]
            margin = 0.15
            x_ok = all(-margin * width <= point[0] <= (1.0 + margin) * width for point in corners_img)
            y_ok = all(-margin * height <= point[1] <= (1.0 + margin) * height for point in corners_img)
            if x_ok and y_ok:
                src = np.array(corners_img, dtype=np.float32)
                dst = np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                    ],
                    dtype=np.float32,
                )
                homography = cv2.getPerspectiveTransform(src, dst)

    line_score = min(1.0, (horizontal_lines + vertical_lines) / 12.0)
    confidence = 0.15 + 0.55 * line_score if (horizontal_lines + vertical_lines) > 0 else 0.0
    if homography is not None:
        confidence = min(1.0, confidence + 0.20)
        status = "ready"
        coordinate_system = "court_normalized"
    elif horizontal_lines >= 2 and vertical_lines >= 2:
        status = "partial"
        coordinate_system = "image_normalized"
    else:
        status = "fallback"
        coordinate_system = "image_normalized"

    notes = [
        f"Detected court-like lines: horizontal={horizontal_lines}, vertical={vertical_lines}.",
        "Phase 2 uses image-normalized fallback when homography is unavailable.",
    ]
    if homography is not None:
        notes.append("Homography estimated from extreme horizontal/vertical court-line intersections.")
    if debug:
        notes.append("Next calibration step will infer court corners and homography from line topology.")

    return CourtCalibrationResult(
        status=status,
        method="hough_lines_v1",
        confidence=float(confidence),
        frame_width=width,
        frame_height=height,
        coordinate_system=coordinate_system,
        homography_img_to_norm=homography,
        court_corners_img=corners_img,
        notes=notes,
    )
