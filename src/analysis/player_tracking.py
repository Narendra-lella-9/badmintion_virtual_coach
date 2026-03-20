from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PlayerTrackingResult:
    status: str
    tracking_confidence: float
    visible_ratio: float
    frame_count: int
    positions_norm: list[tuple[float, float] | None] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def track_bottom_player(clip_path: Path, debug: bool = False) -> PlayerTrackingResult:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return PlayerTrackingResult(
            status="unavailable",
            tracking_confidence=0.0,
            visible_ratio=0.0,
            frame_count=0,
            positions_norm=[],
            notes=[f"Unable to open clip: {clip_path.name}"],
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_top = height // 2
    roi_area = max(1, width * max(1, height - roi_top))

    prev_gray = None
    positions_norm: list[tuple[float, float] | None] = []
    visible_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi = frame[roi_top:height, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            positions_norm.append(None)
            prev_gray = gray
            continue

        diff = cv2.absdiff(gray, prev_gray)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(blur, 22, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_area = 0.0
        min_area = roi_area * 0.0025
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            score = area * (1.0 + ((y + h) / max(1, roi.shape[0])))
            if score > best_area:
                best_area = score
                best_box = (x, y, w, h)

        if best_box is None:
            positions_norm.append(None)
        else:
            x, y, w, h = best_box
            center_x = (x + w * 0.5) / max(1, width)
            center_y = (roi_top + y + h * 0.75) / max(1, height)
            positions_norm.append((float(center_x), float(center_y)))
            visible_count += 1

        prev_gray = gray

    cap.release()

    visible_ratio = visible_count / frame_count if frame_count > 0 else 0.0
    tracking_confidence = min(1.0, visible_ratio * 1.15)
    status = "ready" if visible_ratio >= 0.35 else "low_confidence"

    notes = [
        "Phase 2 baseline uses motion-based bottom-player centroid tracking in image-normalized coordinates.",
        f"Visible ratio: {visible_ratio:.2f}",
    ]
    if debug:
        notes.append("Next tracking step will upgrade to person detection plus temporal association.")

    return PlayerTrackingResult(
        status=status,
        tracking_confidence=float(tracking_confidence),
        visible_ratio=float(visible_ratio),
        frame_count=frame_count,
        positions_norm=positions_norm,
        notes=notes,
    )
