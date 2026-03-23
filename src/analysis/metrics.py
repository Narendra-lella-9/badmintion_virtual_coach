from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .court_calibration import CourtCalibrationResult
from .event_detection import EventDetectionResult
from .io_schema import MatchAnalysisSummary, PointAnalysisResult
from .player_tracking import PlayerTrackingResult

# Standard badminton court dimensions (doubles).
COURT_LENGTH_M: float = 13.4  # baseline to baseline
COURT_WIDTH_M: float = 6.1    # doubles side-line to side-line


def extract_video_timing(clip_path: Path) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return 0.0, 0, 0.0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration_sec = frame_count / fps if fps > 0 else 0.0
    return fps, frame_count, duration_sec


def _valid_positions(positions: list[tuple[float, float] | None]) -> list[tuple[float, float]]:
    return [position for position in positions if position is not None]


def _map_positions_with_homography(
    positions: list[tuple[float, float] | None],
    calibration: CourtCalibrationResult,
) -> list[tuple[float, float] | None]:
    if calibration.homography_img_to_norm is None:
        return positions
    if calibration.frame_width <= 0 or calibration.frame_height <= 0:
        return positions

    mapped: list[tuple[float, float] | None] = []
    H = calibration.homography_img_to_norm
    for position in positions:
        if position is None:
            mapped.append(None)
            continue
        x_px = float(position[0] * calibration.frame_width)
        y_px = float(position[1] * calibration.frame_height)
        src = np.array([[[x_px, y_px]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        x_norm = float(dst[0, 0, 0])
        y_norm = float(dst[0, 0, 1])
        mapped.append((x_norm, y_norm))

    return mapped


def map_positions_for_metrics(
    positions: list[tuple[float, float] | None],
    calibration: CourtCalibrationResult,
) -> list[tuple[float, float] | None]:
    return _map_positions_with_homography(positions, calibration)


def _compute_path_length(positions: list[tuple[float, float] | None]) -> float | None:
    path_length = 0.0
    prev = None
    used = False
    for position in positions:
        if position is None:
            prev = None
            continue
        if prev is not None:
            dx = position[0] - prev[0]
            dy = position[1] - prev[1]
            path_length += float(np.sqrt(dx * dx + dy * dy))
            used = True
        prev = position
    return path_length if used else None


def _compute_convex_hull_area(positions: list[tuple[float, float] | None]) -> float | None:
    valid = _valid_positions(positions)
    if len(valid) < 3:
        return None
    points = np.array(valid, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(points)
    return float(cv2.contourArea(hull))


def _compute_speed_metrics(
    positions: list[tuple[float, float] | None],
    fps: float,
) -> tuple[float | None, float | None]:
    if fps <= 0:
        return None, None

    speeds = []
    prev = None
    for position in positions:
        if position is None:
            prev = None
            continue
        if prev is not None:
            dx = position[0] - prev[0]
            dy = position[1] - prev[1]
            speeds.append(float(np.sqrt(dx * dx + dy * dy) * fps))
        prev = position

    if not speeds:
        return None, None
    return float(sum(speeds) / len(speeds)), float(max(speeds))


def _compute_speed_metrics_mps(
    positions: list[tuple[float, float] | None],
    fps: float,
    court_length_m: float = COURT_LENGTH_M,
    court_width_m: float = COURT_WIDTH_M,
) -> tuple[float | None, float | None]:
    """Convert normalized-position speed to real-world m/s using court dimensions."""
    if fps <= 0:
        return None, None

    speeds_mps = []
    prev = None
    for position in positions:
        if position is None:
            prev = None
            continue
        if prev is not None:
            dx_m = (position[0] - prev[0]) * court_width_m
            dy_m = (position[1] - prev[1]) * court_length_m
            speeds_mps.append(float(np.sqrt(dx_m * dx_m + dy_m * dy_m) * fps))
        prev = position

    if not speeds_mps:
        return None, None
    return float(sum(speeds_mps) / len(speeds_mps)), float(max(speeds_mps))


def create_court_heatmap(
    trajectories: list[list[tuple[float, float] | None]],
    output_path: Path,
    size: int = 640,
) -> bool:
    heat = np.zeros((size, size), dtype=np.float32)
    points_used = 0

    for trajectory in trajectories:
        for position in trajectory:
            if position is None:
                continue
            x = int(round(float(position[0]) * (size - 1)))
            y = int(round(float(position[1]) * (size - 1)))
            if 0 <= x < size and 0 <= y < size:
                heat[y, x] += 1.0
                points_used += 1

    if points_used == 0:
        return False

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=10.0, sigmaY=10.0)
    max_value = float(np.max(heat))
    if max_value <= 0.0:
        return False

    heat_norm = np.clip((heat / max_value) * 255.0, 0.0, 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

    # Draw an approximate badminton court layout in normalized court space.
    # Court length and width are used only for line placement ratios.
    court_length_m = 13.4
    singles_width_ratio = 5.18 / 6.1
    short_service_ratio = 1.98 / court_length_m
    long_service_doubles_ratio = (court_length_m - 0.76) / court_length_m

    margin = 14
    top = margin
    bottom = size - margin
    left = margin
    right = size - margin
    center_x = (left + right) // 2
    net_y = (top + bottom) // 2

    half_extra = (1.0 - singles_width_ratio) * 0.5
    singles_left = int(round(left + (right - left) * half_extra))
    singles_right = int(round(right - (right - left) * half_extra))

    top_short_service_y = int(round(net_y - (bottom - top) * short_service_ratio))
    bottom_short_service_y = int(round(net_y + (bottom - top) * short_service_ratio))
    top_long_service_y = int(round(top + (bottom - top) * (1.0 - long_service_doubles_ratio)))
    bottom_long_service_y = int(round(top + (bottom - top) * long_service_doubles_ratio))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    line_color = (185, 185, 200)
    accent_color = (225, 225, 240)

    # Outer court boundary.
    cv2.rectangle(canvas, (left, top), (right, bottom), accent_color, 2)

    # Singles side lines.
    cv2.line(canvas, (singles_left, top), (singles_left, bottom), line_color, 1)
    cv2.line(canvas, (singles_right, top), (singles_right, bottom), line_color, 1)

    # Net line.
    cv2.line(canvas, (left, net_y), (right, net_y), accent_color, 2)

    # Short service lines on both halves.
    cv2.line(canvas, (singles_left, top_short_service_y), (singles_right, top_short_service_y), line_color, 1)
    cv2.line(canvas, (singles_left, bottom_short_service_y), (singles_right, bottom_short_service_y), line_color, 1)

    # Doubles long service guide lines near the back, useful for reading coverage depth.
    cv2.line(canvas, (left, top_long_service_y), (right, top_long_service_y), (110, 110, 140), 1)
    cv2.line(canvas, (left, bottom_long_service_y), (right, bottom_long_service_y), (110, 110, 140), 1)

    # Center service lines split left/right service boxes.
    cv2.line(canvas, (center_x, top), (center_x, top_short_service_y), line_color, 1)
    cv2.line(canvas, (center_x, bottom_short_service_y), (center_x, bottom), line_color, 1)

    overlay = cv2.addWeighted(canvas, 0.35, heat_color, 0.65, 0.0)
    cv2.putText(
        overlay,
        "Court Coverage Heatmap",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), overlay))


def build_point_analysis(
    point_name: str,
    clip_path: Path,
    point_metadata: dict[str, float] | None,
    calibration: CourtCalibrationResult,
    tracking: PlayerTrackingResult,
    events: EventDetectionResult,
) -> PointAnalysisResult:
    fps, frame_count, duration_sec = extract_video_timing(clip_path)
    metric_positions = map_positions_for_metrics(tracking.positions_norm, calibration)
    area_covered_norm = _compute_convex_hull_area(metric_positions)
    path_length_norm = _compute_path_length(metric_positions)
    speed_avg_norm_per_sec, speed_max_norm_per_sec = _compute_speed_metrics(metric_positions, fps)

    if calibration.coordinate_system == "court_normalized":
        speed_avg_mps, speed_max_mps = _compute_speed_metrics_mps(metric_positions, fps)
    else:
        speed_avg_mps, speed_max_mps = None, None

    quality_flags = []
    if calibration.coordinate_system != "court_normalized":
        quality_flags.append("image_normalized_metrics")
    if tracking.status != "ready":
        quality_flags.append("player_tracking_low_confidence")
    if events.status != "ready":
        quality_flags.append("event_detection_pending")

    if area_covered_norm is None or path_length_norm is None or speed_avg_norm_per_sec is None:
        quality_flags.append("trajectory_metrics_unavailable")

    notes = []
    notes.extend(calibration.notes)
    notes.extend(tracking.notes)
    notes.extend(events.notes)

    metric_confidence = min(1.0, 0.55 * tracking.tracking_confidence + 0.45 * calibration.confidence)

    return PointAnalysisResult(
        point_name=point_name,
        clip_path=str(clip_path),
        start_sec=point_metadata.get("start_sec") if point_metadata else None,
        end_sec=point_metadata.get("end_sec") if point_metadata else None,
        duration_sec=duration_sec,
        fps=fps,
        frame_count=frame_count,
        shots_est=events.shot_count_estimate,
        area_covered_norm=area_covered_norm,
        path_length_norm=path_length_norm,
        speed_avg_norm_per_sec=speed_avg_norm_per_sec,
        speed_max_norm_per_sec=speed_max_norm_per_sec,
        smash_count_est=events.smash_count_estimate,
        metric_confidence=metric_confidence,
        quality_flags=quality_flags,
        court_calibration_status=calibration.status,
        tracking_status=tracking.status,
        notes=notes,
        speed_avg_mps=speed_avg_mps,
        speed_max_mps=speed_max_mps,
    )


def _average_numeric(values: list[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def build_match_summary(point_results: list[PointAnalysisResult]) -> MatchAnalysisSummary:
    warnings = []
    if any("image_normalized_metrics" in result.quality_flags for result in point_results):
        warnings.append("Some rallies use image-normalized fallback metrics because homography was unavailable.")
    if any(result.shots_est is None for result in point_results):
        warnings.append("Shot counting unavailable for some rallies due to low tracking quality.")
    if any(result.smash_count_est is None for result in point_results):
        warnings.append("Smash count unavailable for some low-quality rallies.")

    speed_max_values = [result.speed_max_norm_per_sec for result in point_results if result.speed_max_norm_per_sec is not None]
    speed_max_mps_values = [result.speed_max_mps for result in point_results if result.speed_max_mps is not None]

    return MatchAnalysisSummary(
        total_rallies=len(point_results),
        analyzed_rallies=len(point_results),
        avg_shots_per_rally=_average_numeric([result.shots_est for result in point_results]),
        avg_area_covered_norm=_average_numeric([result.area_covered_norm for result in point_results]),
        avg_speed_norm_per_sec=_average_numeric([result.speed_avg_norm_per_sec for result in point_results]),
        max_speed_norm_per_sec=max(speed_max_values) if speed_max_values else None,
        avg_smashes_per_rally=_average_numeric([result.smash_count_est for result in point_results]),
        total_smashes_est=(
            int(sum(result.smash_count_est for result in point_results if result.smash_count_est is not None))
            if any(result.smash_count_est is not None for result in point_results)
            else None
        ),
        avg_speed_mps=_average_numeric([result.speed_avg_mps for result in point_results]),
        max_speed_mps=max(speed_max_mps_values) if speed_max_mps_values else None,
        analysis_warnings=warnings,
    )
