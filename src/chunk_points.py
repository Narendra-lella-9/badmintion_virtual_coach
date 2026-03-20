from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Segment:
    start_frame: int
    end_frame: int


@dataclass
class LabeledChunk:
    name: str
    label: str
    start_frame: int
    end_frame: int


@dataclass
class DetectorConfig:
    smooth_window_frames: int = 9
    start_confirm_sec: float = 0.6
    end_confirm_sec: float = 1.0
    min_rally_sec: float = 2.0
    min_gap_sec: float = 1.2
    threshold: float = 0.5
    max_rally_sec: float = 45.0


@dataclass
class DetectionDiagnostics:
    preset: str
    used_threshold: float
    score: float
    count: int
    active_ratio: float
    avg_duration_sec: float


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def extract_frame_features(video_path: Path) -> tuple[dict[str, np.ndarray], float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_top = height // 2
    frame_area = width * (height - roi_top)

    prev_gray = None
    prev_center = None

    energies: List[float] = []
    speeds: List[float] = []
    areas: List[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi = frame[roi_top:height, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            energies.append(0.0)
            speeds.append(0.0)
            areas.append(0.0)
            prev_gray = gray
            prev_center = None
            continue

        diff = cv2.absdiff(gray, prev_gray)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=1)

        motion_energy = float(np.mean(blur)) / 255.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0.0
        center = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                x, y, w, h = cv2.boundingRect(contour)
                largest_area = area
                center = (x + w * 0.5, y + h * 0.5)

        area_ratio = largest_area / frame_area if frame_area > 0 else 0.0

        if center is not None and prev_center is not None:
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]
            speed = float(np.sqrt(dx * dx + dy * dy)) / max(width, height // 2)
        else:
            speed = 0.0

        energies.append(motion_energy)
        speeds.append(speed)
        areas.append(area_ratio)

        prev_gray = gray
        prev_center = center if center is not None else prev_center

    cap.release()

    features = {
        "energy": np.array(energies, dtype=np.float32),
        "speed": np.array(speeds, dtype=np.float32),
        "area": np.array(areas, dtype=np.float32),
    }
    return features, float(fps), frame_count, width, height


def normalize_signal(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = np.percentile(values, 5)
    hi = np.percentile(values, 95)
    if hi - lo < 1e-6:
        return np.zeros_like(values)
    scaled = (values - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.zeros_like(values)
    mean = moving_average(values, window)
    mean_sq = moving_average(values * values, window)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance)


def build_model_features(features: dict[str, np.ndarray]) -> np.ndarray:
    energy = normalize_signal(features["energy"])
    speed = normalize_signal(features["speed"])
    area = normalize_signal(features["area"])

    # Add short/medium temporal context and local variance so breaks with incidental
    # motion are less likely to be confused with sustained rally motion.
    e_mean_5 = moving_average(energy, 5)
    e_mean_15 = moving_average(energy, 15)
    s_mean_5 = moving_average(speed, 5)
    s_mean_15 = moving_average(speed, 15)
    a_mean_5 = moving_average(area, 5)
    a_mean_15 = moving_average(area, 15)

    e_std_9 = rolling_std(energy, 9)
    s_std_9 = rolling_std(speed, 9)
    a_std_9 = rolling_std(area, 9)

    e_delta = np.abs(np.diff(energy, prepend=energy[0] if len(energy) else 0.0))
    s_delta = np.abs(np.diff(speed, prepend=speed[0] if len(speed) else 0.0))
    a_delta = np.abs(np.diff(area, prepend=area[0] if len(area) else 0.0))

    return np.column_stack(
        [
            energy,
            speed,
            area,
            e_mean_5,
            e_mean_15,
            s_mean_5,
            s_mean_15,
            a_mean_5,
            a_mean_15,
            e_std_9,
            s_std_9,
            a_std_9,
            e_delta,
            s_delta,
            a_delta,
        ]
    )


def build_activity_probability(
    features: dict[str, np.ndarray],
    model: Optional[object] = None,
) -> np.ndarray:
    energy = normalize_signal(features["energy"])
    speed = normalize_signal(features["speed"])
    area = normalize_signal(features["area"])

    stacked = build_model_features(features)

    if model is not None:
        if hasattr(model, "n_features_in_"):
            expected = int(getattr(model, "n_features_in_"))
            if expected < stacked.shape[1]:
                stacked = stacked[:, :expected]
            elif expected > stacked.shape[1]:
                pad = np.zeros((stacked.shape[0], expected - stacked.shape[1]), dtype=stacked.dtype)
                stacked = np.concatenate([stacked, pad], axis=1)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(stacked)[:, 1]
            return np.asarray(proba, dtype=np.float32)
        preds = model.predict(stacked)
        return np.asarray(preds, dtype=np.float32)

    combined = 0.5 * energy + 0.35 * speed + 0.15 * area
    return np.asarray(combined, dtype=np.float32)


def decode_segments(
    activity_prob: np.ndarray,
    fps: float,
    config: DetectorConfig,
) -> List[Segment]:
    if len(activity_prob) == 0:
        return []

    smooth = moving_average(activity_prob, max(1, config.smooth_window_frames))
    active = smooth >= config.threshold

    start_confirm = max(1, int(round(config.start_confirm_sec * fps)))
    end_confirm = max(1, int(round(config.end_confirm_sec * fps)))
    min_rally = max(1, int(round(config.min_rally_sec * fps)))
    min_gap = max(1, int(round(config.min_gap_sec * fps)))

    segments: List[Segment] = []
    in_rally = False
    start_frame = 0
    active_run = 0
    idle_run = 0
    last_end = -10**9

    for index, is_active in enumerate(active):
        if is_active:
            active_run += 1
            idle_run = 0
        else:
            idle_run += 1
            active_run = 0

        if not in_rally:
            enough_gap = index - last_end >= min_gap
            if enough_gap and active_run >= start_confirm:
                start_frame = index - start_confirm + 1
                in_rally = True
                active_run = 0
                idle_run = 0
        else:
            if idle_run >= end_confirm:
                end_frame = index - end_confirm
                if end_frame - start_frame + 1 >= min_rally:
                    segments.append(Segment(start_frame=start_frame, end_frame=end_frame))
                    last_end = end_frame
                in_rally = False
                active_run = 0
                idle_run = 0

    if in_rally:
        end_frame = len(active_prob := activity_prob) - 1
        if end_frame - start_frame + 1 >= min_rally:
            segments.append(Segment(start_frame=start_frame, end_frame=end_frame))

    merged: List[Segment] = []
    merge_gap = int(round(0.8 * fps))
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        prev = merged[-1]
        if segment.start_frame - prev.end_frame <= merge_gap:
            merged[-1] = Segment(start_frame=prev.start_frame, end_frame=segment.end_frame)
        else:
            merged.append(segment)

    return merged


def _segments_activity_stats(segments: List[Segment], total_frames: int) -> tuple[int, float, float]:
    if total_frames <= 0:
        return 0, 0.0, 0.0
    if not segments:
        return 0, 0.0, 0.0

    durations = [max(1, segment.end_frame - segment.start_frame + 1) for segment in segments]
    active_frames = float(sum(durations))
    active_ratio = active_frames / float(total_frames)
    avg_duration = float(np.mean(durations))
    return len(segments), active_ratio, avg_duration


def _quality_score(
    segments: List[Segment],
    fps: float,
    total_frames: int,
) -> tuple[float, int, float, float]:
    count, active_ratio, avg_duration_frames = _segments_activity_stats(segments, total_frames)
    if count == 0:
        return -1e9, 0, 0.0, 0.0

    avg_duration_sec = avg_duration_frames / max(1e-6, fps)
    duration_sec = total_frames / max(1e-6, fps)

    score = 0.0
    score += min(count, 60) * 0.55

    if avg_duration_sec < 1.6:
        score -= (1.6 - avg_duration_sec) * 8.0
    elif avg_duration_sec > 50.0:
        score -= (avg_duration_sec - 50.0) * 0.3

    if active_ratio < 0.08:
        score -= (0.08 - active_ratio) * 60.0
    elif active_ratio > 0.85:
        score -= (active_ratio - 0.85) * 60.0

    expected_min = max(3, int(duration_sec / 180.0))
    expected_max = max(expected_min + 2, int(duration_sec / 8.0))
    if count < expected_min:
        score -= (expected_min - count) * 1.2
    elif count > expected_max:
        score -= (count - expected_max) * 0.8

    return score, count, active_ratio, avg_duration_sec


def detect_segments_robust(
    activity_prob: np.ndarray,
    fps: float,
    config: DetectorConfig,
) -> tuple[List[Segment], float]:
    base_segments = decode_segments(activity_prob, fps, config)
    if base_segments:
        return base_segments, config.threshold

    threshold_candidates = np.linspace(0.2, 0.75, num=23)
    start_candidates = [config.start_confirm_sec, max(0.30, config.start_confirm_sec * 0.75)]
    end_candidates = [config.end_confirm_sec, max(0.50, config.end_confirm_sec * 0.70)]
    min_rally_candidates = [config.min_rally_sec, max(1.2, config.min_rally_sec * 0.65)]
    smooth_candidates = sorted({
        int(config.smooth_window_frames),
        max(3, int(config.smooth_window_frames - 2)),
        max(5, int(config.smooth_window_frames + 2)),
    })

    best_segments: List[Segment] = []
    best_threshold = config.threshold
    best_score = -1e9

    total_frames = len(activity_prob)

    for threshold in threshold_candidates:
        for start_confirm in start_candidates:
            for end_confirm in end_candidates:
                for min_rally in min_rally_candidates:
                    for smooth_window in smooth_candidates:
                        trial_config = DetectorConfig(
                            smooth_window_frames=smooth_window,
                            start_confirm_sec=start_confirm,
                            end_confirm_sec=end_confirm,
                            min_rally_sec=min_rally,
                            min_gap_sec=config.min_gap_sec,
                            threshold=float(threshold),
                        )
                        segments = decode_segments(activity_prob, fps, trial_config)
                        count, active_ratio, avg_duration = _segments_activity_stats(segments, total_frames)
                        if count == 0:
                            continue

                        avg_duration_sec = avg_duration / max(1e-6, fps)
                        count_penalty = 0.0
                        if count < 8:
                            count_penalty = (8 - count) * 0.9
                        elif count > 80:
                            count_penalty = (count - 80) * 0.5

                        duration_penalty = 0.0
                        if avg_duration_sec < 1.8:
                            duration_penalty += (1.8 - avg_duration_sec) * 2.2
                        elif avg_duration_sec > 35.0:
                            duration_penalty += (avg_duration_sec - 35.0) * 0.25

                        ratio_penalty = abs(active_ratio - 0.42) * 8.0
                        score = count * 0.35 - count_penalty - duration_penalty - ratio_penalty

                        if score > best_score:
                            best_score = score
                            best_segments = segments
                            best_threshold = float(threshold)

    if best_segments:
        return best_segments, best_threshold

    smooth = moving_average(activity_prob, max(5, config.smooth_window_frames))
    spread = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    if spread < 0.03:
        return [], best_threshold

    min_rally_frames = max(1, int(round(max(1.0, config.min_rally_sec * 0.55) * fps)))
    min_gap_frames = max(1, int(round(max(0.6, config.min_gap_sec * 0.5) * fps)))

    for q in (75, 70, 65, 60):
        thr = float(np.percentile(smooth, q))
        mask = smooth >= thr
        if not np.any(mask):
            continue

        ranges: List[Segment] = []
        start = None
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            if not value and start is not None:
                end = i - 1
                if end - start + 1 >= min_rally_frames:
                    ranges.append(Segment(start_frame=start, end_frame=end))
                start = None
        if start is not None:
            end = len(mask) - 1
            if end - start + 1 >= min_rally_frames:
                ranges.append(Segment(start_frame=start, end_frame=end))

        if not ranges:
            continue

        merged: List[Segment] = []
        for segment in ranges:
            if not merged:
                merged.append(segment)
                continue
            prev = merged[-1]
            if segment.start_frame - prev.end_frame <= min_gap_frames:
                merged[-1] = Segment(start_frame=prev.start_frame, end_frame=segment.end_frame)
            else:
                merged.append(segment)

        if merged:
            return merged, thr

    return [], best_threshold


def detect_segments_auto(
    activity_prob: np.ndarray,
    fps: float,
    config: DetectorConfig,
) -> tuple[List[Segment], DetectionDiagnostics]:
    presets = [
        (
            "balanced",
            DetectorConfig(
                smooth_window_frames=config.smooth_window_frames,
                start_confirm_sec=config.start_confirm_sec,
                end_confirm_sec=config.end_confirm_sec,
                min_rally_sec=config.min_rally_sec,
                min_gap_sec=config.min_gap_sec,
                threshold=config.threshold,
            ),
        ),
        (
            "sensitive",
            DetectorConfig(
                smooth_window_frames=max(3, config.smooth_window_frames - 2),
                start_confirm_sec=max(0.30, config.start_confirm_sec * 0.80),
                end_confirm_sec=max(0.50, config.end_confirm_sec * 0.80),
                min_rally_sec=max(1.1, config.min_rally_sec * 0.75),
                min_gap_sec=max(0.7, config.min_gap_sec * 0.80),
                threshold=max(0.20, config.threshold - 0.08),
            ),
        ),
        (
            "strict",
            DetectorConfig(
                smooth_window_frames=min(41, config.smooth_window_frames + 4),
                start_confirm_sec=min(2.5, config.start_confirm_sec * 1.15),
                end_confirm_sec=min(3.0, config.end_confirm_sec * 1.25),
                min_rally_sec=min(10.0, config.min_rally_sec * 1.20),
                min_gap_sec=min(5.0, config.min_gap_sec * 1.15),
                threshold=min(0.85, config.threshold + 0.06),
            ),
        ),
        (
            "short-rally",
            DetectorConfig(
                smooth_window_frames=max(3, config.smooth_window_frames - 2),
                start_confirm_sec=max(0.25, config.start_confirm_sec * 0.70),
                end_confirm_sec=max(0.45, config.end_confirm_sec * 0.70),
                min_rally_sec=max(0.9, config.min_rally_sec * 0.55),
                min_gap_sec=max(0.5, config.min_gap_sec * 0.65),
                threshold=max(0.18, config.threshold - 0.12),
            ),
        ),
    ]

    total_frames = len(activity_prob)
    best_segments: List[Segment] = []
    best_diag = DetectionDiagnostics(
        preset="balanced",
        used_threshold=config.threshold,
        score=-1e9,
        count=0,
        active_ratio=0.0,
        avg_duration_sec=0.0,
    )

    for preset_name, preset_config in presets:
        segments, used_threshold = detect_segments_robust(activity_prob, fps, preset_config)
        score, count, active_ratio, avg_duration_sec = _quality_score(segments, fps, total_frames)
        if score > best_diag.score:
            best_segments = segments
            best_diag = DetectionDiagnostics(
                preset=preset_name,
                used_threshold=used_threshold,
                score=score,
                count=count,
                active_ratio=active_ratio,
                avg_duration_sec=avg_duration_sec,
            )

    return best_segments, best_diag


def filter_rally_segments(
    segments: List[Segment],
    activity_prob: np.ndarray,
    fps: float,
    config: DetectorConfig,
) -> List[Segment]:
    if not segments:
        return []

    min_rally_frames = max(1, int(round(config.min_rally_sec * fps)))
    max_rally_frames = max(min_rally_frames + 1, int(round(config.max_rally_sec * fps)))

    smooth = moving_average(activity_prob, max(5, config.smooth_window_frames))
    kept: List[Segment] = []

    for segment in segments:
        duration = segment.end_frame - segment.start_frame + 1
        if duration < min_rally_frames:
            continue
        if duration > max_rally_frames:
            continue

        slice_start = max(0, segment.start_frame)
        slice_end = min(len(smooth), segment.end_frame + 1)
        if slice_end <= slice_start:
            continue

        confidence = float(np.mean(smooth[slice_start:slice_end]))
        if confidence < max(0.25, config.threshold * 0.70):
            continue

        kept.append(segment)

    return kept


def build_labeled_timeline_chunks(
    rally_segments: List[Segment],
    total_frames: int,
    fps: float,
    min_break_sec: float = 0.6,
) -> List[LabeledChunk]:
    chunks: List[LabeledChunk] = []
    if total_frames <= 0:
        return chunks

    min_break_frames = max(1, int(round(min_break_sec * fps)))

    rally_count = 0
    break_count = 0
    prev_end = -1

    for segment in rally_segments:
        break_start = prev_end + 1
        break_end = segment.start_frame - 1
        if break_end >= break_start and (break_end - break_start + 1) >= min_break_frames:
            break_count += 1
            chunks.append(
                LabeledChunk(
                    name=f"break_{break_count:02d}",
                    label="break",
                    start_frame=break_start,
                    end_frame=break_end,
                )
            )

        rally_count += 1
        chunks.append(
            LabeledChunk(
                name=f"point_{rally_count:02d}",
                label="rally",
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
            )
        )
        prev_end = segment.end_frame

    tail_start = prev_end + 1
    tail_end = total_frames - 1
    if tail_end >= tail_start and (tail_end - tail_start + 1) >= min_break_frames:
        break_count += 1
        chunks.append(
            LabeledChunk(
                name=f"break_{break_count:02d}",
                label="break",
                start_frame=tail_start,
                end_frame=tail_end,
            )
        )

    return chunks


def export_segments(
    video_path: Path,
    chunks: List[LabeledChunk],
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
    pre_pad_sec: float,
    post_pad_sec: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for export: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for chunk in chunks:
        chunk_folder = output_dir / chunk.name
        chunk_folder.mkdir(parents=True, exist_ok=True)
        chunk_video = chunk_folder / f"{chunk.name}.mp4"

        start_frame = max(0, chunk.start_frame - int(round(pre_pad_sec * fps)))
        end_frame = min(total_frames - 1, chunk.end_frame + int(round(post_pad_sec * fps)))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(
            str(chunk_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        current = start_frame
        while current <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            current += 1

        writer.release()

    cap.release()


def save_metadata(
    chunks: List[LabeledChunk],
    output_dir: Path,
    fps: float,
    activity_prob: np.ndarray,
    diagnostics: DetectionDiagnostics | None = None,
) -> None:
    chunk_metadata = []
    rally_points = []

    for chunk in chunks:
        entry = {
            "name": chunk.name,
            "label": chunk.label,
            "start_frame": int(chunk.start_frame),
            "end_frame": int(chunk.end_frame),
            "start_sec": float(chunk.start_frame / fps),
            "end_sec": float(chunk.end_frame / fps),
        }
        chunk_metadata.append(entry)

        if chunk.label == "rally":
            rally_points.append(
            {
                "point": chunk.name,
                "start_frame": int(chunk.start_frame),
                "end_frame": int(chunk.end_frame),
                "start_sec": float(chunk.start_frame / fps),
                "end_sec": float(chunk.end_frame / fps),
            }
        )

    payload = {
        "points": rally_points,
        "chunks": chunk_metadata,
        "total_detected_points": len(rally_points),
        "total_chunks": len(chunk_metadata),
    }

    if diagnostics is not None:
        payload["detection"] = {
            "preset": diagnostics.preset,
            "used_threshold": float(diagnostics.used_threshold),
            "quality_score": float(diagnostics.score),
            "active_ratio": float(diagnostics.active_ratio),
            "avg_point_duration_sec": float(diagnostics.avg_duration_sec),
        }

    with (output_dir / "segments.json").open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    signal_path = output_dir / "activity_signal.csv"
    np.savetxt(signal_path, activity_prob, delimiter=",", fmt="%.6f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split badminton singles video into point-wise chunks (target player in bottom half)."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for chunks")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional trained model (.joblib). If omitted, script tries src/source/point_activity_2.joblib.",
    )

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--start-confirm-sec", type=float, default=0.6)
    parser.add_argument("--end-confirm-sec", type=float, default=1.0)
    parser.add_argument("--min-rally-sec", type=float, default=2.0)
    parser.add_argument("--min-gap-sec", type=float, default=1.2)
    parser.add_argument("--max-rally-sec", type=float, default=45.0)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--pre-pad-sec", type=float, default=0.25)
    parser.add_argument("--post-pad-sec", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = None
    model_name = "heuristic"
    default_model_path = (Path(__file__).resolve().parent / "source" / "point_activity.joblib").resolve()
    model_path = args.model if args.model is not None else default_model_path

    if model_path.exists():
        import joblib

        model = joblib.load(model_path)
        model_name = str(model_path)

    features, fps, frame_count, width, height = extract_frame_features(args.video)
    activity_prob = build_activity_probability(features, model=model)

    config = DetectorConfig(
        smooth_window_frames=max(1, args.smooth_window),
        start_confirm_sec=args.start_confirm_sec,
        end_confirm_sec=args.end_confirm_sec,
        min_rally_sec=args.min_rally_sec,
        min_gap_sec=args.min_gap_sec,
        max_rally_sec=args.max_rally_sec,
        threshold=args.threshold,
    )

    segments, diagnostics = detect_segments_auto(activity_prob, fps, config)
    segments = filter_rally_segments(segments, activity_prob, fps, config)
    diagnostics.count = len(segments)

    labeled_chunks = build_labeled_timeline_chunks(segments, frame_count, fps)

    export_segments(
        video_path=args.video,
        chunks=labeled_chunks,
        output_dir=args.output_dir,
        fps=fps,
        width=width,
        height=height,
        pre_pad_sec=args.pre_pad_sec,
        post_pad_sec=args.post_pad_sec,
    )
    save_metadata(labeled_chunks, args.output_dir, fps, activity_prob, diagnostics=diagnostics)

    print(f"Frames: {frame_count}")
    print(f"FPS: {fps:.3f}")
    print(f"Detected points: {len(segments)}")
    print(f"Model: {model_name}")
    print(f"Preset: {diagnostics.preset}")
    print(f"Used threshold: {diagnostics.used_threshold:.3f}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
