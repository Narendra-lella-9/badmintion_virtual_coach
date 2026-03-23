from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np

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
from chunk_points import (
    DetectorConfig,
    build_activity_probability,
    build_labeled_timeline_chunks,
    detect_segments_auto,
    export_segments,
    extract_frame_features,
    filter_rally_segments,
    save_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def _safe_stem(file_path: Path) -> str:
    stem = file_path.stem.strip().replace(" ", "_")
    return stem or "video"


def _build_output_dir(base_dir: Path, source_video: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{_safe_stem(source_video)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_browser_preview(input_video: Path, preview_dir: Path) -> Path:
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_video = preview_dir / input_video.name
    if output_video.exists():
        return output_video

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_video),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return output_video
    except Exception:
        return input_video


def _resolve_uploaded_path(upload_value: Any) -> Path | None:
    if upload_value is None:
        return None
    if isinstance(upload_value, str):
        return Path(upload_value)
    if isinstance(upload_value, dict):
        file_path = upload_value.get("path") or upload_value.get("name")
        return Path(file_path) if file_path else None
    file_name = getattr(upload_value, "name", None)
    if isinstance(file_name, str):
        return Path(file_name)
    return None


def process_video(
    video_file: Any,
    model_file: Any,
    threshold: float,
    start_confirm_sec: float,
    end_confirm_sec: float,
    min_rally_sec: float,
    min_gap_sec: float,
    smooth_window: int,
    pre_pad_sec: float,
    post_pad_sec: float,

) -> tuple[str, dict[str, Any], list[str], str, Any, str | None, dict[str, str]]:
    video_path = _resolve_uploaded_path(video_file)
    if video_path is None:
        return "Please drag/drop a video file.", {}, [], "", gr.update(choices=[], value=None), None, {}

    model = None
    model_name = "heuristic"
    model_path = _resolve_uploaded_path(model_file)
    if model_path is not None:
        import joblib

        model = joblib.load(model_path)
        model_name = model_path.name

    output_dir = _build_output_dir(OUTPUTS_ROOT, video_path)

    features, fps, frame_count, width, height = extract_frame_features(video_path)
    activity_prob = build_activity_probability(features, model=model)

    config = DetectorConfig(
        smooth_window_frames=max(1, int(smooth_window)),
        start_confirm_sec=float(start_confirm_sec),
        end_confirm_sec=float(end_confirm_sec),
        min_rally_sec=float(min_rally_sec),
        min_gap_sec=float(min_gap_sec),
        threshold=float(threshold),
    )

    segments, diagnostics = detect_segments_auto(activity_prob, fps, config)
    segments = filter_rally_segments(segments, activity_prob, fps, config)
    diagnostics.count = len(segments)

    labeled_chunks = build_labeled_timeline_chunks(segments, frame_count, fps)

    export_segments(
        video_path=video_path,
        chunks=labeled_chunks,
        output_dir=output_dir,
        fps=fps,
        width=width,
        height=height,
        pre_pad_sec=float(pre_pad_sec),
        post_pad_sec=float(post_pad_sec),
    )
    save_metadata(labeled_chunks, output_dir, fps, activity_prob, diagnostics=diagnostics)

    segments_path = output_dir / "segments.json"
    payload = json.loads(segments_path.read_text(encoding="utf-8"))

    rally_videos = sorted(str(path) for path in output_dir.glob("point_*/point_*.mp4"))
    break_videos = sorted(str(path) for path in output_dir.glob("break_*/break_*.mp4"))
    downloadable_files = rally_videos + break_videos + [str(segments_path)]

    preview_dir = output_dir / "_preview"
    preview_videos = [str(_make_browser_preview(Path(path), preview_dir)) for path in rally_videos]

    point_names = [Path(path).stem for path in rally_videos]
    video_map = {name: path for name, path in zip(point_names, preview_videos)}
    first_preview = preview_videos[0] if preview_videos else None

    if len(segments) == 0:
        status = (
            "No points detected from this run. "
            f"Frames: {frame_count} | FPS: {fps:.2f} | Preset: {diagnostics.preset} | "
            f"Threshold: {diagnostics.used_threshold:.2f} | Model: {model_name}. "
            "Try lowering Threshold and End Confirm in Advanced Settings."
        )
    else:
        status = (
            f"Done. Detected points: {len(segments)} | "
            f"Frames: {frame_count} | FPS: {fps:.2f} | Preset: {diagnostics.preset} | "
            f"Threshold: {diagnostics.used_threshold:.2f} | "
            f"Quality: {diagnostics.score:.2f} | Model: {model_name}"
        )

    return (
        status,
        payload,
        downloadable_files,
        str(output_dir),
        gr.update(choices=point_names, value=point_names[0] if point_names else None),
        first_preview,
        video_map,
    )


def _load_point_metadata(chunks_dir: Path) -> dict[str, dict[str, float]]:
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


def run_phase2_analysis(output_dir_text: str) -> tuple[str, dict[str, Any], list[str], str | None, str, list[list[Any]], plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    if not output_dir_text:
        empty_fig = _empty_figure("No output folder selected")
        return "No output folder selected. Run 'Process Video' first.", {}, [], None, "", [], empty_fig, empty_fig, empty_fig, empty_fig

    chunks_dir = Path(output_dir_text)
    if not chunks_dir.exists():
        empty_fig = _empty_figure(f"Folder not found: {chunks_dir}")
        return f"Output folder does not exist: {chunks_dir}", {}, [], None, "", [], empty_fig, empty_fig, empty_fig, empty_fig

    point_clips = sorted(chunks_dir.glob("point_*/point_*.mp4"))
    if not point_clips:
        empty_fig = _empty_figure("No rally clips found")
        return f"No rally clips found in: {chunks_dir}", {}, [], None, "", [], empty_fig, empty_fig, empty_fig, empty_fig

    point_metadata = _load_point_metadata(chunks_dir)
    point_results = []
    all_trajectories = []
    for clip_path in point_clips:
        point_name = clip_path.stem
        calibration = calibrate_court_from_clip(clip_path)
        tracking = track_bottom_player(clip_path)
        events = detect_events_for_clip(tracking, clip_path=clip_path)
        result = build_point_analysis(
            point_name=point_name,
            clip_path=clip_path,
            point_metadata=point_metadata.get(point_name),
            calibration=calibration,
            tracking=tracking,
            events=events,
        )
        point_results.append(result)
        all_trajectories.append(map_positions_for_metrics(tracking.positions_norm, calibration))

    match_summary = build_match_summary(point_results)
    analysis_dir = chunks_dir / "analysis"
    write_analysis_outputs(analysis_dir, point_results, match_summary)

    heatmap_path = analysis_dir / "court_heatmap.png"
    heatmap_output = None
    if create_court_heatmap(all_trajectories, heatmap_path):
        heatmap_output = str(heatmap_path)
        match_summary.heatmap_image = str(heatmap_path)
        write_analysis_outputs(analysis_dir, point_results, match_summary)

    files = [
        str(analysis_dir / "analysis_match_summary.json"),
        str(analysis_dir / "analysis_per_point.json"),
        str(analysis_dir / "analysis_per_point.csv"),
    ]
    if heatmap_output is not None:
        files.append(heatmap_output)

    status = (
        f"Phase 2 analysis completed. Rallies analyzed: {len(point_results)} | "
        f"Summary: {analysis_dir / 'analysis_match_summary.json'}"
    )

    def _fmt(val: Any, decimals: int = 2) -> str:
        return "N/A" if val is None else f"{val:.{decimals}f}"

    avg_speed_str = (
        f"{match_summary.avg_speed_mps:.2f} m/s"
        if match_summary.avg_speed_mps is not None
        else _fmt(match_summary.avg_speed_norm_per_sec, 3) + " norm/s"
    )
    max_speed_str = (
        f"{match_summary.max_speed_mps:.2f} m/s"
        if match_summary.max_speed_mps is not None
        else _fmt(match_summary.max_speed_norm_per_sec, 3) + " norm/s"
    )

    stats_text = (
        f"Rallies analyzed:        {len(point_results)}\n"
        f"Avg shots / rally:       {_fmt(match_summary.avg_shots_per_rally, 1)}\n"
        f"Avg smashes / rally:     {_fmt(match_summary.avg_smashes_per_rally, 1)}\n"
        f"Total smashes est:       {match_summary.total_smashes_est if match_summary.total_smashes_est is not None else 'N/A'}\n"
        f"Avg player speed:        {avg_speed_str}\n"
        f"Peak player speed:       {max_speed_str}\n"
        f"Avg area covered (norm): {_fmt(match_summary.avg_area_covered_norm, 4)}"
    )

    # Build per-point analytics table (list of lists for gr.Dataframe).
    table_rows = []
    for r in point_results:
        if r.speed_avg_mps is not None:
            spd_str = f"{r.speed_avg_mps:.2f} m/s"
        elif r.speed_avg_norm_per_sec is not None:
            spd_str = f"{r.speed_avg_norm_per_sec:.3f} n/s"
        else:
            spd_str = "N/A"
        table_rows.append([
            r.point_name,
            f"{r.duration_sec:.1f}",
            str(r.shots_est) if r.shots_est is not None else "N/A",
            str(r.smash_count_est) if r.smash_count_est is not None else "N/A",
            spd_str,
            f"{r.area_covered_norm:.4f}" if r.area_covered_norm is not None else "N/A",
            r.court_calibration_status,
        ])

    shots_chart = _build_shots_chart(point_results)
    speed_chart = _build_speed_chart(point_results)
    area_chart = _build_area_chart(point_results)
    duration_chart = _build_duration_chart(point_results)

    return status, asdict(match_summary), files, heatmap_output, stats_text, table_rows, shots_chart, speed_chart, area_chart, duration_chart


def select_clip(point_name: str, video_map: dict[str, str]) -> str | None:
    if not point_name:
        return None
    return video_map.get(point_name)


def _build_shots_chart(point_results: list[Any]) -> plt.Figure:
    """Create bar chart for shots per point."""
    point_names = [r.point_name for r in point_results]
    shots_data = [r.shots_est if r.shots_est is not None else 0 for r in point_results]
    smashes_data = [r.smash_count_est if r.smash_count_est is not None else 0 for r in point_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(point_names))
    width = 0.35
    ax.bar(x - width / 2, shots_data, width, label="Shots", color="steelblue")
    ax.bar(x + width / 2, smashes_data, width, label="Smashes", color="coral")
    ax.set_xlabel("Rally")
    ax.set_ylabel("Count")
    ax.set_title("Shot Count per Rally")
    ax.set_xticks(x)
    ax.set_xticklabels(point_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def _build_speed_chart(point_results: list[Any]) -> plt.Figure:
    """Create line chart for speed progression."""
    point_names = [r.point_name for r in point_results]
    avg_speeds = [r.speed_avg_mps if r.speed_avg_mps is not None else r.speed_avg_norm_per_sec for r in point_results]
    max_speeds = [r.speed_max_mps if r.speed_max_mps is not None else r.speed_max_norm_per_sec for r in point_results]

    unit_label = "m/s" if avg_speeds and point_results[0].speed_avg_mps is not None else "norm/s"
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(point_names))
    ax.plot(x, avg_speeds, marker="o", label="Avg Speed", color="green", linewidth=2)
    ax.plot(x, max_speeds, marker="s", label="Peak Speed", color="red", linewidth=2, linestyle="--")
    ax.set_xlabel("Rally")
    ax.set_ylabel(f"Speed ({unit_label})")
    ax.set_title(f"Player Speed per Rally ({unit_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(point_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def _build_area_chart(point_results: list[Any]) -> plt.Figure:
    """Create bar chart for area covered per point."""
    point_names = [r.point_name for r in point_results]
    area_data = [r.area_covered_norm if r.area_covered_norm is not None else 0.0 for r in point_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(point_names, area_data, color="purple", alpha=0.7)
    ax.set_xlabel("Rally")
    ax.set_ylabel("Normalized Area")
    ax.set_title("Court Area Covered per Rally (normalized)")
    ax.set_xticklabels(point_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def _build_duration_chart(point_results: list[Any]) -> plt.Figure:
    """Create bar chart for rally duration."""
    point_names = [r.point_name for r in point_results]
    durations = [r.duration_sec for r in point_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(point_names, durations, color="teal", alpha=0.7)
    ax.set_xlabel("Rally")
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("Rally Duration")
    ax.set_xticklabels(point_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def _empty_figure(title: str = "No Data") -> plt.Figure:
    """Create an empty matplotlib figure with a message."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, title, ha="center", va="center", transform=ax.transAxes, fontsize=14)
    ax.axis("off")
    return fig


def select_clip(point_name: str, video_map: dict[str, str]) -> str | None:
    if not point_name:
        return None
    return video_map.get(point_name)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Badminton Point Chunking") as app:
        gr.Markdown("## Badminton Point Chunking (Singles)")
        gr.Markdown(
            "Target player must stay in the bottom half of the screen. "
            "Drag/drop a video and click **Process Video**."
        )

        with gr.Row():
            video_input = gr.File(
                label="Input Match Video",
                file_types=[".mp4", ".mov", ".avi", ".mkv"],
                type="filepath",
            )
            model_input = gr.File(
                label="Optional Trained Model (.joblib)",
                file_types=[".joblib"],
                type="filepath",
            )

        with gr.Accordion("Advanced Detection Settings", open=False):
            with gr.Row():
                threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Threshold")
                smooth_window = gr.Slider(1, 31, value=9, step=2, label="Smooth Window (frames)")

            with gr.Row():
                start_confirm = gr.Slider(0.2, 2.0, value=0.6, step=0.05, label="Start Confirm (sec)")
                end_confirm = gr.Slider(0.3, 3.0, value=1.0, step=0.05, label="End Confirm (sec)")

            with gr.Row():
                min_rally = gr.Slider(0.5, 8.0, value=2.0, step=0.1, label="Min Rally (sec)")
                min_gap = gr.Slider(0.2, 5.0, value=1.2, step=0.1, label="Min Gap (sec)")

            with gr.Row():
                pre_pad = gr.Slider(0.0, 2.0, value=0.25, step=0.05, label="Clip Pre Pad (sec)")
                post_pad = gr.Slider(0.0, 2.0, value=0.35, step=0.05, label="Clip Post Pad (sec)")

        run_button = gr.Button("Process Video", variant="primary")

        status_output = gr.Textbox(label="Status", interactive=False)
        json_output = gr.JSON(label="Detected Segments")
        files_output = gr.Files(label="Point-wise Output Videos")
        output_dir_box = gr.Textbox(label="Output Folder", interactive=False)
        point_selector = gr.Dropdown(label="Preview Point Clip", choices=[])
        preview_video = gr.Video(label="Trimmed Point Preview", format="mp4")
        clip_map_state = gr.State({})

        with gr.Accordion("Phase 2: Game Analysis (Current Implementation)", open=False):
            gr.Markdown(
                "Runs current analysis pipeline on generated rally clips and writes "
                "`analysis_match_summary.json`, `analysis_per_point.json`, and `analysis_per_point.csv`."
            )
            run_analysis_button = gr.Button("Run Phase 2 Analysis")
            analysis_status_output = gr.Textbox(label="Analysis Status", interactive=False)
            analysis_stats_output = gr.Textbox(label="Key Metrics", interactive=False, lines=7)
            analysis_table_output = gr.Dataframe(
                label="Per-Point Analytics",
                headers=["Point", "Duration(s)", "Shots", "Smashes", "Avg Speed", "Area(norm)", "Calibration"],
                interactive=False,
            )

            with gr.Row():
                shots_chart = gr.Plot(label="Rally Shots & Smashes")
                speed_chart = gr.Plot(label="Player Speed Progression")

            with gr.Row():
                area_chart = gr.Plot(label="Court Area Coverage")
                duration_chart = gr.Plot(label="Rally Duration")

            analysis_summary_output = gr.JSON(label="Analysis Match Summary")
            analysis_heatmap_output = gr.Image(label="Court Coverage Heatmap", type="filepath")
            analysis_files_output = gr.Files(label="Analysis Output Files")

        run_button.click(
            fn=process_video,
            inputs=[
                video_input,
                model_input,
                threshold,
                start_confirm,
                end_confirm,
                min_rally,
                min_gap,
                smooth_window,
                pre_pad,
                post_pad,
            ],
            outputs=[
                status_output,
                json_output,
                files_output,
                output_dir_box,
                point_selector,
                preview_video,
                clip_map_state,
            ],
        )

        point_selector.change(
            fn=select_clip,
            inputs=[point_selector, clip_map_state],
            outputs=[preview_video],
        )

        run_analysis_button.click(
            fn=run_phase2_analysis,
            inputs=[output_dir_box],
            outputs=[
                analysis_status_output,
                analysis_summary_output,
                analysis_files_output,
                analysis_heatmap_output,
                analysis_stats_output,
                analysis_table_output,
                shots_chart,
                speed_chart,
                area_chart,
                duration_chart,
            ],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    outputs_root = OUTPUTS_ROOT.resolve()
    app.launch(inbrowser=True, allowed_paths=[str(outputs_root)])
