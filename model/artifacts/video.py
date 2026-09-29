import hashlib
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from model.artifacts.config import NamedRegion

COMET_COLOR = (255, 225, 60)
COMET_HEAD_COLOR = (245, 250, 255)
COMET_TRAIL_LENGTH = 10
COMET_MAX_JUMP_PX = 96.0


def rank_test_videos(
        predictions: pd.DataFrame,
        thresholds: Sequence[int] = (5, 10, 20),
        confidence_threshold: float = 0.10,
) -> pd.DataFrame:
    required_columns = {
        "video_id",
        "visibility",
        "distance_px",
        "peak_value",
    }
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Predictions are missing required columns: {sorted(missing_columns)}"
        )
    if 10 not in thresholds:
        raise ValueError("Video selection requires a PCK@10 threshold.")

    rows = []
    for video_id, video_predictions in predictions.groupby("video_id", sort=True):
        visible = video_predictions.loc[
            video_predictions["visibility"].astype(bool)
            & video_predictions["distance_px"].notna()
            ]
        invisible = video_predictions.loc[
            ~video_predictions["visibility"].astype(bool)
        ]
        distances = visible["distance_px"].astype(float)
        row = {
            "video_id": str(video_id),
            "frames": int(len(video_predictions)),
            "visible_frames": int(len(visible)),
            "invisible_frames": int(len(invisible)),
            "mean_distance_px": float(distances.mean()),
            "median_distance_px": float(distances.median()),
            "mean_peak_value": float(video_predictions["peak_value"].mean()),
            "invisible_high_confidence_rate": (
                float((invisible["peak_value"] >= confidence_threshold).mean())
                if not invisible.empty
                else float("nan")
            ),
        }
        for threshold in thresholds:
            row[f"pck_{threshold}"] = float((distances <= threshold).mean())
        rows.append(row)

    if not rows:
        raise ValueError("Cannot rank an empty prediction table.")

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        by=[
            "pck_10",
            "median_distance_px",
            "mean_distance_px",
            "visible_frames",
            "video_id",
        ],
        ascending=[False, True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def render_test_video(
        frame_table: pd.DataFrame,
        predictions: pd.DataFrame,
        video_id: str,
        output_path: Path,
        confidence_threshold: float,
        codec: str = "mp4v",
        fallback_fps: float = 30.0,
) -> dict[str, object]:
    selected = predictions.loc[predictions["video_id"].eq(video_id)].copy()
    if selected.empty:
        raise ValueError(f"No test predictions found for video {video_id!r}.")
    selected = selected.sort_values("frame")

    source_rows = frame_table.iloc[selected["dataset_index"].astype(int)].copy()
    if not source_rows["video_id"].astype(str).eq(video_id).all():
        raise ValueError("Prediction dataset indices do not match the selected video.")

    first_frame = cv2.imread(str(source_rows.iloc[0]["frame_path"]))
    if first_frame is None:
        raise RuntimeError(f"Could not read {source_rows.iloc[0]['frame_path']}")
    height, width = first_frame.shape[:2]
    fps = _frame_table_fps(source_rows, fallback_fps)
    writer = _open_writer(output_path, codec, fps, width, height)
    history: list[tuple[int, int]] = []

    try:
        for source, prediction in tqdm(
                zip(source_rows.itertuples(index=False), selected.itertuples(index=False)),
                total=len(selected),
                desc=f"Rendering test video {video_id}",
                leave=False,
        ):
            frame = cv2.imread(str(source.frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read {source.frame_path}")
            predicted = (int(prediction.predicted_x), int(prediction.predicted_y))
            show_prediction = (
                    bool(prediction.visibility)
                    or float(prediction.peak_value) >= confidence_threshold
            )
            if show_prediction:
                _append_trail_point(history, predicted)
            else:
                history.clear()
            annotated = _annotate_test_frame(
                frame,
                prediction,
                confidence_threshold,
                history,
            )
            writer.write(annotated)
    finally:
        writer.release()

    return {
        "video_id": video_id,
        "source_video_path": _first_value(source_rows, "video_path"),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": int(len(selected)),
        "confidence_threshold": confidence_threshold,
        "codec": codec,
        "marker_style": "comet_trail",
        "trail_length": COMET_TRAIL_LENGTH,
    }


@torch.inference_mode()
def render_domain_shift_video(
        model: nn.Module,
        device: torch.device,
        video_path: Path,
        output_path: Path,
        model_width: int,
        model_height: int,
        confidence_threshold: float,
        excluded_regions: Sequence[NamedRegion] = (),
        codec: str = "mp4v",
        fallback_fps: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Domain-shift video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open domain-shift video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions reported by {video_path}")
    if fps <= 0:
        fps = fallback_fps

    writer = _open_writer(output_path, codec, fps, width, height)
    predictions = []
    history: list[tuple[int, int]] = []
    frame_index = 0
    was_training = model.training
    model.eval()

    try:
        progress = tqdm(
            total=expected_frames or None,
            desc="Rendering domain-shift video",
            unit="frame",
            leave=False,
        )
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                heatmap = model(
                    _preprocess_frame(frame, model_width, model_height, device)
                )[0, 0].float().cpu()
                filtered_heatmap = _suppress_regions(heatmap, excluded_regions)
                flat_index = int(filtered_heatmap.argmax().item())
                model_y, model_x = divmod(flat_index, model_width)
                peak_value = float(filtered_heatmap[model_y, model_x].item())
                x = round(model_x * (width - 1) / (model_width - 1))
                y = round(model_y * (height - 1) / (model_height - 1))
                detected = peak_value >= confidence_threshold
                prediction = {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / fps,
                    "x": int(x),
                    "y": int(y),
                    "peak_value": peak_value,
                    "detected": bool(detected),
                }
                predictions.append(prediction)
                if detected:
                    _append_trail_point(history, (int(x), int(y)))
                else:
                    history.clear()
                writer.write(_annotate_domain_frame(frame, prediction, history))
                frame_index += 1
                progress.update(1)
        finally:
            progress.close()
    finally:
        capture.release()
        writer.release()
        if was_training:
            model.train()

    prediction_table = pd.DataFrame(predictions)
    if prediction_table.empty:
        raise RuntimeError(f"Domain-shift video contains no readable frames: {video_path}")
    summary = {
        "source_video_path": str(video_path.resolve()),
        "source_sha256": sha256_file(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_index,
        "expected_frames": expected_frames,
        "detected_frames": int(prediction_table["detected"].sum()),
        "detection_rate": float(prediction_table["detected"].mean()),
        "confidence_threshold": confidence_threshold,
        "excluded_regions": {
            name: list(region) for name, region in excluded_regions
        },
        "codec": codec,
        "marker_style": "comet_trail",
        "trail_length": COMET_TRAIL_LENGTH,
        "ground_truth_available": False,
    }
    return prediction_table, summary


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preprocess_frame(
        frame_bgr: np.ndarray,
        width: int,
        height: int,
        device: torch.device,
) -> torch.Tensor:
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


def _suppress_regions(
        heatmap: torch.Tensor,
        excluded_regions: Sequence[NamedRegion],
) -> torch.Tensor:
    filtered = heatmap.clone()
    height, width = filtered.shape
    for name, region in excluded_regions:
        left, top, right, bottom = region
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError(f"Invalid excluded region {name!r}: {region}")
        x1 = max(0, math.floor(left * width))
        y1 = max(0, math.floor(top * height))
        x2 = min(width, math.ceil(right * width))
        y2 = min(height, math.ceil(bottom * height))
        filtered[y1:y2, x1:x2] = -torch.inf
    return filtered


def _annotate_test_frame(
        frame: np.ndarray,
        prediction,
        confidence_threshold: float,
        history: Sequence[tuple[int, int]],
) -> np.ndarray:
    annotated = frame.copy()
    visible = bool(prediction.visibility)
    predicted = (int(prediction.predicted_x), int(prediction.predicted_y))
    peak_value = float(prediction.peak_value)

    if visible:
        target = (int(prediction.target_x), int(prediction.target_y))
        error = float(prediction.distance_px)
        line_color = (0, 210, 0) if error <= 10 else (0, 165, 255) if error <= 20 else (0, 0, 255)
        cv2.line(annotated, target, predicted, line_color, 2, cv2.LINE_AA)
        cv2.circle(annotated, target, 10, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(annotated, target, 10, (0, 255, 0), 2, cv2.LINE_AA)
        status = f"error {error:.1f}px"
    else:
        status = "GT: ball not visible"

    if visible or peak_value >= confidence_threshold:
        _draw_comet_trail(annotated, predicted, history)

    _draw_header(
        annotated,
        f"TEST | frame {int(prediction.frame)} | peak {peak_value:.3f} | {status}",
    )
    return annotated


def _annotate_domain_frame(
        frame: np.ndarray,
        prediction: dict,
        history: Sequence[tuple[int, int]],
) -> np.ndarray:
    annotated = frame.copy()
    if prediction["detected"]:
        center = (int(prediction["x"]), int(prediction["y"]))
        _draw_comet_trail(annotated, center, history)
    state = "detected" if prediction["detected"] else "below threshold"
    _draw_header(
        annotated,
        "DOMAIN SHIFT | no ground truth | "
        f"frame {prediction['frame_index']} | peak {prediction['peak_value']:.3f} | {state}",
    )
    return annotated


def _append_trail_point(
        history: list[tuple[int, int]],
        point: tuple[int, int],
) -> None:
    if history and math.dist(history[-1], point) > COMET_MAX_JUMP_PX:
        history.clear()
    history.append(point)
    del history[:-COMET_TRAIL_LENGTH]


def _draw_comet_trail(
        frame: np.ndarray,
        center: tuple[int, int],
        history: Sequence[tuple[int, int]],
) -> None:
    points = list(history[-COMET_TRAIL_LENGTH:])
    if len(points) > 1:
        overlay = frame.copy()
        for index in range(1, len(points)):
            strength = index / (len(points) - 1)
            color = tuple(
                round(channel * strength)
                for channel in COMET_COLOR
            )
            cv2.line(
                overlay,
                points[index - 1],
                points[index],
                color,
                max(1, round(4 * strength)),
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.circle(frame, center, 10, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.circle(frame, center, 10, COMET_HEAD_COLOR, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 3, COMET_COLOR, -1, cv2.LINE_AA)


def _draw_header(frame: np.ndarray, text: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.putText(
        frame,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _open_writer(
        output_path: Path,
        codec: str,
        fps: float,
        width: int,
        height: int,
) -> cv2.VideoWriter:
    if len(codec) != 4:
        raise ValueError("Video codec must be a four-character code.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create video: {output_path}")
    return writer


def _frame_table_fps(frame_table: pd.DataFrame, fallback_fps: float) -> float:
    if "fps" not in frame_table.columns:
        return fallback_fps
    values = frame_table["fps"].dropna()
    if values.empty or float(values.iloc[0]) <= 0:
        return fallback_fps
    return float(values.iloc[0])


def _first_value(frame_table: pd.DataFrame, column: str) -> str | None:
    if column not in frame_table.columns:
        return None
    values = frame_table[column].dropna()
    return str(values.iloc[0]) if not values.empty else None
