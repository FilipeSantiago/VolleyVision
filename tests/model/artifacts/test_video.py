import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from model.artifacts.video import (
    rank_test_videos,
    render_domain_shift_video,
    render_test_video,
)


class FixedPeakModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        heatmaps = torch.zeros(
            (images.shape[0], 1, images.shape[2], images.shape[3]),
            device=images.device,
        )
        heatmaps[:, :, 2, 3] = 0.75
        return heatmaps


class VideoArtifactTest(unittest.TestCase):
    def test_rank_test_videos_prefers_highest_pck_10(self):
        predictions = pd.DataFrame(
            [
                _prediction("accurate", 0, 1, 2.0, 0.8),
                _prediction("accurate", 1, 1, 8.0, 0.7),
                _prediction("accurate", 2, 0, np.nan, 0.2),
                _prediction("mixed", 0, 1, 2.0, 0.8),
                _prediction("mixed", 1, 1, 18.0, 0.7),
            ]
        )

        ranking = rank_test_videos(predictions)

        self.assertEqual(ranking.iloc[0]["video_id"], "accurate")
        self.assertEqual(ranking.iloc[0]["pck_10"], 1.0)
        self.assertEqual(
            ranking.iloc[0]["invisible_high_confidence_rate"],
            1.0,
        )

    def test_renderers_preserve_frame_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame_paths = []
            for index in range(3):
                frame_path = root / f"{index:06d}.jpg"
                cv2.imwrite(
                    str(frame_path),
                    np.full((48, 64, 3), 40 + index * 20, dtype=np.uint8),
                )
                frame_paths.append(frame_path)

            frame_table = pd.DataFrame(
                {
                    "video_id": ["sample"] * 3,
                    "frame": [0, 1, 2],
                    "frame_path": [str(path) for path in frame_paths],
                    "fps": [12.0] * 3,
                    "video_path": ["source.mp4"] * 3,
                }
            )
            predictions = pd.DataFrame(
                [
                    {
                        **_prediction("sample", index, 1, float(index), 0.8),
                        "dataset_index": index,
                        "predicted_x": 20 + index,
                        "predicted_y": 20,
                        "target_x": 20,
                        "target_y": 20,
                    }
                    for index in range(3)
                ]
            )
            test_output = root / "test.mp4"
            summary = render_test_video(
                frame_table,
                predictions,
                "sample",
                test_output,
                confidence_threshold=0.1,
            )
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["marker_style"], "comet_trail")
            self.assertEqual(_video_frame_count(test_output), 3)

            source_video = root / "domain.mp4"
            _write_video(source_video, frame_paths, fps=12.0)
            domain_output = root / "domain_annotated.mp4"
            domain_predictions, domain_summary = render_domain_shift_video(
                model=FixedPeakModel(),
                device=torch.device("cpu"),
                video_path=source_video,
                output_path=domain_output,
                model_width=64,
                model_height=48,
                confidence_threshold=0.1,
            )
            self.assertEqual(len(domain_predictions), 3)
            self.assertEqual(domain_summary["frames"], 3)
            self.assertEqual(domain_summary["marker_style"], "comet_trail")
            self.assertEqual(_video_frame_count(domain_output), 3)


def _prediction(
        video_id: str,
        frame: int,
        visibility: int,
        distance_px: float,
        peak_value: float,
) -> dict:
    return {
        "video_id": video_id,
        "frame": frame,
        "visibility": visibility,
        "distance_px": distance_px,
        "peak_value": peak_value,
    }


def _write_video(path: Path, frame_paths: list[Path], fps: float) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create test video {path}")
    try:
        for frame_path in frame_paths:
            writer.write(cv2.imread(str(frame_path)))
    finally:
        writer.release()


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


if __name__ == "__main__":
    unittest.main()
