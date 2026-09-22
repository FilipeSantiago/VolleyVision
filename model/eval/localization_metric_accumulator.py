from typing import Sequence

import numpy as np
import torch

from model.eval.heatmap_decoder import HeatmapDecoder


class LocalizationMetricAccumulator:
    def __init__(
        self,
        thresholds: Sequence[int] = (5, 10, 20),
        decoder: HeatmapDecoder | None = None,
    ):
        self.thresholds = tuple(thresholds)
        self.decoder = decoder or HeatmapDecoder()
        self.reset()

    def reset(self) -> None:
        self.squared_error_sum = 0.0
        self.pixel_count = 0
        self.total_frames = 0
        self._distances: list[torch.Tensor] = []

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        visibility: torch.Tensor,
    ) -> None:
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction shape {tuple(predictions.shape)} does not match "
                f"target shape {tuple(targets.shape)}."
            )

        self.squared_error_sum += torch.sum(
            (predictions - targets) ** 2
        ).item()
        self.pixel_count += targets.numel()
        self.total_frames += targets.shape[0]

        visible_mask = visibility.reshape(-1).bool().to(predictions.device)
        if not visible_mask.any():
            return

        predicted_coordinates, _ = self.decoder.peak_coordinates(
            predictions[visible_mask]
        )
        target_coordinates, _ = self.decoder.peak_coordinates(
            targets[visible_mask]
        )
        distances = torch.linalg.vector_norm(
            predicted_coordinates.float() - target_coordinates.float(),
            dim=1,
        )
        self._distances.append(distances.detach().cpu())

    def compute(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "frames": self.total_frames,
            "heatmap_mse": (
                self.squared_error_sum / self.pixel_count
                if self.pixel_count
                else float("nan")
            ),
        }

        if self._distances:
            distances = torch.cat(self._distances).numpy()
        else:
            distances = np.array([], dtype=np.float32)

        metrics["visible_frames"] = int(distances.size)
        metrics["invisible_frames"] = self.total_frames - int(distances.size)
        metrics["mean_distance_px"] = (
            float(distances.mean())
            if distances.size
            else float("nan")
        )
        metrics["median_distance_px"] = (
            float(np.median(distances))
            if distances.size
            else float("nan")
        )
        for threshold in self.thresholds:
            metrics[f"pck_{threshold}"] = (
                float((distances <= threshold).mean())
                if distances.size
                else float("nan")
            )
        return metrics