from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.eval.heatmap_decoder import HeatmapDecoder
from model.eval.localization_metric_accumulator import LocalizationMetricAccumulator


class BallLocalizationEvaluator:
    def __init__(
        self,
        thresholds: Sequence[int] = (5, 10, 20),
        decoder: HeatmapDecoder | None = None,
    ):
        self.thresholds = tuple(thresholds)
        self.decoder = decoder or HeatmapDecoder()

    @torch.inference_mode()
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        frame_table: pd.DataFrame | None = None,
        description: str = "Evaluating",
    ) -> tuple[dict[str, float | int], pd.DataFrame | None]:
        accumulator = LocalizationMetricAccumulator(
            thresholds=self.thresholds,
            decoder=self.decoder,
        )
        prediction_rows = []
        dataset_offset = 0
        was_training = model.training
        model.eval()

        try:
            for batch in tqdm(data_loader, desc=description, leave=False):
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                visibility = batch["visibility"]

                predictions = model(images)
                accumulator.update(predictions, targets, visibility)

                if frame_table is not None:
                    prediction_rows.extend(
                        self._prediction_rows(
                            predictions=predictions,
                            targets=targets,
                            visibility=visibility,
                            frame_table=frame_table,
                            dataset_offset=dataset_offset,
                        )
                    )
                dataset_offset += images.shape[0]
        finally:
            if was_training:
                model.train()

        if frame_table is not None and dataset_offset != len(frame_table):
            raise ValueError(
                "The frame table and data loader produced different row counts: "
                f"{len(frame_table)} != {dataset_offset}."
            )

        prediction_table = (
            pd.DataFrame(prediction_rows)
            if frame_table is not None
            else None
        )
        return accumulator.compute(), prediction_table

    def _prediction_rows(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        visibility: torch.Tensor,
        frame_table: pd.DataFrame,
        dataset_offset: int,
    ) -> list[dict]:
        predicted_coordinates, peak_values = self.decoder.peak_coordinates(
            predictions
        )
        target_coordinates, _ = self.decoder.peak_coordinates(targets)

        predicted_coordinates = predicted_coordinates.detach().cpu()
        target_coordinates = target_coordinates.detach().cpu()
        peak_values = peak_values.detach().cpu()
        visibility = visibility.reshape(-1).bool().cpu()
        rows = []

        for local_index in range(predictions.shape[0]):
            dataset_index = dataset_offset + local_index
            source_row = frame_table.iloc[dataset_index]
            predicted_x, predicted_y = (
                int(value)
                for value in predicted_coordinates[local_index]
            )
            is_visible = bool(visibility[local_index])
            result = {
                "dataset_index": dataset_index,
                "video_id": source_row["video_id"],
                "frame": int(source_row["frame"]),
                "visibility": int(is_visible),
                "predicted_x": predicted_x,
                "predicted_y": predicted_y,
                "peak_value": float(peak_values[local_index]),
                "target_x": np.nan,
                "target_y": np.nan,
                "distance_px": np.nan,
            }

            if is_visible:
                target_x, target_y = (
                    int(value)
                    for value in target_coordinates[local_index]
                )
                result.update(
                    {
                        "target_x": target_x,
                        "target_y": target_y,
                        "distance_px": float(
                            np.hypot(
                                predicted_x - target_x,
                                predicted_y - target_y,
                            )
                        ),
                    }
                )
            rows.append(result)
        return rows