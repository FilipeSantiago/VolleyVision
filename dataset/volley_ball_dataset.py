from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.augmentation.context import AugmentationContext, BallCenter


class VolleyBallDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        sigma: float = 3.0,
        image_transform: Callable[
            [np.ndarray, AugmentationContext],
            np.ndarray,
        ] | None = None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.sigma = sigma
        self.image_transform = image_transform
        self._frame_indices = {
            (row.video_id, int(row.frame)): dataset_index
            for dataset_index, row in enumerate(
                self.manifest.itertuples(index=False)
            )
        }

        self.height = int(self.manifest.iloc[0]["height"])
        self.width = int(self.manifest.iloc[0]["width"])

        # Create once instead of recreating for every sample
        yy, xx = torch.meshgrid(
            torch.arange(self.height),
            torch.arange(self.width),
            indexing="ij",
        )

        self.xx = xx.float()
        self.yy = yy.float()

    def __len__(self):
        return len(self.manifest)

    def augmentation_context(self, index: int) -> AugmentationContext:
        row = self.manifest.iloc[index]
        video_id = row["video_id"]
        frame_number = int(row["frame"])
        previous_row = self._neighbor_row(video_id, frame_number - 1)
        next_row = self._neighbor_row(video_id, frame_number + 1)
        return AugmentationContext(
            video_id=str(video_id),
            frame_number=frame_number,
            previous_center=self._visible_center(previous_row),
            center=self._visible_center(row),
            next_center=self._visible_center(next_row),
        )

    def _neighbor_row(
        self,
        video_id,
        frame_number: int,
    ) -> pd.Series | None:
        neighbor_index = self._frame_indices.get((video_id, frame_number))
        if neighbor_index is None:
            return None
        return self.manifest.iloc[neighbor_index]

    @staticmethod
    def _visible_center(row: pd.Series | None) -> BallCenter | None:
        if row is None or int(row["visibility"]) != 1:
            return None
        x_coordinate = row["x"]
        y_coordinate = row["y"]
        if pd.isna(x_coordinate) or pd.isna(y_coordinate):
            return None
        return float(x_coordinate), float(y_coordinate)

    def _gaussian_heatmap(
        self,
        x: float,
        y: float,
    ) -> torch.Tensor:
        heatmap = torch.exp(
            -(
                (self.xx - x) ** 2
                + (self.yy - y) ** 2
            )
            / (2 * self.sigma ** 2)
        )

        return heatmap.unsqueeze(0)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        image = cv2.imread(row["frame_path"])

        if image is None:
            raise RuntimeError(
                f"Could not read {row['frame_path']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        if self.image_transform is not None:
            image = self.image_transform(
                image,
                self.augmentation_context(index),
            )

        image = torch.from_numpy(
            np.ascontiguousarray(image).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        visibility = int(row["visibility"])

        if visibility:
            target = self._gaussian_heatmap(
                x=float(row["x"]),
                y=float(row["y"]),
            )
        else:
            target = torch.zeros(
                (1, self.height, self.width),
                dtype=torch.float32,
            )

        return {
            "image": image,
            "target": target,
            "visibility": torch.tensor(
                visibility,
                dtype=torch.float32,
            ),
        }
