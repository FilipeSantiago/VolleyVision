from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class VolleyBallDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        sigma: float = 3.0,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.sigma = sigma

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

        image = torch.from_numpy(
            image.astype(np.float32) / 255.0
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
