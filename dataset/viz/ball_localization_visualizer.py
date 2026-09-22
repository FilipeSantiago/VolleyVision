import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class BallLocalizationVisualizer:
    def __init__(self, heatmap_cmap: str = "magma"):
        self.heatmap_cmap = heatmap_cmap

    @staticmethod
    def _image_from_sample(sample: dict[str, torch.Tensor]) -> np.ndarray:
        return sample["image"].permute(1, 2, 0).detach().cpu().numpy()

    def plot_visible_target(
        self,
        dataset: Dataset,
        frame_table: pd.DataFrame,
        visible_rank: int = 0,
    ):
        visible_indices = frame_table.index[frame_table["visibility"].eq(1)]
        if visible_indices.empty:
            raise ValueError("The selected split has no visible-ball frames.")
        if not 0 <= visible_rank < len(visible_indices):
            raise IndexError(
                f"visible_rank must be between 0 and {len(visible_indices) - 1}."
            )
        sample_index = int(visible_indices[visible_rank])
        return self.plot_target(dataset, frame_table, sample_index)

    def plot_target(
        self,
        dataset: Dataset,
        frame_table: pd.DataFrame,
        sample_index: int,
    ):
        sample = dataset[sample_index]
        image = self._image_from_sample(sample)
        target = sample["target"].squeeze(0).detach().cpu().numpy()
        metadata = frame_table.iloc[sample_index]

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(16, 5),
            constrained_layout=True,
        )
        axes[0].imshow(image)
        axes[0].set_title("Original frame")
        axes[0].axis("off")

        axes[1].imshow(image)
        heatmap = axes[1].imshow(
            target,
            cmap=self.heatmap_cmap,
            alpha=0.90,
            vmin=0.0,
            vmax=1.0,
        )
        axes[1].set_title("Gaussian target overlay")
        axes[1].axis("off")
        figure.colorbar(
            heatmap,
            ax=axes[1],
            fraction=0.046,
            pad=0.04,
            label="Mask intensity",
        )
        figure.suptitle(
            f"{metadata['video_id']} · frame {int(metadata['frame'])}"
        )
        plt.show()
        return figure

    def plot_errors(
        self,
        dataset: Dataset,
        predictions: pd.DataFrame,
        count: int = 6,
        columns: int = 3,
    ):
        examples = (
            predictions
            .dropna(subset=["distance_px"])
            .nlargest(count, "distance_px")
        )
        if examples.empty:
            raise ValueError("There are no visible-ball predictions to plot.")

        rows = int(np.ceil(len(examples) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6 * columns, 5 * rows),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for axis, (_, result) in zip(flat_axes, examples.iterrows()):
            sample = dataset[int(result["dataset_index"])]
            image = self._image_from_sample(sample)
            video_id = str(result["video_id"])
            short_video_id = (
                video_id
                if len(video_id) <= 28
                else f"{video_id[:27]}…"
            )

            axis.imshow(image)
            axis.scatter(
                result["target_x"],
                result["target_y"],
                s=160,
                facecolors="none",
                edgecolors="lime",
                linewidths=2,
                label="ground truth",
            )
            axis.scatter(
                result["predicted_x"],
                result["predicted_y"],
                s=120,
                marker="x",
                color="yellow",
                linewidths=2,
                label="prediction",
            )
            title = f"{short_video_id} · frame {int(result['frame'])}"
            subtitle = (
                f"error={result['distance_px']:.1f}px · "
                f"peak={result['peak_value']:.3f}"
            )
            axis.set_title(title + chr(10) + subtitle, fontsize=9)
            axis.axis("off")

        for axis in flat_axes[len(examples):]:
            axis.axis("off")

        flat_axes[0].legend(loc="lower right")
        figure.tight_layout()
        plt.show()
        return figure