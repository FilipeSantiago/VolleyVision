from typing import Callable

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from dataset.preprocess.vballnet import VBallNet
from dataset.volley_ball_dataset import VolleyBallDataset
from model.engine.experiment_config import ExperimentConfig


class BallDataModule:
    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        config: ExperimentConfig,
        dataset_factory: Callable[..., Dataset] = VolleyBallDataset,
        manifest_builder_factory: Callable[[], VBallNet] = VBallNet,
    ):
        self.config = config
        self.dataset_factory = dataset_factory
        self.manifest_builder_factory = manifest_builder_factory
        self.manifest_df = pd.DataFrame()
        self._frame_tables: dict[str, pd.DataFrame] = {}
        self._datasets: dict[str, Dataset] = {}

    def prepare_data(self) -> "BallDataModule":
        builder = self.manifest_builder_factory()
        manifest = (
            builder
            .build_vballnet_manifest(self.config.raw_data_dir)
            .split_dataset()
            .preprocess_frames(
                output_dir=self.config.processed_data_dir,
                width=self.config.image_width,
                height=self.config.image_height,
            )
            .get_manifest()
        )
        return self.setup(manifest)

    def setup(self, manifest: pd.DataFrame) -> "BallDataModule":
        required_columns = {
            "split",
            "video_id",
            "frame",
            "frame_path",
            "visibility",
            "x",
            "y",
            "width",
            "height",
        }
        missing_columns = required_columns.difference(manifest.columns)
        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing_columns)}"
            )

        self.manifest_df = manifest.reset_index(drop=True).copy()
        self._frame_tables = {
            split: self.manifest_df.loc[
                self.manifest_df["split"].eq(split)
            ].reset_index(drop=True)
            for split in self.SPLITS
        }
        self._validate_splits()
        self._datasets = {
            split: self.dataset_factory(
                frame_table,
                sigma=self.config.sigma,
            )
            for split, frame_table in self._frame_tables.items()
        }
        return self

    def _validate_splits(self) -> None:
        empty_splits = [
            split
            for split, frame_table in self._frame_tables.items()
            if frame_table.empty
        ]
        if empty_splits:
            raise ValueError(f"Empty dataset splits: {empty_splits}")

        video_sets = {
            split: set(frame_table["video_id"])
            for split, frame_table in self._frame_tables.items()
        }
        for index, left_split in enumerate(self.SPLITS):
            for right_split in self.SPLITS[index + 1:]:
                overlap = video_sets[left_split].intersection(video_sets[right_split])
                if overlap:
                    raise ValueError(
                        f"Video leakage between {left_split} and {right_split}: "
                        f"{sorted(overlap)[:5]}"
                    )

    def frame_table(self, split: str) -> pd.DataFrame:
        self._require_split(split)
        return self._frame_tables[split]

    def dataset(self, split: str) -> Dataset:
        self._require_split(split)
        return self._datasets[split]

    def dataloader(self, split: str) -> DataLoader:
        self._require_split(split)
        is_training = split == "train"
        loader_kwargs = {
            "dataset": self._datasets[split],
            "batch_size": (
                self.config.train_batch_size
                if is_training
                else self.config.evaluation_batch_size
            ),
            "shuffle": is_training,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
        }
        if self.config.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(**loader_kwargs)

    def split_summary(self) -> pd.DataFrame:
        summary = (
            self.manifest_df
            .groupby("split")
            .agg(
                frames=("frame", "size"),
                videos=("video_id", "nunique"),
                visible_frames=("visibility", "sum"),
            )
            .reindex(self.SPLITS)
        )
        summary["visible_frames"] = summary["visible_frames"].astype(int)
        summary["invisible_frames"] = (
            summary["frames"] - summary["visible_frames"]
        )
        return summary

    def _require_split(self, split: str) -> None:
        if split not in self.SPLITS:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {self.SPLITS}."
            )
        if split not in self._datasets:
            raise RuntimeError("Call prepare_data() or setup() before requesting data.")
