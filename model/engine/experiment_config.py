from dataclasses import dataclass, field
from pathlib import Path

import torch

from dataset.augmentation import AugmentationConfig
from model.artifacts.config import ArtifactConfig
from monitoring.mlflow_config import MLflowConfig


@dataclass(frozen=True)
class ExperimentConfig:
    raw_data_dir: Path
    processed_data_dir: Path
    checkpoint_path: Path
    image_width: int = 512
    image_height: int = 288
    sigma: float = 3.0
    train_batch_size: int = 16
    evaluation_batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    epochs: int = 51
    learning_rate: float = 1e-3
    validation_interval: int = 5
    small_val_size: int = 1500
    small_val_seed: int = 0
    early_stopping_monitor: str = "val_small_pck_10"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.002
    thresholds: tuple[int, ...] = (5, 10, 20)
    device: str | None = None
    train_augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    def resolved_device(self) -> torch.device:
        selected_device = self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return torch.device(selected_device)
