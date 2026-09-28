from pathlib import Path
from typing import Callable

from torch import nn

from dataset.augmentation import MotionBlurConfig, AugmentationConfig
from model.cnn import CNN
from model.engine.ball_data_module import BallDataModule
from model.engine.ball_localization_trainer import BallLocalizationTrainer
from model.engine.experiment_config import ExperimentConfig
from model.engine.model_checkpoint import ModelCheckpoint
from model.eval.ball_location_evaluator import BallLocalizationEvaluator

def load_config():
    config = ExperimentConfig(
        raw_data_dir=Path(
            "/home/bugslayer/Documents/research/data/VolleyVision/VballNet/train"
        ),
        processed_data_dir=Path(
            "/home/bugslayer/Documents/research/data/VolleyVision/processed/512x288"
        ),
        checkpoint_path=Path(
            "/home/bugslayer/Documents/research/VolleyVision/notebooks/dataset/ball_model.pth"
        ),
        train_augmentation=AugmentationConfig(
            motion_blur=MotionBlurConfig(
                probability=0.30,
                kernel_size_range=(3, 15),
                strength_bias=1.5,
                trajectory_displacement_range=(1.0, 30.0),
            )
        ),
    )

    return config

def load_data(config):
    data_module = BallDataModule(config).prepare_data()
    train_loader = data_module.dataloader("train")
    val_loader = data_module.dataloader("val")

    return train_loader, val_loader

def train_pipeline():

    config = load_config()
    device = config.resolved_device()

    train_loader, val_loader = load_data(config)

    model_factory: Callable[[], nn.Module] = CNN

    model = model_factory()
    evaluator = BallLocalizationEvaluator(config.thresholds)
    trainer = BallLocalizationTrainer(
        model=model,
        evaluator=evaluator,
        device=device,
        learning_rate=config.learning_rate,
    )
    checkpoint = ModelCheckpoint(config.checkpoint_path)

    model, history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        validation_interval=config.validation_interval,
    )


