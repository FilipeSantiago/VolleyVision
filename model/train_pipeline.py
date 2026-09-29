import json
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import torch
from dotenv import load_dotenv
from torch import nn

from dataset.augmentation import MotionBlurConfig, AugmentationConfig
from dataset.viz.ball_localization_visualizer import BallLocalizationVisualizer
from dataset.viz.training_curves_visualizer import TrainingCurvesVisualizer
from model.artifacts import (
    ArtifactConfig,
    export_cnn_onnx,
    rank_test_videos,
    render_domain_shift_video,
    render_test_video,
    sha256_file,
)
from model.cnn import CNN
from model.engine.ball_data_module import BallDataModule
from model.engine.ball_localization_trainer import BallLocalizationTrainer
from model.engine.early_stopping import EarlyStopping
from model.engine.experiment_config import ExperimentConfig
from model.engine.model_checkpoint import ModelCheckpoint
from model.eval.ball_location_evaluator import BallLocalizationEvaluator
from monitoring import MLflowTracker


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
        artifacts=ArtifactConfig(
            domain_shift_video_path=Path(
                "/home/bugslayer/Documents/research/VolleyVision/domain_shift/"
                "videoplayback.1773931843922.publer.com-00.00.00.000-"
                "00.10.03.976-00.06.08.875-00.10.04.033__seg001__0002.35-0010.82.mp4"
            ),
            excluded_regions=(
                ("scoreboard", (0.26, 0.79, 0.74, 0.97)),
            ),
        ),
    )

    return config


def train_pipeline(
        config: ExperimentConfig | None = None,
        run_name: str | None = None,
):
    # MLflow settings (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME) live in .env;
    # variables already set in the environment take precedence.
    load_dotenv()
    config = config or load_config()
    # Fail fast on a missing or unreachable server before preparing data.
    tracker = MLflowTracker(config.mlflow).connect()
    device = config.resolved_device()

    with tracker.start_run(run_name):
        tracker.log_config(config)
        tracker.set_run_tags(device=str(device))

        data_module = BallDataModule(config).prepare_data()
        tracker.log_table(
            data_module.split_summary().reset_index(),
            "data/split_summary.json",
        )
        tracker.log_text(
            data_module.frame_table("val_small").to_csv(index=False),
            "data/val_small_frames.csv",
        )

        model_factory: Callable[[], nn.Module] = CNN

        model = model_factory()
        tracker.log_param(
            "n_params", sum(parameter.numel() for parameter in model.parameters())
        )
        tracker.log_text(str(model), "model/architecture.txt")

        evaluator = BallLocalizationEvaluator(config.thresholds)
        trainer = BallLocalizationTrainer(
            model=model,
            evaluator=evaluator,
            device=device,
            learning_rate=config.learning_rate,
        )
        checkpoint = ModelCheckpoint(config.checkpoint_path)
        early_stopping = EarlyStopping(
            monitor=config.early_stopping_monitor,
            mode=config.early_stopping_mode,
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

        model, history = trainer.fit(
            train_loader=data_module.dataloader("train"),
            val_loader=data_module.dataloader("val_small"),
            epochs=config.epochs,
            full_val_loader=data_module.dataloader("val"),
            full_validation_interval=config.validation_interval,
            early_stopping=early_stopping,
            on_epoch_end=tracker.epoch_logger(),
        )

        tracker.log_metrics(
            {
                "best_epoch": early_stopping.best_epoch,
                "stopped_epoch": trainer.stopped_epoch or len(history),
                f"best_{early_stopping.monitor}": early_stopping.best_value,
            }
        )
        history_path = config.checkpoint_path.with_name(
            f"{config.checkpoint_path.stem}_history.csv"
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_path, index=False)
        tracker.log_artifact(history_path, "training")
        tracker.log_figure(
            TrainingCurvesVisualizer().plot(history),
            "training/curves.png",
        )

        # The trainer restored the best early-stopping weights.
        saved_checkpoint = checkpoint.save(model)
        print(f"Saved checkpoint to {saved_checkpoint}")
        tracker.log_checkpoint(saved_checkpoint)

        test_model = checkpoint.load(model_factory(), device)
        with TemporaryDirectory(prefix="volleyvision-model-") as temp_dir:
            architecture_path = export_cnn_onnx(
                model=test_model,
                output_path=Path(temp_dir) / "architecture.onnx",
                image_width=config.image_width,
                image_height=config.image_height,
            )
            tracker.log_artifact(architecture_path, "model")
        tracker.set_run_tags(architecture_artifact="model/architecture.onnx")

        test_metrics, test_predictions = evaluator.evaluate(
            model=test_model,
            data_loader=data_module.dataloader("test"),
            device=device,
            frame_table=data_module.frame_table("test"),
            description="Testing",
        )
        if test_predictions is None:
            raise RuntimeError("Test evaluation did not return per-frame predictions.")
        tracker.log_metrics(
            {f"test_{name}": value for name, value in test_metrics.items()}
        )
        print(pd.Series(test_metrics, name="test").to_string())

        tracker.log_table(test_predictions, "test/predictions.json")
        worst_predictions = (
            test_predictions
            .dropna(subset=["distance_px"])
            .nlargest(10, "distance_px")
        )
        tracker.log_table(worst_predictions, "test/worst_predictions.json")
        errors_figure = BallLocalizationVisualizer().plot_errors(
            dataset=data_module.dataset("test"),
            predictions=test_predictions,
            count=6,
            columns=3,
        )
        tracker.log_figure(errors_figure, "test/errors.png")

        _log_video_artifacts(
            tracker=tracker,
            config=config,
            model=test_model,
            device=device,
            test_frame_table=data_module.frame_table("test"),
            test_predictions=test_predictions,
        )

    return model, history


def _log_video_artifacts(
        tracker: MLflowTracker,
        config: ExperimentConfig,
        model: nn.Module,
        device: torch.device,
        test_frame_table: pd.DataFrame,
        test_predictions: pd.DataFrame,
) -> None:
    artifact_config = config.artifacts
    ranking = rank_test_videos(
        test_predictions,
        thresholds=config.thresholds,
        confidence_threshold=artifact_config.confidence_threshold,
    )
    tracker.log_table(ranking, "test/video_ranking.json")

    best = ranking.iloc[0]
    best_video_id = str(best["video_id"])
    tracker.set_run_tags(best_test_video_id=best_video_id)
    tracker.log_metrics(
        {
            "best_test_video_pck_10": float(best["pck_10"]),
            "best_test_video_mean_distance_px": float(best["mean_distance_px"]),
            "best_test_video_median_distance_px": float(best["median_distance_px"]),
        }
    )

    with TemporaryDirectory(prefix="volleyvision-artifacts-") as temp_dir:
        staging_dir = Path(temp_dir)
        best_video_path = staging_dir / "test_best_annotated.mp4"
        best_summary = render_test_video(
            frame_table=test_frame_table,
            predictions=test_predictions,
            video_id=best_video_id,
            output_path=best_video_path,
            confidence_threshold=artifact_config.confidence_threshold,
            codec=artifact_config.video_codec,
            fallback_fps=artifact_config.fallback_fps,
        )
        best_summary.update(
            {
                key: _json_scalar(best[key])
                for key in best.index
                if key != "video_id"
            }
        )
        best_summary["checkpoint_sha256"] = sha256_file(config.checkpoint_path)
        best_predictions = test_predictions.loc[
            test_predictions["video_id"].eq(best_video_id)
        ]
        tracker.log_artifact(best_video_path, "test/best_video")
        tracker.log_text(
            best_predictions.to_csv(index=False),
            "test/best_video/predictions.csv",
        )
        tracker.log_text(
            json.dumps(best_summary, indent=2, sort_keys=True),
            "test/best_video/summary.json",
        )

        domain_path = artifact_config.domain_shift_video_path
        if domain_path is None:
            return

        domain_video_path = staging_dir / "domain_shift_annotated.mp4"
        domain_predictions, domain_summary = render_domain_shift_video(
            model=model,
            device=device,
            video_path=domain_path,
            output_path=domain_video_path,
            model_width=config.image_width,
            model_height=config.image_height,
            confidence_threshold=artifact_config.confidence_threshold,
            excluded_regions=artifact_config.excluded_regions,
            codec=artifact_config.video_codec,
            fallback_fps=artifact_config.fallback_fps,
        )
        domain_summary["checkpoint_sha256"] = sha256_file(config.checkpoint_path)
        domain_sha256 = domain_summary["source_sha256"]
        if not isinstance(domain_sha256, str):
            raise TypeError("Domain-shift source hash must be a string.")
        tracker.set_run_tags(
            domain_shift_video=domain_path.name,
            domain_shift_sha256=domain_sha256,
        )
        tracker.log_artifact(domain_video_path, "domain_shift")
        tracker.log_text(
            domain_predictions.to_csv(index=False),
            "domain_shift/predictions.csv",
        )
        tracker.log_text(
            json.dumps(domain_summary, indent=2, sort_keys=True),
            "domain_shift/summary.json",
        )


def _json_scalar(value):
    value = value.item() if hasattr(value, "item") else value
    return None if pd.isna(value) else value


if __name__ == "__main__":
    train_pipeline(run_name="baseline-cnn")
