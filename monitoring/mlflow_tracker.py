import dataclasses
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import mlflow
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from monitoring.mlflow_config import MLflowConfig


class MLflowTracker:
    def __init__(self, config: MLflowConfig):
        self.config = config

    def connect(self) -> "MLflowTracker":
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if not experiment:
            raise RuntimeError(
                "No MLflow experiment configured. Set MLFLOW_EXPERIMENT_NAME in .env "
                "or the environment."
            )
        tracking_uri = self.config.resolved_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            with self._fast_failing_http():
                mlflow.search_experiments(max_results=1)
        except Exception as error:
            raise RuntimeError(
                f"MLflow tracking server at {tracking_uri} is unreachable: {error}"
            ) from error
        mlflow.set_experiment(experiment)
        return self

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Iterator[mlflow.ActiveRun]:
        with mlflow.start_run(run_name=run_name) as run:
            yield run

    def log_config(self, config: Any) -> None:
        mlflow.log_params(self.flatten_config(config))

    def set_run_tags(self, **extra: str) -> None:
        mlflow.set_tags({**self.git_tags(), **extra})

    def log_metrics(
        self,
        metrics: Mapping[str, float | None],
        step: int | None = None,
    ) -> None:
        mlflow.log_metrics(
            {
                name: value
                for name, value in metrics.items()
                if name != "epoch" and value is not None
            },
            step=step,
        )

    def epoch_logger(self) -> Callable[[int, dict], None]:
        return lambda epoch, row: self.log_metrics(row, step=epoch)

    def log_param(self, name: str, value: Any) -> None:
        mlflow.log_param(name, value)

    def log_table(self, table: pd.DataFrame, path: str) -> None:
        mlflow.log_table(table, path)

    def log_text(self, text: str, path: str) -> None:
        mlflow.log_text(text, path)

    def log_artifact(self, path: Path | str, artifact_dir: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_dir)

    def log_figure(self, figure: Figure, path: str) -> None:
        mlflow.log_figure(figure, path)
        plt.close(figure)

    def log_checkpoint(self, path: Path | str) -> None:
        try:
            self.log_artifact(path, "checkpoint")
        except Exception as error:
            raise RuntimeError(
                "Uploading the checkpoint through the MLflow server failed "
                "(check the server's S3 credentials and bucket permissions). "
                f"The local copy is kept at {path}."
            ) from error

    @staticmethod
    def flatten_config(config: Any) -> dict[str, str]:
        def flatten(value, prefix: str) -> dict[str, str]:
            if isinstance(value, dict):
                flat = {}
                for key, item in value.items():
                    flat.update(flatten(item, f"{prefix}.{key}" if prefix else key))
                return flat
            return {prefix: str(value)}

        return flatten(dataclasses.asdict(config), "")

    @staticmethod
    def git_tags() -> dict[str, str]:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()

        try:
            return {
                "git_sha": git("rev-parse", "HEAD"),
                "git_dirty": str(bool(git("status", "--porcelain"))),
            }
        except (OSError, subprocess.CalledProcessError):
            return {"git_sha": "unknown", "git_dirty": "unknown"}

    @staticmethod
    @contextmanager
    def _fast_failing_http():
        # MLflow retries with exponential backoff by default, which turns a wrong
        # URI into minutes of waiting before the reachability check fails.
        overrides = {
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
        }
        previous = {name: os.environ.get(name) for name in overrides}
        os.environ.update(overrides)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
