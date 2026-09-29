from typing import Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.engine.early_stopping import EarlyStopping
from model.eval.ball_location_evaluator import BallLocalizationEvaluator


class BallLocalizationTrainer:
    def __init__(
        self,
        model: nn.Module,
        evaluator: BallLocalizationEvaluator,
        device: torch.device,
        learning_rate: float = 1e-3,
        criterion: nn.Module | None = None,
        optimizer_factory: Callable | None = None,
    ):
        self.model = model.to(device)
        self.evaluator = evaluator
        self.device = device
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = (
            optimizer_factory(self.model.parameters())
            if optimizer_factory is not None
            else torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )
        )
        self.history = pd.DataFrame()
        self.stopped_epoch: int | None = None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        full_val_loader: DataLoader | None = None,
        full_validation_interval: int = 5,
        early_stopping: EarlyStopping | None = None,
        on_epoch_end: Callable[[int, dict], None] | None = None,
    ) -> tuple[nn.Module, pd.DataFrame]:
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if full_validation_interval <= 0:
            raise ValueError("full_validation_interval must be positive.")

        self.stopped_epoch = None
        history_rows = []
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch, epochs)
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **self._validate(val_loader, "val_small", epoch, epochs),
            }

            ran_full_validation = full_val_loader is not None and (
                epoch == 1
                or epoch % full_validation_interval == 0
                or epoch == epochs
            )
            if ran_full_validation:
                history_row.update(
                    self._validate(full_val_loader, "val_full", epoch, epochs)
                )

            should_stop = early_stopping is not None and early_stopping.step(
                epoch, history_row, self.model
            )
            if (
                should_stop
                and full_val_loader is not None
                and not ran_full_validation
            ):
                history_row.update(
                    self._validate(full_val_loader, "val_full", epoch, epochs)
                )

            self._print_validation(epoch, epochs, history_row)
            history_rows.append(history_row)
            if on_epoch_end is not None:
                on_epoch_end(epoch, history_row)

            if should_stop:
                self.stopped_epoch = epoch
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"{early_stopping.monitor} did not improve for "
                    f"{early_stopping.patience} epochs "
                    f"(best={early_stopping.best_value:.4f} "
                    f"at epoch {early_stopping.best_epoch})."
                )
                break

        if early_stopping is not None:
            early_stopping.restore(self.model)

        self.history = pd.DataFrame(history_rows)
        return self.model, self.history

    def _validate(
        self,
        data_loader: DataLoader,
        prefix: str,
        epoch: int,
        epochs: int,
    ) -> dict[str, float]:
        metrics, _ = self.evaluator.evaluate(
            model=self.model,
            data_loader=data_loader,
            device=self.device,
            description=f"Validation ({prefix}) {epoch}/{epochs}",
        )
        return self._prefixed(metrics, prefix)

    def _prefixed(
        self,
        metrics: dict[str, float | int],
        prefix: str,
    ) -> dict[str, float]:
        metric_names = [
            "heatmap_mse",
            "mean_distance_px",
            "median_distance_px",
            *(f"pck_{threshold}" for threshold in self.evaluator.thresholds),
        ]
        return {
            f"{prefix}_{name}": metrics[name]
            for name in metric_names
        }

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        epochs: int,
    ) -> float:
        self.model.train()
        loss_sum = 0.0
        example_count = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
        )

        for batch in progress:
            images = batch["image"].to(
                self.device,
                non_blocking=True,
            )
            targets = batch["target"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = images.shape[0]
            loss_sum += loss.detach().item() * batch_size
            example_count += batch_size
            progress.set_postfix(loss=f"{loss.item():.6f}")

        if example_count == 0:
            raise ValueError("The training loader produced no examples.")
        return loss_sum / example_count

    def _print_validation(
        self,
        epoch: int,
        epochs: int,
        metrics: dict[str, float],
    ) -> None:
        summaries = [
            f"Epoch {epoch}/{epochs}",
            f"train_loss={metrics['train_loss']:.6f}",
        ]
        for prefix in ("val_small", "val_full"):
            if f"{prefix}_heatmap_mse" in metrics:
                summaries.append(self._validation_summary(metrics, prefix))
        print(" | ".join(summaries))

    def _validation_summary(
        self,
        metrics: dict[str, float],
        prefix: str,
    ) -> str:
        pck_summary = " ".join(
            f"PCK@{threshold}={metrics[f'{prefix}_pck_{threshold}']:.3f}"
            for threshold in self.evaluator.thresholds
        )
        return (
            f"{prefix}: "
            f"mse={metrics[f'{prefix}_heatmap_mse']:.6f} "
            f"mean_dist={metrics[f'{prefix}_mean_distance_px']:.2f}px "
            f"median_dist={metrics[f'{prefix}_median_distance_px']:.2f}px "
            f"{pck_summary}"
        )
