from typing import Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        validation_interval: int = 1,
    ) -> tuple[nn.Module, pd.DataFrame]:
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if validation_interval <= 0:
            raise ValueError("validation_interval must be positive.")

        history_rows = []
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch, epochs)
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
            }

            should_validate = (
                epoch == 1
                or epoch % validation_interval == 0
                or epoch == epochs
            )
            if should_validate:
                validation_metrics, _ = self.evaluator.evaluate(
                    model=self.model,
                    data_loader=val_loader,
                    device=self.device,
                    description=f"Validation {epoch}/{epochs}",
                )
                history_row.update(
                    {
                        "val_loss": validation_metrics["heatmap_mse"],
                        "mean_distance_px": validation_metrics[
                            "mean_distance_px"
                        ],
                        "median_distance_px": validation_metrics[
                            "median_distance_px"
                        ],
                        **{
                            f"pck_{threshold}": validation_metrics[
                                f"pck_{threshold}"
                            ]
                            for threshold in self.evaluator.thresholds
                        },
                    }
                )
                self._print_validation(epoch, epochs, history_row)

            history_rows.append(history_row)

        self.history = pd.DataFrame(history_rows)
        return self.model, self.history

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
        pck_summary = " | ".join(
            f"PCK@{threshold}={metrics[f'pck_{threshold}']:.3f}"
            for threshold in self.evaluator.thresholds
        )
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={metrics['train_loss']:.6f} | "
            f"val_loss={metrics['val_loss']:.6f} | "
            f"mean_dist={metrics['mean_distance_px']:.2f}px | "
            f"median_dist={metrics['median_distance_px']:.2f}px | "
            f"{pck_summary}"
        )
