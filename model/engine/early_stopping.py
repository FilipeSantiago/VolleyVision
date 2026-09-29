import copy
import math

from torch import nn


class EarlyStopping:
    MODES = ("min", "max")

    def __init__(
        self,
        monitor: str,
        mode: str = "max",
        patience: int = 8,
        min_delta: float = 0.0,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")
        if patience < 0:
            raise ValueError("patience cannot be negative.")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.best_state_dict: dict | None = None
        self.epochs_without_improvement = 0

    def step(self, epoch: int, metrics: dict, model: nn.Module) -> bool:
        if self.monitor not in metrics:
            raise KeyError(
                f"Early-stopping metric {self.monitor!r} is missing; "
                f"available metrics: {sorted(metrics)}."
            )
        value = float(metrics[self.monitor])

        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore(self, model: nn.Module) -> nn.Module:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        return model

    def _is_improvement(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta
