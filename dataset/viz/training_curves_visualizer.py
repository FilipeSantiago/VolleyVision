import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


class TrainingCurvesVisualizer:
    SERIES_COLORS = {
        "train": "#1baf7a",
        "val_small": "#2a78d6",
        "val_full": "#eb6834",
    }

    def plot(self, history: pd.DataFrame, pck_threshold: int = 10) -> Figure:
        figure, (loss_axis, pck_axis) = plt.subplots(
            1,
            2,
            figsize=(14, 5),
            constrained_layout=True,
        )

        loss_axis.plot(
            history["epoch"],
            history["train_loss"],
            color=self.SERIES_COLORS["train"],
            linewidth=2,
            label="train",
        )
        for prefix in ("val_small", "val_full"):
            self._plot_validation_series(
                loss_axis, history, prefix, f"{prefix}_heatmap_mse"
            )
            self._plot_validation_series(
                pck_axis, history, prefix, f"{prefix}_pck_{pck_threshold}"
            )

        loss_axis.set_title("Heatmap MSE")
        loss_axis.set_yscale("log")
        pck_axis.set_title(f"PCK@{pck_threshold}")
        pck_axis.set_ylim(0.0, 1.0)
        for axis in (loss_axis, pck_axis):
            axis.set_xlabel("epoch")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.3)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend()
        return figure

    def _plot_validation_series(
        self,
        axis,
        history: pd.DataFrame,
        prefix: str,
        column: str,
    ):
        if column not in history:
            return
        points = history[["epoch", column]].dropna()
        axis.plot(
            points["epoch"],
            points[column],
            color=self.SERIES_COLORS[prefix],
            linewidth=2,
            marker="o" if prefix == "val_full" else None,
            markersize=8,
            label=prefix,
        )
