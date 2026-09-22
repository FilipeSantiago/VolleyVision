import torch


class HeatmapDecoder:
    @staticmethod
    def peak_coordinates(
        heatmaps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if heatmaps.ndim != 4 or heatmaps.shape[1] != 1:
            raise ValueError(
                "Expected heatmaps with shape [batch, 1, height, width], "
                f"received {tuple(heatmaps.shape)}."
            )

        batch_size, _, _, width = heatmaps.shape
        flattened = heatmaps[:, 0].reshape(batch_size, -1)
        peak_values, peak_indices = flattened.max(dim=1)
        y_coordinates = torch.div(
            peak_indices,
            width,
            rounding_mode="floor",
        )
        x_coordinates = peak_indices.remainder(width)
        coordinates = torch.stack(
            (x_coordinates, y_coordinates),
            dim=1,
        )
        return coordinates, peak_values
