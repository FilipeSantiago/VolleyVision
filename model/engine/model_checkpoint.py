from pathlib import Path

import torch
from torch import nn


class ModelCheckpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, model: nn.Module) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        return self.path

    def load(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> nn.Module:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.path}. Train and save the model first."
            )
        state_dict = torch.load(
            self.path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model.to(device)
