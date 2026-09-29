import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str | None = None

    def resolved_tracking_uri(self) -> str:
        tracking_uri = self.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            # Without a URI MLflow silently writes to ./mlruns instead of the server.
            raise RuntimeError(
                "No MLflow tracking server configured. Set the MLFLOW_TRACKING_URI "
                "environment variable (e.g. http://<server>:5000) or "
                "ExperimentConfig.mlflow.tracking_uri."
            )
        return tracking_uri
