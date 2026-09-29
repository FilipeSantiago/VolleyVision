from dataclasses import dataclass
from pathlib import Path

NormalizedRegion = tuple[float, float, float, float]
NamedRegion = tuple[str, NormalizedRegion]


@dataclass(frozen=True)
class ArtifactConfig:
    domain_shift_video_path: Path | None = None
    confidence_threshold: float = 0.10
    excluded_regions: tuple[NamedRegion, ...] = ()
    video_codec: str = "mp4v"
    fallback_fps: float = 30.0
