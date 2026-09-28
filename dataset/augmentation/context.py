from dataclasses import dataclass

BallCenter = tuple[float, float]


@dataclass(frozen=True)
class AugmentationContext:
    video_id: str
    frame_number: int
    previous_center: BallCenter | None
    center: BallCenter | None
    next_center: BallCenter | None

    @property
    def has_complete_trajectory(self) -> bool:
        return all(
            center is not None
            for center in (
                self.previous_center,
                self.center,
                self.next_center,
            )
        )
