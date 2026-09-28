from dataclasses import dataclass


@dataclass(frozen=True)
class MotionBlurConfig:
    probability: float = 0.3
    kernel_size_range: tuple[int, int] = (3, 15)
    strength_bias: float = 1.5
    trajectory_displacement_range: tuple[float, float] = (1.0, 30.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")

        minimum, maximum = self.kernel_size_range
        if minimum < 1 or maximum < 1:
            raise ValueError("Motion-blur kernel sizes must be positive.")
        if minimum > maximum:
            raise ValueError("kernel_size_range must be ordered from low to high.")
        if not any(size % 2 == 1 for size in range(minimum, maximum + 1)):
            raise ValueError("kernel_size_range must contain an odd kernel size.")
        if self.strength_bias <= 0:
            raise ValueError("strength_bias must be positive.")

        minimum_displacement, maximum_displacement = (
            self.trajectory_displacement_range
        )
        if minimum_displacement < 0:
            raise ValueError("Minimum trajectory displacement cannot be negative.")
        if minimum_displacement >= maximum_displacement:
            raise ValueError(
                "trajectory_displacement_range must have increasing values."
            )

    @property
    def odd_kernel_sizes(self) -> tuple[int, ...]:
        minimum, maximum = self.kernel_size_range
        return tuple(
            size
            for size in range(minimum, maximum + 1)
            if size % 2 == 1
        )


@dataclass(frozen=True)
class AugmentationConfig:
    motion_blur: MotionBlurConfig | None = None
