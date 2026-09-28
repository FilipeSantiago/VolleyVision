import math
import random
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from dataset.augmentation.config import MotionBlurConfig
from dataset.augmentation.context import AugmentationContext


@dataclass(frozen=True)
class MotionBlurParameters:
    kernel_size: int
    angle_degrees: float
    source: Literal["trajectory", "random"] = "random"
    displacement: float | None = None


class RandomMotionBlur:
    def __init__(
        self,
        config: MotionBlurConfig,
        rng: random.Random | None = None,
    ):
        self.config = config
        self.rng = rng

    def __call__(
        self,
        image: np.ndarray,
        context: AugmentationContext | None = None,
    ) -> np.ndarray:
        self._validate_image(image)
        rng = self.rng or random
        if rng.random() >= self.config.probability:
            return image

        parameters = self.select_parameters(context)
        if parameters is None:
            return image
        return self.apply(image, parameters)

    def sample_parameters(self) -> MotionBlurParameters:
        return self.sample_random_parameters()

    def sample_random_parameters(self) -> MotionBlurParameters:
        rng = self.rng or random
        kernel_sizes = self.config.odd_kernel_sizes
        minimum = kernel_sizes[0]
        weights = [
            (minimum / kernel_size) ** self.config.strength_bias
            for kernel_size in kernel_sizes
        ]
        kernel_size = rng.choices(kernel_sizes, weights=weights, k=1)[0]
        angle_degrees = rng.uniform(0.0, 180.0)
        return MotionBlurParameters(
            kernel_size=kernel_size,
            angle_degrees=angle_degrees,
        )

    def select_parameters(
        self,
        context: AugmentationContext | None,
    ) -> MotionBlurParameters | None:
        if context is None or not context.has_complete_trajectory:
            return self.sample_random_parameters()
        return self._trajectory_parameters(context)

    def _trajectory_parameters(
        self,
        context: AugmentationContext,
    ) -> MotionBlurParameters | None:
        previous_x, previous_y = context.previous_center
        next_x, next_y = context.next_center
        delta_x = next_x - previous_x
        delta_y = next_y - previous_y
        displacement = math.hypot(delta_x, delta_y)

        minimum_displacement, maximum_displacement = (
            self.config.trajectory_displacement_range
        )
        if displacement <= minimum_displacement:
            return None

        normalized_displacement = min(
            (displacement - minimum_displacement)
            / (maximum_displacement - minimum_displacement),
            1.0,
        )
        kernel_sizes = self.config.odd_kernel_sizes
        kernel_index = round(
            normalized_displacement * (len(kernel_sizes) - 1)
        )
        angle_degrees = math.degrees(
            math.atan2(delta_y, delta_x)
        ) % 180.0
        return MotionBlurParameters(
            kernel_size=kernel_sizes[kernel_index],
            angle_degrees=angle_degrees,
            source="trajectory",
            displacement=displacement,
        )

    def apply(
        self,
        image: np.ndarray,
        parameters: MotionBlurParameters,
    ) -> np.ndarray:
        self._validate_image(image)
        kernel = self._build_kernel(parameters)
        return cv2.filter2D(
            image,
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )

    @staticmethod
    def _build_kernel(parameters: MotionBlurParameters) -> np.ndarray:
        kernel_size = parameters.kernel_size
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Motion-blur kernel size must be a positive odd number.")

        center = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[center, :] = 1.0
        rotation = cv2.getRotationMatrix2D(
            (center, center),
            -parameters.angle_degrees,
            1.0,
        )
        kernel = cv2.warpAffine(
            kernel,
            rotation,
            (kernel_size, kernel_size),
            flags=cv2.INTER_LINEAR,
        )
        kernel_sum = float(kernel.sum())
        if kernel_sum <= 0:
            raise RuntimeError("Motion-blur kernel has no non-zero weights.")
        return kernel / kernel_sum

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("Motion blur expects an image as a NumPy array.")
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Expected a 2D or 3D image, received shape {image.shape}."
            )
