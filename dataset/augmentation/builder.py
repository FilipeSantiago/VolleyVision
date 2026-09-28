from collections.abc import Callable

import numpy as np

from dataset.augmentation.config import AugmentationConfig, MotionBlurConfig
from dataset.augmentation.context import AugmentationContext
from dataset.augmentation.motion_blur import RandomMotionBlur

ImageAugmentation = Callable[
    [np.ndarray, AugmentationContext | None],
    np.ndarray,
]


class AugmentationPipeline:
    def __init__(self, augmentations: tuple[ImageAugmentation, ...]):
        self.augmentations = augmentations

    def __call__(
        self,
        image: np.ndarray,
        context: AugmentationContext | None = None,
    ) -> np.ndarray:
        original_shape = image.shape
        augmented_image = image

        for augmentation in self.augmentations:
            augmented_image = augmentation(augmented_image, context)
            if augmented_image.shape != original_shape:
                raise ValueError(
                    "Image augmentations must preserve dimensions: "
                    f"{original_shape} != {augmented_image.shape}."
                )

        return augmented_image


class AugmentationBuilder:
    def __init__(self):
        self._augmentations: list[ImageAugmentation] = []

    def add(self, augmentation: ImageAugmentation) -> "AugmentationBuilder":
        self._augmentations.append(augmentation)
        return self

    def add_motion_blur(
        self,
        config: MotionBlurConfig,
    ) -> "AugmentationBuilder":
        return self.add(RandomMotionBlur(config))

    def build(self) -> AugmentationPipeline:
        return AugmentationPipeline(tuple(self._augmentations))

    @classmethod
    def from_config(
        cls,
        config: AugmentationConfig,
    ) -> AugmentationPipeline:
        builder = cls()
        if config.motion_blur is not None:
            builder.add_motion_blur(config.motion_blur)
        return builder.build()
