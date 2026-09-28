import random

import matplotlib.pyplot as plt
import numpy as np

from dataset.augmentation.config import MotionBlurConfig
from dataset.augmentation.context import AugmentationContext
from dataset.augmentation.motion_blur import RandomMotionBlur


def plot_motion_blur_preview(
    image: np.ndarray,
    config: MotionBlurConfig,
    variants: int = 4,
    seed: int | None = None,
):
    if variants < 1:
        raise ValueError("variants must be at least 1.")

    augmentation = RandomMotionBlur(config, rng=random.Random(seed))
    figure, axes = plt.subplots(
        1,
        variants + 1,
        figsize=(5 * (variants + 1), 5),
        constrained_layout=True,
    )

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for axis in axes[1:]:
        parameters = augmentation.sample_parameters()
        augmented_image = augmentation.apply(image, parameters)
        axis.imshow(augmented_image)
        axis.set_title(
            f"kernel={parameters.kernel_size}px"
            + chr(10)
            + f"angle={parameters.angle_degrees:.1f}°"
        )
        axis.axis("off")

    plt.show()
    return figure


def plot_trajectory_motion_blur_preview(
    image: np.ndarray,
    context: AugmentationContext,
    config: MotionBlurConfig,
    seed: int | None = None,
):
    augmentation = RandomMotionBlur(config, rng=random.Random(seed))
    parameters = augmentation.select_parameters(context)
    augmented_image = (
        image
        if parameters is None
        else augmentation.apply(image, parameters)
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        constrained_layout=True,
    )
    axes[0].imshow(image)
    axes[0].set_title("Original frame t")
    axes[0].axis("off")

    axes[1].imshow(image)
    centers = (
        ("t-1", context.previous_center),
        ("t", context.center),
        ("t+1", context.next_center),
    )
    available_centers = [
        (label, center)
        for label, center in centers
        if center is not None
    ]
    if available_centers:
        x_coordinates = [center[0] for _, center in available_centers]
        y_coordinates = [center[1] for _, center in available_centers]
        axes[1].plot(
            x_coordinates,
            y_coordinates,
            "o-",
            color="cyan",
            linewidth=2,
        )
        for label, center in available_centers:
            axes[1].annotate(
                label,
                center,
                color="yellow",
                xytext=(5, 5),
                textcoords="offset points",
            )
    axes[1].set_title("Annotated trajectory")
    axes[1].axis("off")

    axes[2].imshow(augmented_image)
    if parameters is None:
        details = "blur skipped: displacement below threshold"
    else:
        displacement = (
            f" · displacement={parameters.displacement:.1f}px"
            if parameters.displacement is not None
            else ""
        )
        details = (
            f"{parameters.source} · angle={parameters.angle_degrees:.1f}°"
            + chr(10)
            + f"kernel={parameters.kernel_size}px{displacement}"
        )
    axes[2].set_title("Augmented frame" + chr(10) + details)
    axes[2].axis("off")

    plt.show()
    return figure
