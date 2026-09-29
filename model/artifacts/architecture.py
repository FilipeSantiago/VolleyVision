import copy
from pathlib import Path
from typing import Sequence

import onnx
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from torch import nn


def export_cnn_onnx(
        model: nn.Module,
        output_path: Path | str,
        image_width: int,
        image_height: int,
) -> Path:
    """Export and validate the executable CNN graph used by Netron."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("ONNX input dimensions must be positive.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu().eval()
    example_input = torch.zeros(1, 3, image_height, image_width)
    with torch.inference_mode():
        onnx_program = torch.onnx.export(
            export_model,
            (example_input,),
            dynamo=True,
            input_names=["rgb_frame"],
            output_names=["ball_heatmap"],
            opset_version=18,
        )
    onnx_program.save(output_path)

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    input_shape = _tensor_shape(onnx_model.graph.input[0])
    output_shape = _tensor_shape(onnx_model.graph.output[0])
    expected_input = [1, 3, image_height, image_width]
    expected_output = [1, 1, image_height, image_width]
    if input_shape != expected_input or output_shape != expected_output:
        raise RuntimeError(
            "Unexpected ONNX graph boundary shapes: "
            f"input={input_shape}, output={output_shape}."
        )
    return output_path


def _tensor_shape(value_info: onnx.ValueInfoProto) -> list[int]:
    return [
        dimension.dim_value
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def plot_cnn_architecture(
        image_width: int,
        image_height: int,
        sigma: float = 3.0,
        learning_rate: float = 1e-3,
        thresholds: Sequence[int] = (5, 10, 20),
) -> Figure:
    stages = [
        ("RGB frame", f"3 × {image_height} × {image_width}", "#d9eaf7"),
        ("Encoder 1", f"Conv 3→32→32\n32 × {image_height // 2} × {image_width // 2}", "#a9d3f0"),
        ("Encoder 2", f"Conv 32→64→64\n64 × {image_height // 4} × {image_width // 4}", "#75b8e5"),
        ("Bottleneck", f"Conv 64→128→128\n128 × {image_height // 8} × {image_width // 8}", "#438fc4"),
        ("Decoder 1", f"Up 128→64 + Conv\n64 × {image_height // 4} × {image_width // 4}", "#8bc9a8"),
        ("Decoder 2", f"Up 64→32 + Conv\n32 × {image_height // 2} × {image_width // 2}", "#61b98d"),
        ("Decoder 3", f"Up 32→16 + Conv\n16 × {image_height} × {image_width}", "#3b9f72"),
        ("Heatmap", f"1 × {image_height} × {image_width}\n1×1 convolution", "#f4ca64"),
    ]

    figure, axis = plt.subplots(figsize=(20, 8), constrained_layout=True)
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 8)
    axis.axis("off")

    box_width = 2.1
    box_height = 2.0
    gap = 0.32
    start_x = 0.35
    y = 4.1

    for index, (title, details, color) in enumerate(stages):
        x = start_x + index * (box_width + gap)
        box = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#203040",
            linewidth=1.5,
        )
        axis.add_patch(box)
        axis.text(
            x + box_width / 2,
            y + 1.35,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        axis.text(
            x + box_width / 2,
            y + 0.65,
            details,
            ha="center",
            va="center",
            fontsize=8.5,
        )
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(x + box_width + gap - 0.05, y + box_height / 2),
                xytext=(x + box_width + 0.05, y + box_height / 2),
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#203040"},
            )

    axis.text(
        10,
        7.15,
        "VolleyVision ball-localization architecture",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
    )
    axis.text(
        10,
        6.65,
        "Three max-pooling stages · three learned upsampling stages · no skip connections",
        ha="center",
        va="center",
        fontsize=11,
        color="#405060",
    )

    training_text = (
        "TRAINING\n"
        f"Gaussian target (σ = {sigma:g})  →  MSE loss  →  Adam (lr = {learning_rate:g})\n"
        "Motion-blur augmentation · "
        + " / ".join(f"PCK@{threshold}" for threshold in thresholds)
    )
    inference_text = (
        "INFERENCE\n"
        "Optional excluded-region suppression  →  heatmap argmax  →  confidence threshold\n"
        "Scale (x, y) back to the source frame  →  annotated video + per-frame table"
    )
    axis.text(
        0.55,
        2.55,
        training_text,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f5f7fa", "edgecolor": "#9aa8b5"},
    )
    axis.text(
        10.15,
        2.55,
        inference_text,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f5f7fa", "edgecolor": "#9aa8b5"},
    )
    return figure
