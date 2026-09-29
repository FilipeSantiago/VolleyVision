import onnx

from model.artifacts import export_cnn_onnx
from model.cnn import CNN


def test_export_cnn_onnx_writes_valid_graph(tmp_path):
    output_path = export_cnn_onnx(
        model=CNN(),
        output_path=tmp_path / "architecture.onnx",
        image_width=64,
        image_height=32,
    )

    graph = onnx.load(output_path).graph

    assert output_path.is_file()
    assert graph.input[0].name == "rgb_frame"
    assert graph.output[0].name == "ball_heatmap"
    assert [
        dimension.dim_value
        for dimension in graph.input[0].type.tensor_type.shape.dim
    ] == [1, 3, 32, 64]
    assert [
        dimension.dim_value
        for dimension in graph.output[0].type.tensor_type.shape.dim
    ] == [1, 1, 32, 64]
