import onnx
from onnx import helper, TensorProto, onnx_pb

# ZipMap with string class labels: takes float tensor [1, 3], outputs seq(map(string, float))
zipmap = helper.make_node("ZipMap", ["x"], ["y"], domain="ai.onnx.ml", classlabels_strings=["a", "b", "c"])

# Build output type: seq(map(string, float))
output = onnx_pb.ValueInfoProto()
output.name = "y"
output.type.sequence_type.elem_type.map_type.key_type = TensorProto.STRING
output.type.sequence_type.elem_type.map_type.value_type.tensor_type.elem_type = TensorProto.FLOAT

graph = helper.make_graph(
    [zipmap],
    "zipmap_string",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
    [output],
)
model = helper.make_model(graph, opset_imports=[
    helper.make_opsetid("", 13),
    helper.make_opsetid("ai.onnx.ml", 2),
])
model.ir_version = 8
onnx.save(model, "test/support/zipmap_string.onnx")
print("Generated test/support/zipmap_string.onnx")
