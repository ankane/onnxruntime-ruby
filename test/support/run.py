import numpy
import onnxruntime as rt

sess = rt.InferenceSession("test/support/model.onnx")
input_name = sess.get_inputs()[0].name
numpy.random.seed(0)
X = numpy.random.random((3, 4, 5)).astype(numpy.float32)
predictions = sess.run(None, {input_name: X})

print("Inputs")
[print((n.name, n.type, n.shape)) for n in sess.get_inputs()]
print("Outputs")
[print((n.name, n.type, n.shape)) for n in sess.get_outputs()]
print("Predictions")
print(predictions)

metadata = sess.get_modelmeta()
print(metadata.custom_metadata_map)
print(metadata.description)
print(metadata.domain)
print(metadata.graph_name)
print(metadata.producer_name)
print(metadata.version)
