require_relative "test_helper"

class DatasetsTest < Minitest::Test
  def test_examples
    assert_example "logreg_iris.onnx", ["float_input"]
    assert_example "mul_1.onnx", ["X"]
    assert_example "sigmoid.onnx", ["x"]
  end

  def test_bad_example
    error = assert_raises(ArgumentError) do
      OnnxRuntime::Datasets.example("bad.onnx")
    end
    # same message as Python
    assert_equal "Unable to find example 'bad.onnx'", error.message
  end

  def test_no_path_traversal
    error = assert_raises(ArgumentError) do
      OnnxRuntime::Datasets.example("../datasets/sigmoid.onnx")
    end
    assert_equal "Unable to find example '../datasets/sigmoid.onnx'", error.message
  end

  private

  def assert_example(name, input_names)
    example = OnnxRuntime::Datasets.example(name)
    model = OnnxRuntime::Model.new(example)
    assert_equal input_names, model.inputs.map { |i| i[:name] }
  end
end
