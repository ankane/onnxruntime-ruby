require_relative "test_helper"

class InferenceSessionTest < Minitest::Test
  def test_run_with_ort_values
    sess = OnnxRuntime::InferenceSession.new("test/support/lightgbm.onnx")
    x = OnnxRuntime::OrtValue.ortvalue_from_numo(Numo::SFloat.cast([[5.8, 2.8]]))
    output = sess.run_with_ort_values(nil, {input: x})
    assert_equal true, output[0].tensor?
    assert_equal "tensor(int64)", output[0].data_type
    assert_equal Numo::Int64.cast([1]), output[0].numo
    assert_equal [1], output[0].to_a
    assert_equal false, output[1].tensor?
    assert_equal "seq(map(int64,tensor(float)))", output[1].data_type
  end

  def test_run_with_ort_values_invalid_type
    sess = OnnxRuntime::InferenceSession.new("test/support/lightgbm.onnx")
    x = OnnxRuntime::OrtValue.ortvalue_from_numo(Numo::DFloat.cast([[5.8, 2.8]]))
    error = assert_raises(OnnxRuntime::Error) do
      sess.run_with_ort_values(nil, {input: x})
    end
    assert_equal "Unexpected input data type. Actual: (tensor(double)) , expected: (tensor(float))", error.message
  end

  def test_providers
    sess = OnnxRuntime::InferenceSession.new("test/support/model.onnx")
    assert_includes sess.providers, "CPUExecutionProvider"
  end

  def test_providers_cuda
    assert_output nil, /Provider not available: CUDAExecutionProvider/ do
      OnnxRuntime::InferenceSession.new("test/support/model.onnx", providers: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    end
  end

  def test_providers_coreml
    skip unless mac?

    options = {providers: ["CoreMLExecutionProvider", "CPUExecutionProvider"]}
    options[:log_severity_level] = 1 if ENV["VERBOSE"]
    sess = OnnxRuntime::InferenceSession.new("datasets/mul_1.onnx", **options)
    output, _ = sess.run(nil, {"X" => [[1, 2], [3, 4], [5, 6]]})
    assert_elements_in_delta [1, 4], output[0]
    assert_elements_in_delta [9, 16], output[1]
    assert_elements_in_delta [25, 36], output[2]
  end

  def test_profiling
    sess = OnnxRuntime::InferenceSession.new("test/support/model.onnx", enable_profiling: true)
    file = sess.end_profiling
    assert_match ".json", file
    File.unlink(file)
  end

  def test_profile_file_prefix
    sess = OnnxRuntime::InferenceSession.new("test/support/model.onnx", enable_profiling: true, profile_file_prefix: "hello")
    file = sess.end_profiling
    assert_match "hello", file
    File.unlink(file)
  end

  def test_copy
    sess = OnnxRuntime::InferenceSession.new("test/support/model.onnx")
    sess.dup
    sess.clone
  end
end
