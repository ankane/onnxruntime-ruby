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
end
