require_relative "test_helper"

class OrtValueTest < Minitest::Test
  def test_ortvalue_from_numo
    x = Numo::SFloat.cast([[5.8, 2.8]])
    value = OnnxRuntime::OrtValue.ortvalue_from_numo(x)
    assert_equal true, value.tensor?
    assert_equal "tensor(float)", value.data_type
    assert_equal 1, value.element_type
    assert_equal [1, 2], value.shape
    assert_equal "cpu", value.device_name
  end
end
