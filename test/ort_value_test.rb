require_relative "test_helper"

class OrtValueTest < Minitest::Test
  def test_from_numo
    x = Numo::SFloat.cast([[5.8, 2.8]])
    value = OnnxRuntime::OrtValue.from_numo(x)
    assert_equal true, value.tensor?
    assert_equal "tensor(float)", value.data_type
    assert_equal 1, value.element_type
    assert_equal [1, 2], value.shape
    assert_equal "cpu", value.device_name
    assert_equal x, value.numo
    assert_elements_in_delta [5.8, 2.8], value.to_ruby[0]
  end
end
