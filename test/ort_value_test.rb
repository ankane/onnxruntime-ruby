require_relative "test_helper"

class OrtValueTest < Minitest::Test
  def test_from_numo
    skip unless numo?

    x = Numo::SFloat.cast([[5.8, 2.8]])
    value = OnnxRuntime::OrtValue.from_numo(x)
    assert_equal true, value.tensor?
    assert_equal "tensor(float)", value.data_type
    assert_equal :float, value.element_type
    assert_equal [1, 2], value.shape
    assert_equal "cpu", value.device_name
    assert_equal x, value.numo
    assert_elements_in_delta [5.8, 2.8], value.to_ruby[0]
    assert_elements_in_delta [5.8, 2.8], value.data_ptr.read_array_of_float(2)
  end

  def test_from_array
    value = OnnxRuntime::OrtValue.from_array([[5.8, 2.8]], element_type: :double)
    assert_equal true, value.tensor?
    assert_equal "tensor(double)", value.data_type
    assert_equal :double, value.element_type
    assert_equal [1, 2], value.shape
    assert_equal "cpu", value.device_name
    assert_elements_in_delta [5.8, 2.8], value.to_ruby[0]
    assert_elements_in_delta [5.8, 2.8], value.data_ptr.read_array_of_double(2)
  end

  def test_from_array_float16
    # float16 for [1.0, 2.0, 3.0]
    value = OnnxRuntime::OrtValue.from_array([[15360, 16384, 16896]], element_type: :float16)
    assert_equal true, value.tensor?
    assert_equal "tensor(float16)", value.data_type
    assert_equal :float16, value.element_type
    assert_equal [1, 3], value.shape
    assert_equal [15360, 16384, 16896], value.to_ruby[0]
    assert_equal [15360, 16384, 16896], value.data_ptr.read_array_of_uint16(3)
  end

  def test_from_array_bfloat16
    # bfloat16 for [1.0, 2.0, 3.0]
    value = OnnxRuntime::OrtValue.from_array([[16256, 16384, 16448]], element_type: :bfloat16)
    assert_equal true, value.tensor?
    assert_equal "tensor(bfloat16)", value.data_type
    assert_equal :bfloat16, value.element_type
    assert_equal [1, 3], value.shape
    assert_equal [16256, 16384, 16448], value.to_ruby[0]
    assert_equal [16256, 16384, 16448], value.data_ptr.read_array_of_uint16(3)
  end

  def test_from_shape_and_type
    value = OnnxRuntime::OrtValue.from_shape_and_type([1, 2], :double)
    data_ptr = value.data_ptr
    data_ptr.write_array_of_double([5.8, 2.8])
    assert_equal true, value.tensor?
    assert_equal "tensor(double)", value.data_type
    assert_equal :double, value.element_type
    assert_equal [1, 2], value.shape
    assert_equal "cpu", value.device_name
    assert_elements_in_delta [5.8, 2.8], value.to_ruby[0]
    assert_elements_in_delta [5.8, 2.8], value.data_ptr.read_array_of_double(2)
  end
end
