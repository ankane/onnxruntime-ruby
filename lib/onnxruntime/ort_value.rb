module OnnxRuntime
  class OrtValue
    def initialize(ortvalue, numo_obj = nil)
      @ortvalue = ortvalue
      @numo_obj = numo_obj
    end

    def self.ortvalue_from_numo(numo_obj)
      type_enum =
        case numo_obj
        when Numo::SFloat
          1
        else
          Utils.unsupported_type("Numo", numo_obj.class.name)
        end

      shape = numo_obj.shape
      input_tensor_values = numo_obj.to_binary
      input_node_dims = ::FFI::MemoryPointer.new(:int64, shape.size)
      input_node_dims.write_array_of_int64(shape)
      ptr = ::FFI::MemoryPointer.new(:pointer)
      Utils.check_status FFI.api[:CreateTensorWithDataAsOrtValue].call(allocator_info.read_pointer, input_tensor_values, input_tensor_values.size, input_node_dims, shape.size, type_enum, ptr)

      new(ptr, numo_obj)
    end

    def tensor?
      FFI::OnnxType[value_type] == :tensor
    end

    def data_type
      @data_type ||= begin
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        Utils.check_status FFI.api[:GetTypeInfo].call(out_ptr, typeinfo)
        Utils.node_info(typeinfo)[:type]
      end
    end

    def element_type
      type_and_shape_info[0]
    end

    def shape
      type_and_shape_info[1]
    end

    def device_name
      "cpu"
    end

    private

    def out_ptr
      @ortvalue.read_pointer
    end

    def value_type
      @value_type ||= begin
        out_type = ::FFI::MemoryPointer.new(:int)
        Utils.check_status FFI.api[:GetValueType].call(out_ptr, out_type)
        out_type.read_int
      end
    end

    def type_and_shape_info
      @type_and_shape_info ||= begin
        begin
          typeinfo = ::FFI::MemoryPointer.new(:pointer)
          Utils.check_status FFI.api[:GetTensorTypeAndShape].call(out_ptr, typeinfo)
          Utils.tensor_type_and_shape(typeinfo)
        ensure
          Utils.release :TensorTypeAndShapeInfo, typeinfo
        end
      end
    end

    def self.allocator_info
      @allocator_info ||= begin
        allocator_info = ::FFI::MemoryPointer.new(:pointer)
        Utils.check_status FFI.api[:CreateCpuMemoryInfo].call(1, 0, allocator_info)
        allocator_info
      end
    end
  end
end
