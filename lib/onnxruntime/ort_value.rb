module OnnxRuntime
  class OrtValue
    def initialize(ortvalue, binary_data = nil)
      @ortvalue = ortvalue.read_pointer
      @binary_data = binary_data
      ObjectSpace.define_finalizer(@ortvalue, self.class.finalize(@ortvalue.to_i))
    end

    def self.ortvalue_from_numo(numo_obj)
      element_type = numo_obj.is_a?(Numo::Bit) ? :bool : Utils.numo_types.invert[numo_obj.class]
      Utils.unsupported_type("Numo", numo_obj.class.name) unless element_type

      ortvalue_from_array(numo_obj, element_type: element_type)
    end

    def self.ortvalue_from_array(input, element_type:)
      type_enum = FFI::TensorElementDataType[element_type]
      Utils.unsupported_type("element", element_type) unless type_enum

      shape = Utils.input_shape(input)
      input_node_dims = ::FFI::MemoryPointer.new(:int64, shape.size)
      input_node_dims.write_array_of_int64(shape)

      ptr = ::FFI::MemoryPointer.new(:pointer)
      if element_type == :string
        # keep reference to _str_ptrs until FillStringTensor call
        input_tensor_values, _str_ptrs = create_input_strings(input)
        Utils.check_status FFI.api[:CreateTensorAsOrtValue].call(Utils.allocator.read_pointer, input_node_dims, shape.size, type_enum, ptr)
        Utils.check_status FFI.api[:FillStringTensor].call(ptr.read_pointer, input_tensor_values, input_tensor_values.size / input_tensor_values.type_size)
      else
        input_tensor_values = create_input_data(input, element_type)
        Utils.check_status FFI.api[:CreateTensorWithDataAsOrtValue].call(allocator_info.read_pointer, input_tensor_values, input_tensor_values.size, input_node_dims, shape.size, type_enum, ptr)
      end

      new(ptr, input_tensor_values)
    end

    def self.create_input_data(input, tensor_type)
      if Utils.numo_array?(input)
        input.cast_to(Utils.numo_types[tensor_type]).to_binary
      else
        flat_input = input.flatten.to_a
        input_tensor_values = ::FFI::MemoryPointer.new(tensor_type, flat_input.size)
        if tensor_type == :bool
          input_tensor_values.write_array_of_uint8(flat_input.map { |v| v ? 1 : 0 })
        else
          input_tensor_values.send("write_array_of_#{tensor_type}", flat_input)
        end
        input_tensor_values
      end
    end
    private_class_method :create_input_data

    def self.create_input_strings(input)
      str_ptrs =
        if Utils.numo_array?(input)
          input.size.times.map { |i| ::FFI::MemoryPointer.from_string(input[i]) }
        else
          input.flatten.map { |v| ::FFI::MemoryPointer.from_string(v) }
        end

      input_tensor_values = ::FFI::MemoryPointer.new(:pointer, str_ptrs.size)
      input_tensor_values.write_array_of_pointer(str_ptrs)
      [input_tensor_values, str_ptrs]
    end
    private_class_method :create_input_strings

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

    def numo
      create_from_onnx_value(out_ptr, :numo)
    end

    def to_a
      create_from_onnx_value(out_ptr, :ruby)
    end

    private

    def out_ptr
      @ortvalue
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

    def create_from_onnx_value(out_ptr, output_type)
      out_type = ::FFI::MemoryPointer.new(:int)
      Utils.check_status FFI.api[:GetValueType].call(out_ptr, out_type)
      type = FFI::OnnxType[out_type.read_int]

      case type
      when :tensor
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        Utils.check_status FFI.api[:GetTensorTypeAndShape].call(out_ptr, typeinfo)

        type, shape = Utils.tensor_type_and_shape(typeinfo)

        tensor_data = ::FFI::MemoryPointer.new(:pointer)
        Utils.check_status FFI.api[:GetTensorMutableData].call(out_ptr, tensor_data)

        out_size = ::FFI::MemoryPointer.new(:size_t)
        Utils.check_status FFI.api[:GetTensorShapeElementCount].call(typeinfo.read_pointer, out_size)
        output_tensor_size = out_size.read(:size_t)

        Utils.release :TensorTypeAndShapeInfo, typeinfo

        # TODO support more types
        type = FFI::TensorElementDataType[type]

        case output_type
        when :numo
          case type
          when :string
            result = Numo::RObject.new(shape)
            result.allocate
            create_strings_from_onnx_value(out_ptr, output_tensor_size, result)
          else
            numo_type = Utils.numo_types[type]
            Utils.unsupported_type("element", type) unless numo_type
            numo_type.from_binary(tensor_data.read_pointer.read_bytes(output_tensor_size * numo_type::ELEMENT_BYTE_SIZE), shape)
          end
        when :ruby
          arr =
            case type
            when :float, :uint8, :int8, :uint16, :int16, :int32, :int64, :double, :uint32, :uint64
              tensor_data.read_pointer.send("read_array_of_#{type}", output_tensor_size)
            when :bool
              tensor_data.read_pointer.read_array_of_uint8(output_tensor_size).map { |v| v == 1 }
            when :string
              create_strings_from_onnx_value(out_ptr, output_tensor_size, [])
            else
              Utils.unsupported_type("element", type)
            end

          reshape(arr, shape)
        else
          raise ArgumentError, "Invalid output type: #{output_type}"
        end
      when :sequence
        out = ::FFI::MemoryPointer.new(:size_t)
        Utils.check_status FFI.api[:GetValueCount].call(out_ptr, out)

        out.read(:size_t).times.map do |i|
          seq = ::FFI::MemoryPointer.new(:pointer)
          Utils.check_status FFI.api[:GetValue].call(out_ptr, i, Utils.allocator.read_pointer, seq)
          create_from_onnx_value(seq.read_pointer, output_type)
        end
      when :map
        type_shape = ::FFI::MemoryPointer.new(:pointer)
        map_keys = ::FFI::MemoryPointer.new(:pointer)
        map_values = ::FFI::MemoryPointer.new(:pointer)
        elem_type = ::FFI::MemoryPointer.new(:int)

        Utils.check_status FFI.api[:GetValue].call(out_ptr, 0, Utils.allocator.read_pointer, map_keys)
        Utils.check_status FFI.api[:GetValue].call(out_ptr, 1, Utils.allocator.read_pointer, map_values)
        Utils.check_status FFI.api[:GetTensorTypeAndShape].call(map_keys.read_pointer, type_shape)
        Utils.check_status FFI.api[:GetTensorElementType].call(type_shape.read_pointer, elem_type)
        Utils.release :TensorTypeAndShapeInfo, type_shape

        # TODO support more types
        elem_type = FFI::TensorElementDataType[elem_type.read_int]
        case elem_type
        when :int64
          ret = {}
          keys = create_from_onnx_value(map_keys.read_pointer, output_type)
          values = create_from_onnx_value(map_values.read_pointer, output_type)
          keys.zip(values).each do |k, v|
            ret[k] = v
          end
          ret
        else
          Utils.unsupported_type("element", elem_type)
        end
      else
        Utils.unsupported_type("ONNX", type)
      end
    end

    def create_strings_from_onnx_value(out_ptr, output_tensor_size, result)
      len = ::FFI::MemoryPointer.new(:size_t)
      Utils.check_status FFI.api[:GetStringTensorDataLength].call(out_ptr, len)

      s_len = len.read(:size_t)
      s = ::FFI::MemoryPointer.new(:uchar, s_len)
      offsets = ::FFI::MemoryPointer.new(:size_t, output_tensor_size)
      Utils.check_status FFI.api[:GetStringTensorContent].call(out_ptr, s, s_len, offsets, output_tensor_size)

      offsets = output_tensor_size.times.map { |i| offsets[i].read(:size_t) }
      offsets << s_len
      output_tensor_size.times do |i|
        result[i] = s.get_bytes(offsets[i], offsets[i + 1] - offsets[i])
      end
      result
    end

    def reshape(arr, dims)
      arr = arr.flatten
      dims[1..-1].reverse.each do |dim|
        arr = arr.each_slice(dim)
      end
      arr.to_a
    end

    def self.finalize(addr)
      # must use proc instead of stabby lambda
      proc { FFI.api[:ReleaseValue].call(::FFI::Pointer.new(:pointer, addr)) }
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
