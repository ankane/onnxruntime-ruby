module OnnxRuntime
  module Utils
    class << self
      attr_accessor :mutex
    end
    self.mutex = Mutex.new

    def self.reshape(arr, dims)
      arr = arr.flatten
      dims[1..-1].reverse.each do |dim|
        arr = arr.each_slice(dim)
      end
      arr.to_a
    end

    def self.check_status(status)
      unless status.null?
        message = api[:GetErrorMessage].call(status).read_string
        api[:ReleaseStatus].call(status)
        raise Error, message
      end
    end

    def self.api
      FFI.api
    end

    def self.release(type, pointer)
      FFI.api[:"Release#{type}"].call(pointer.read_pointer) if pointer && !pointer.null?
    end

    def self.unsupported_type(name, type)
      raise Error, "Unsupported #{name} type: #{type}"
    end

    def self.tensor_type_and_shape(tensor_info)
      type = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetTensorElementType].call(tensor_info.read_pointer, type)

      num_dims_ptr = ::FFI::MemoryPointer.new(:size_t)
      check_status api[:GetDimensionsCount].call(tensor_info.read_pointer, num_dims_ptr)
      num_dims = num_dims_ptr.read(:size_t)

      node_dims = ::FFI::MemoryPointer.new(:int64, num_dims)
      check_status api[:GetDimensions].call(tensor_info.read_pointer, node_dims, num_dims)
      dims = node_dims.read_array_of_int64(num_dims)

      symbolic_dims = ::FFI::MemoryPointer.new(:pointer, num_dims)
      check_status api[:GetSymbolicDimensions].call(tensor_info.read_pointer, symbolic_dims, num_dims)
      named_dims = num_dims.times.map { |i| symbolic_dims[i].read_pointer.read_string }
      dims = named_dims.zip(dims).map { |n, d| n.empty? ? d : n }

      [type.read_int, dims]
    end

    def self.node_info(typeinfo)
      onnx_type = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetOnnxTypeFromTypeInfo].call(typeinfo.read_pointer, onnx_type)

      type = FFI::OnnxType[onnx_type.read_int]
      case type
      when :tensor
        tensor_info = ::FFI::MemoryPointer.new(:pointer)
        # don't free tensor_info
        check_status api[:CastTypeInfoToTensorInfo].call(typeinfo.read_pointer, tensor_info)

        type, shape = Utils.tensor_type_and_shape(tensor_info)
        {
          type: "tensor(#{FFI::TensorElementDataType[type]})",
          shape: shape
        }
      when :sequence
        sequence_type_info = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:CastTypeInfoToSequenceTypeInfo].call(typeinfo.read_pointer, sequence_type_info)
        nested_type_info = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:GetSequenceElementType].call(sequence_type_info.read_pointer, nested_type_info)
        v = node_info(nested_type_info)[:type]

        {
          type: "seq(#{v})",
          shape: []
        }
      when :map
        map_type_info = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:CastTypeInfoToMapTypeInfo].call(typeinfo.read_pointer, map_type_info)

        # key
        key_type = ::FFI::MemoryPointer.new(:int)
        check_status api[:GetMapKeyType].call(map_type_info.read_pointer, key_type)
        k = FFI::TensorElementDataType[key_type.read_int]

        # value
        value_type_info = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:GetMapValueType].call(map_type_info.read_pointer, value_type_info)
        v = node_info(value_type_info)[:type]

        {
          type: "map(#{k},#{v})",
          shape: []
        }
      else
        Utils.unsupported_type("ONNX", type)
      end
    ensure
      release :TypeInfo, typeinfo
    end

    def self.create_from_onnx_value(out_ptr, output_type)
      out_type = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetValueType].call(out_ptr, out_type)
      type = FFI::OnnxType[out_type.read_int]

      case type
      when :tensor
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:GetTensorTypeAndShape].call(out_ptr, typeinfo)

        type, shape = Utils.tensor_type_and_shape(typeinfo)

        tensor_data = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:GetTensorMutableData].call(out_ptr, tensor_data)

        out_size = ::FFI::MemoryPointer.new(:size_t)
        check_status api[:GetTensorShapeElementCount].call(typeinfo.read_pointer, out_size)
        output_tensor_size = out_size.read(:size_t)

        release :TensorTypeAndShapeInfo, typeinfo

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
            numo_type = numo_types[type]
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

          Utils.reshape(arr, shape)
        else
          raise ArgumentError, "Invalid output type: #{output_type}"
        end
      when :sequence
        out = ::FFI::MemoryPointer.new(:size_t)
        check_status api[:GetValueCount].call(out_ptr, out)

        out.read(:size_t).times.map do |i|
          seq = ::FFI::MemoryPointer.new(:pointer)
          check_status api[:GetValue].call(out_ptr, i, allocator.read_pointer, seq)
          create_from_onnx_value(seq.read_pointer, output_type)
        end
      when :map
        type_shape = ::FFI::MemoryPointer.new(:pointer)
        map_keys = ::FFI::MemoryPointer.new(:pointer)
        map_values = ::FFI::MemoryPointer.new(:pointer)
        elem_type = ::FFI::MemoryPointer.new(:int)

        check_status api[:GetValue].call(out_ptr, 0, allocator.read_pointer, map_keys)
        check_status api[:GetValue].call(out_ptr, 1, allocator.read_pointer, map_values)
        check_status api[:GetTensorTypeAndShape].call(map_keys.read_pointer, type_shape)
        check_status api[:GetTensorElementType].call(type_shape.read_pointer, elem_type)
        release :TensorTypeAndShapeInfo, type_shape

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
    ensure
      api[:ReleaseValue].call(out_ptr) unless out_ptr.null?
    end

    def self.create_strings_from_onnx_value(out_ptr, output_tensor_size, result)
      len = ::FFI::MemoryPointer.new(:size_t)
      check_status api[:GetStringTensorDataLength].call(out_ptr, len)

      s_len = len.read(:size_t)
      s = ::FFI::MemoryPointer.new(:uchar, s_len)
      offsets = ::FFI::MemoryPointer.new(:size_t, output_tensor_size)
      check_status api[:GetStringTensorContent].call(out_ptr, s, s_len, offsets, output_tensor_size)

      offsets = output_tensor_size.times.map { |i| offsets[i].read(:size_t) }
      offsets << s_len
      output_tensor_size.times do |i|
        result[i] = s.get_bytes(offsets[i], offsets[i + 1] - offsets[i])
      end
      result
    end

    def self.numo_types
      @numo_types ||= {
        float: Numo::SFloat,
        uint8: Numo::UInt8,
        int8: Numo::Int8,
        uint16: Numo::UInt16,
        int16: Numo::Int16,
        int32: Numo::Int32,
        int64: Numo::Int64,
        bool: Numo::UInt8,
        double: Numo::DFloat,
        uint32: Numo::UInt32,
        uint64: Numo::UInt64
      }
    end

    def self.allocator
      @allocator ||= begin
        allocator = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:GetAllocatorWithDefaultOptions].call(allocator)
        allocator
      end
    end
  end
end
