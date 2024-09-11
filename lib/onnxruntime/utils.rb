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
  end
end
