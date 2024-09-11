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
  end
end
