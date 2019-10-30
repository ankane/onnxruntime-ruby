module OnnxRuntime
  class InferenceSession
    attr_reader :inputs, :outputs

    def initialize(path_or_bytes)
      # session options
      session_options = ::FFI::MemoryPointer.new(:pointer)
      check_status api[:CreateSessionOptions].call(session_options)

      # session
      @session = ::FFI::MemoryPointer.new(:pointer)
      path_or_bytes = path_or_bytes.to_str

      # fix for Windows "File doesn't exist"
      if Gem.win_platform? && path_or_bytes.encoding != Encoding::BINARY
        path_or_bytes = File.binread(path_or_bytes)
      end

      if path_or_bytes.encoding == Encoding::BINARY
        check_status api[:CreateSessionFromArray].call(env.read_pointer, path_or_bytes, path_or_bytes.bytesize, session_options.read_pointer, @session)
      else
        check_status api[:CreateSession].call(env.read_pointer, path_or_bytes, session_options.read_pointer, @session)
      end

      # input info
      allocator = ::FFI::MemoryPointer.new(:pointer)
      check_status api[:GetAllocatorWithDefaultOptions].call(allocator)
      @allocator = allocator

      @inputs = []
      @outputs = []

      # input
      num_input_nodes = ::FFI::MemoryPointer.new(:size_t)
      check_status api[:SessionGetInputCount].call(read_pointer, num_input_nodes)
      read_size_t(num_input_nodes).times do |i|
        name_ptr = ::FFI::MemoryPointer.new(:string)
        check_status api[:SessionGetInputName].call(read_pointer, i, @allocator.read_pointer, name_ptr)
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:SessionGetInputTypeInfo].call(read_pointer, i, typeinfo)
        @inputs << {name: name_ptr.read_pointer.read_string}.merge(node_info(typeinfo))
      end

      # output
      num_output_nodes = ::FFI::MemoryPointer.new(:size_t)
      check_status api[:SessionGetOutputCount].call(read_pointer, num_output_nodes)
      read_size_t(num_output_nodes).times do |i|
        name_ptr = ::FFI::MemoryPointer.new(:string)
        check_status api[:SessionGetOutputName].call(read_pointer, i, allocator.read_pointer, name_ptr)
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:SessionGetOutputTypeInfo].call(read_pointer, i, typeinfo)
        @outputs << {name: name_ptr.read_pointer.read_string}.merge(node_info(typeinfo))
      end
    end

    def run(output_names, input_feed)
      input_tensor = create_input_tensor(input_feed)

      output_names ||= @outputs.map { |v| v[:name] }

      output_tensor = ::FFI::MemoryPointer.new(:pointer, outputs.size)
      input_node_names = create_node_names(input_feed.keys.map(&:to_s))
      output_node_names = create_node_names(output_names.map(&:to_s))
      # TODO support run options
      check_status api[:Run].call(read_pointer, nil, input_node_names, input_tensor, input_feed.size, output_node_names, output_names.size, output_tensor)

      output_names.size.times.map do |i|
        create_from_onnx_value(output_tensor[i].read_pointer)
      end
    end

    private

    def create_input_tensor(input_feed)
      allocator_info = ::FFI::MemoryPointer.new(:pointer)
      check_status = api[:CreateCpuMemoryInfo].call(1, 0, allocator_info)
      input_tensor = ::FFI::MemoryPointer.new(:pointer, input_feed.size)

      input_feed.each_with_index do |(input_name, input), idx|
        input = input.to_a unless input.is_a?(Array)

        shape = []
        s = input
        while s.is_a?(Array)
          shape << s.size
          s = s.first
        end

        flat_input = input.flatten
        input_tensor_size = flat_input.size

        # TODO support more types
        inp = @inputs.find { |i| i[:name] == input_name.to_s }
        raise "Unknown input: #{input_name}" unless inp

        input_node_dims = ::FFI::MemoryPointer.new(:int64, shape.size)
        input_node_dims.write_array_of_int64(shape)

        if inp[:type] == "tensor(string)"
          input_tensor_values = ::FFI::MemoryPointer.new(:pointer, input_tensor_size)
          input_tensor_values.write_array_of_pointer(flat_input.map { |v| ::FFI::MemoryPointer.from_string(v) })
          type_enum = FFI::TensorElementDataType[:string]
          check_status api[:CreateTensorAsOrtValue].call(@allocator.read_pointer, input_node_dims, shape.size, type_enum, input_tensor[idx])
          check_status api[:FillStringTensor].call(input_tensor[idx].read_pointer, input_tensor_values, flat_input.size)
        else
          tensor_types = [:float, :uint8, :int8, :uint16, :int16, :int32, :int64, :bool, :double, :uint32, :uint64].map { |v| ["tensor(#{v})", v] }.to_h
          tensor_type = tensor_types[inp[:type]]

          if tensor_type
            input_tensor_values = ::FFI::MemoryPointer.new(tensor_type, input_tensor_size)
            if tensor_type == :bool
              tensor_type = :uchar
              flat_input = flat_input.map { |v| v ? 1 : 0 }
            end
            input_tensor_values.send("write_array_of_#{tensor_type}", flat_input)
            type_enum = FFI::TensorElementDataType[tensor_type]
          else
            unsupported_type("input", inp[:type])
          end

          check_status api[:CreateTensorWithDataAsOrtValue].call(allocator_info.read_pointer, input_tensor_values, input_tensor_values.size, input_node_dims, shape.size, type_enum, input_tensor[idx])
        end
      end

      input_tensor
    end

    def create_node_names(names)
      ptr = ::FFI::MemoryPointer.new(:pointer, names.size)
      ptr.write_array_of_pointer(names.map { |v| ::FFI::MemoryPointer.from_string(v) })
      ptr
    end

    def create_from_onnx_value(out_ptr)
      out_type = ::FFI::MemoryPointer.new(:int)
      check_status = api[:GetValueType].call(out_ptr, out_type)
      type = FFI::OnnxType[out_type.read_int]

      case type
      when :tensor
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:GetTensorTypeAndShape].call(out_ptr, typeinfo)

        type, shape = tensor_type_and_shape(typeinfo)

        tensor_data = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:GetTensorMutableData].call(out_ptr, tensor_data)

        out_size = ::FFI::MemoryPointer.new(:size_t)
        output_tensor_size = api[:GetTensorShapeElementCount].call(typeinfo.read_pointer, out_size)
        output_tensor_size = read_size_t(out_size)

        # TODO support more types
        type = FFI::TensorElementDataType[type]
        arr =
          case type
          when :float, :uint8, :int8, :uint16, :int16, :int32, :int64, :double, :uint32, :uint64
            tensor_data.read_pointer.send("read_array_of_#{type}", output_tensor_size)
          when :bool
            tensor_data.read_pointer.read_array_of_uchar(output_tensor_size).map { |v| v == 1 }
          else
            unsupported_type("element", type)
          end

        Utils.reshape(arr, shape)
      when :sequence
        out = ::FFI::MemoryPointer.new(:size_t)
        check_status api[:GetValueCount].call(out_ptr, out)

        read_size_t(out).times.map do |i|
          seq = ::FFI::MemoryPointer.new(:pointer)
          check_status api[:GetValue].call(out_ptr, i, @allocator.read_pointer, seq)
          create_from_onnx_value(seq.read_pointer)
        end
      when :map
        type_shape = ::FFI::MemoryPointer.new(:pointer)
        map_keys = ::FFI::MemoryPointer.new(:pointer)
        map_values = ::FFI::MemoryPointer.new(:pointer)
        elem_type = ::FFI::MemoryPointer.new(:int)

        check_status api[:GetValue].call(out_ptr, 0, @allocator.read_pointer, map_keys)
        check_status api[:GetValue].call(out_ptr, 1, @allocator.read_pointer, map_values)
        check_status api[:GetTensorTypeAndShape].call(map_keys.read_pointer, type_shape)
        check_status api[:GetTensorElementType].call(type_shape.read_pointer, elem_type)

        # TODO support more types
        elem_type = FFI::TensorElementDataType[elem_type.read_int]
        case elem_type
        when :int64
          ret = {}
          keys = create_from_onnx_value(map_keys.read_pointer)
          values = create_from_onnx_value(map_values.read_pointer)
          keys.zip(values).each do |k, v|
            ret[k] = v
          end
          ret
        else
          unsupported_type("element", elem_type)
        end
      else
        unsupported_type("ONNX", type)
      end
    end

    def read_pointer
      @session.read_pointer
    end

    def check_status(status)
      unless status.null?
        message = api[:GetErrorMessage].call(status)
        api[:ReleaseStatus].call(status)
        raise OnnxRuntime::Error, message
      end
    end

    def node_info(typeinfo)
      onnx_type = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetOnnxTypeFromTypeInfo].call(typeinfo.read_pointer, onnx_type)

      type = FFI::OnnxType[onnx_type.read_int]
      case type
      when :tensor
        tensor_info = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:CastTypeInfoToTensorInfo].call(typeinfo.read_pointer, tensor_info)

        type, shape = tensor_type_and_shape(tensor_info)
        {
          type: "tensor(#{FFI::TensorElementDataType[type]})",
          shape: shape
        }
      when :sequence
        # TODO show nested
        {
          type: "seq",
          shape: []
        }
      when :map
        # TODO show nested
        {
          type: "map",
          shape: []
        }
      else
        unsupported_type("ONNX", type)
      end
    ensure
      api[:ReleaseTypeInfo].call(typeinfo.read_pointer)
    end

    def tensor_type_and_shape(tensor_info)
      type = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetTensorElementType].call(tensor_info.read_pointer, type)

      num_dims_ptr = ::FFI::MemoryPointer.new(:size_t)
      check_status api[:GetDimensionsCount].call(tensor_info.read_pointer, num_dims_ptr)
      num_dims = read_size_t(num_dims_ptr)

      node_dims = ::FFI::MemoryPointer.new(:int64, num_dims)
      check_status api[:GetDimensions].call(tensor_info.read_pointer, node_dims, num_dims)

      [type.read_int, node_dims.read_array_of_int64(num_dims)]
    end

    def unsupported_type(name, type)
      raise "Unsupported #{name} type: #{type}"
    end

    # read(:size_t) not supported in FFI JRuby
    def read_size_t(ptr)
      if RUBY_PLATFORM == "java"
        ptr.read_long
      else
        ptr.read(:size_t)
      end
    end

    def api
      @api ||= FFI.OrtGetApiBase[:GetApi].call(1)
    end

    def env
      # use mutex for thread-safety
      Utils.mutex.synchronize do
        @@env ||= begin
          env = ::FFI::MemoryPointer.new(:pointer)
          check_status api[:CreateEnv].call(3, "Default", env)
          at_exit { api[:ReleaseEnv].call(env.read_pointer) }
          # disable telemetry
          # https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md
          check_status api[:DisableTelemetryEvents].call(env)
          env
        end
      end
    end
  end
end
