module OnnxRuntime
  class InferenceSession
    attr_reader :inputs, :outputs

    def initialize(path_or_bytes, enable_cpu_mem_arena: true, enable_mem_pattern: true, enable_profiling: false, execution_mode: nil, free_dimension_overrides_by_denotation: nil, free_dimension_overrides_by_name: nil, graph_optimization_level: nil, inter_op_num_threads: nil, intra_op_num_threads: nil, log_severity_level: nil, log_verbosity_level: nil, logid: nil, optimized_model_filepath: nil, profile_file_prefix: nil, session_config_entries: nil)
      # session options
      session_options = ::FFI::MemoryPointer.new(:pointer)
      check_status api[:CreateSessionOptions].call(session_options)
      if enable_cpu_mem_arena
        check_status api[:EnableCpuMemArena].call(session_options.read_pointer)
      else
        check_status api[:DisableCpuMemArena].call(session_options.read_pointer)
      end
      if enable_mem_pattern
        check_status api[:EnableMemPattern].call(session_options.read_pointer)
      else
        check_status api[:DisableMemPattern].call(session_options.read_pointer)
      end
      if enable_profiling
        check_status api[:EnableProfiling].call(session_options.read_pointer, ort_string(profile_file_prefix || "onnxruntime_profile_"))
      else
        check_status api[:DisableProfiling].call(session_options.read_pointer)
      end
      if execution_mode
        execution_modes = {sequential: 0, parallel: 1}
        mode = execution_modes[execution_mode]
        raise ArgumentError, "Invalid execution mode" unless mode
        check_status api[:SetSessionExecutionMode].call(session_options.read_pointer, mode)
      end
      if free_dimension_overrides_by_denotation
        free_dimension_overrides_by_denotation.each do |k, v|
          check_status api[:AddFreeDimensionOverride].call(session_options.read_pointer, k.to_s, v)
        end
      end
      if free_dimension_overrides_by_name
        free_dimension_overrides_by_name.each do |k, v|
          check_status api[:AddFreeDimensionOverrideByName].call(session_options.read_pointer, k.to_s, v)
        end
      end
      if graph_optimization_level
        optimization_levels = {none: 0, basic: 1, extended: 2, all: 99}
        level = optimization_levels[graph_optimization_level]
        raise ArgumentError, "Invalid graph optimization level" unless level
        check_status api[:SetSessionGraphOptimizationLevel].call(session_options.read_pointer, level)
      end
      check_status api[:SetInterOpNumThreads].call(session_options.read_pointer, inter_op_num_threads) if inter_op_num_threads
      check_status api[:SetIntraOpNumThreads].call(session_options.read_pointer, intra_op_num_threads) if intra_op_num_threads
      check_status api[:SetSessionLogSeverityLevel].call(session_options.read_pointer, log_severity_level) if log_severity_level
      check_status api[:SetSessionLogVerbosityLevel].call(session_options.read_pointer, log_verbosity_level) if log_verbosity_level
      check_status api[:SetSessionLogId].call(session_options.read_pointer, logid) if logid
      check_status api[:SetOptimizedModelFilePath].call(session_options.read_pointer, ort_string(optimized_model_filepath)) if optimized_model_filepath
      if session_config_entries
        session_config_entries.each do |k, v|
          check_status api[:AddSessionConfigEntry].call(session_options.read_pointer, k.to_s, v.to_s)
        end
      end

      @session = load_session(path_or_bytes, session_options)
      ObjectSpace.define_finalizer(self, self.class.finalize(@session))

      @allocator = load_allocator
      @inputs = load_inputs
      @outputs = load_outputs
    ensure
      release :SessionOptions, session_options
    end

    # TODO support logid
    def run(output_names, input_feed, log_severity_level: nil, log_verbosity_level: nil, logid: nil, terminate: nil, output_type: :ruby)
      # pointer references
      refs = []

      input_tensor = create_input_tensor(input_feed, refs)

      output_names ||= @outputs.map { |v| v[:name] }

      output_tensor = ::FFI::MemoryPointer.new(:pointer, outputs.size)
      input_node_names = create_node_names(input_feed.keys.map(&:to_s), refs)
      output_node_names = create_node_names(output_names.map(&:to_s), refs)

      # run options
      run_options = ::FFI::MemoryPointer.new(:pointer)
      check_status api[:CreateRunOptions].call(run_options)
      check_status api[:RunOptionsSetRunLogSeverityLevel].call(run_options.read_pointer, log_severity_level) if log_severity_level
      check_status api[:RunOptionsSetRunLogVerbosityLevel].call(run_options.read_pointer, log_verbosity_level) if log_verbosity_level
      check_status api[:RunOptionsSetRunTag].call(run_options.read_pointer, logid) if logid
      check_status api[:RunOptionsSetTerminate].call(run_options.read_pointer) if terminate

      check_status api[:Run].call(read_pointer, run_options.read_pointer, input_node_names, input_tensor, input_feed.size, output_node_names, output_names.size, output_tensor)

      output_names.size.times.map do |i|
        create_from_onnx_value(output_tensor[i].read_pointer, output_type)
      end
    ensure
      release :RunOptions, run_options
      if input_tensor
        input_feed.size.times do |i|
          release :Value, input_tensor[i]
        end
      end
      # output values released in create_from_onnx_value
    end

    def modelmeta
      keys = ::FFI::MemoryPointer.new(:pointer)
      num_keys = ::FFI::MemoryPointer.new(:int64_t)
      description = ::FFI::MemoryPointer.new(:string)
      domain = ::FFI::MemoryPointer.new(:string)
      graph_name = ::FFI::MemoryPointer.new(:string)
      graph_description = ::FFI::MemoryPointer.new(:string)
      producer_name = ::FFI::MemoryPointer.new(:string)
      version = ::FFI::MemoryPointer.new(:int64_t)

      metadata = ::FFI::MemoryPointer.new(:pointer)
      check_status api[:SessionGetModelMetadata].call(read_pointer, metadata)

      custom_metadata_map = {}
      check_status api[:ModelMetadataGetCustomMetadataMapKeys].call(metadata.read_pointer, @allocator.read_pointer, keys, num_keys)
      num_keys.read(:int64_t).times do |i|
        key_ptr = keys.read_pointer[i * ::FFI::Pointer.size]
        key = key_ptr.read_pointer.read_string
        value = ::FFI::MemoryPointer.new(:string)
        check_status api[:ModelMetadataLookupCustomMetadataMap].call(metadata.read_pointer, @allocator.read_pointer, key, value)
        custom_metadata_map[key] = value.read_pointer.read_string

        allocator_free key_ptr
        allocator_free value
      end
      allocator_free keys

      check_status api[:ModelMetadataGetDescription].call(metadata.read_pointer, @allocator.read_pointer, description)
      check_status api[:ModelMetadataGetDomain].call(metadata.read_pointer, @allocator.read_pointer, domain)
      check_status api[:ModelMetadataGetGraphName].call(metadata.read_pointer, @allocator.read_pointer, graph_name)
      check_status api[:ModelMetadataGetGraphDescription].call(metadata.read_pointer, @allocator.read_pointer, graph_description)
      check_status api[:ModelMetadataGetProducerName].call(metadata.read_pointer, @allocator.read_pointer, producer_name)
      check_status api[:ModelMetadataGetVersion].call(metadata.read_pointer, version)

      {
        custom_metadata_map: custom_metadata_map,
        description: description.read_pointer.read_string,
        domain: domain.read_pointer.read_string,
        graph_name: graph_name.read_pointer.read_string,
        graph_description: graph_description.read_pointer.read_string,
        producer_name: producer_name.read_pointer.read_string,
        version: version.read(:int64_t)
      }
    ensure
      release :ModelMetadata, metadata
      allocator_free description
      allocator_free domain
      allocator_free graph_name
      allocator_free graph_description
      allocator_free producer_name
    end

    # return value has double underscore like Python
    def end_profiling
      out = ::FFI::MemoryPointer.new(:string)
      check_status api[:SessionEndProfiling].call(read_pointer, @allocator.read_pointer, out)
      out.read_pointer.read_string
    end

    # no way to set providers with C API yet
    # so we can return all available providers
    def providers
      out_ptr = ::FFI::MemoryPointer.new(:pointer)
      length_ptr = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetAvailableProviders].call(out_ptr, length_ptr)
      length = length_ptr.read_int
      providers = []
      length.times do |i|
        providers << out_ptr.read_pointer[i * ::FFI::Pointer.size].read_pointer.read_string
      end
      api[:ReleaseAvailableProviders].call(out_ptr.read_pointer, length)
      providers
    end

    private

    def load_session(path_or_bytes, session_options)
      session = ::FFI::MemoryPointer.new(:pointer)
      from_memory =
        if path_or_bytes.respond_to?(:read)
          path_or_bytes = path_or_bytes.read
          true
        else
          path_or_bytes = path_or_bytes.to_str
          # TODO remove ability to load byte string directly in 0.8.0
          path_or_bytes.encoding == Encoding::BINARY
        end

      if from_memory
        check_status api[:CreateSessionFromArray].call(env.read_pointer, path_or_bytes, path_or_bytes.bytesize, session_options.read_pointer, session)
      else
        check_status api[:CreateSession].call(env.read_pointer, ort_string(path_or_bytes), session_options.read_pointer, session)
      end
      session
    end

    def load_allocator
      allocator = ::FFI::MemoryPointer.new(:pointer)
      check_status api[:GetAllocatorWithDefaultOptions].call(allocator)
      allocator
    end

    def load_inputs
      inputs = []
      num_input_nodes = ::FFI::MemoryPointer.new(:size_t)
      check_status api[:SessionGetInputCount].call(read_pointer, num_input_nodes)
      num_input_nodes.read(:size_t).times do |i|
        name_ptr = ::FFI::MemoryPointer.new(:string)
        check_status api[:SessionGetInputName].call(read_pointer, i, @allocator.read_pointer, name_ptr)
        # freed in node_info
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:SessionGetInputTypeInfo].call(read_pointer, i, typeinfo)
        inputs << {name: name_ptr.read_pointer.read_string}.merge(node_info(typeinfo))
        allocator_free name_ptr
      end
      inputs
    end

    def load_outputs
      outputs = []
      num_output_nodes = ::FFI::MemoryPointer.new(:size_t)
      check_status api[:SessionGetOutputCount].call(read_pointer, num_output_nodes)
      num_output_nodes.read(:size_t).times do |i|
        name_ptr = ::FFI::MemoryPointer.new(:string)
        check_status api[:SessionGetOutputName].call(read_pointer, i, @allocator.read_pointer, name_ptr)
        # freed in node_info
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:SessionGetOutputTypeInfo].call(read_pointer, i, typeinfo)
        outputs << {name: name_ptr.read_pointer.read_string}.merge(node_info(typeinfo))
        allocator_free name_ptr
      end
      outputs
    end

    def create_input_tensor(input_feed, refs)
      allocator_info = ::FFI::MemoryPointer.new(:pointer)
      check_status api[:CreateCpuMemoryInfo].call(1, 0, allocator_info)
      input_tensor = ::FFI::MemoryPointer.new(:pointer, input_feed.size)

      input_feed.each_with_index do |(input_name, input), idx|
        if numo_array?(input)
          shape = input.shape
        else
          input = input.to_a unless input.is_a?(Array)

          shape = []
          s = input
          while s.is_a?(Array)
            shape << s.size
            s = s.first
          end
        end

        # TODO support more types
        inp = @inputs.find { |i| i[:name] == input_name.to_s }
        raise Error, "Unknown input: #{input_name}" unless inp

        input_node_dims = ::FFI::MemoryPointer.new(:int64, shape.size)
        input_node_dims.write_array_of_int64(shape)

        if inp[:type] == "tensor(string)"
          str_ptrs =
            if numo_array?(input)
              input.size.times.map { |i| ::FFI::MemoryPointer.from_string(input[i]) }
            else
              input.flatten.map { |v| ::FFI::MemoryPointer.from_string(v) }
            end

          input_tensor_values = ::FFI::MemoryPointer.new(:pointer, str_ptrs.size)
          input_tensor_values.write_array_of_pointer(str_ptrs)

          type_enum = FFI::TensorElementDataType[:string]
          check_status api[:CreateTensorAsOrtValue].call(@allocator.read_pointer, input_node_dims, shape.size, type_enum, input_tensor[idx])
          check_status api[:FillStringTensor].call(input_tensor[idx].read_pointer, input_tensor_values, str_ptrs.size)

          refs << str_ptrs
        else
          tensor_type = tensor_types[inp[:type]]

          if tensor_type
            if numo_array?(input)
              input_tensor_values = input.cast_to(numo_types[tensor_type]).to_binary
            else
              flat_input = input.flatten.to_a
              input_tensor_values = ::FFI::MemoryPointer.new(tensor_type, flat_input.size)
              if tensor_type == :bool
                input_tensor_values.write_array_of_uint8(flat_input.map { |v| v ? 1 : 0 })
              else
                input_tensor_values.send("write_array_of_#{tensor_type}", flat_input)
              end
            end

            type_enum = FFI::TensorElementDataType[tensor_type]
          else
            unsupported_type("input", inp[:type])
          end

          check_status api[:CreateTensorWithDataAsOrtValue].call(allocator_info.read_pointer, input_tensor_values, input_tensor_values.size, input_node_dims, shape.size, type_enum, input_tensor[idx])

          refs << input_node_dims
          refs << input_tensor_values
        end
      end

      refs << allocator_info

      input_tensor
    ensure
      release :MemoryInfo, allocator_info
    end

    def create_node_names(names, refs)
      str_ptrs = names.map { |v| ::FFI::MemoryPointer.from_string(v) }
      refs << str_ptrs

      ptr = ::FFI::MemoryPointer.new(:pointer, names.size)
      ptr.write_array_of_pointer(str_ptrs)
      ptr
    end

    def create_from_onnx_value(out_ptr, output_type)
      out_type = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetValueType].call(out_ptr, out_type)
      type = FFI::OnnxType[out_type.read_int]

      case type
      when :tensor
        typeinfo = ::FFI::MemoryPointer.new(:pointer)
        check_status api[:GetTensorTypeAndShape].call(out_ptr, typeinfo)

        type, shape = tensor_type_and_shape(typeinfo)

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
            unsupported_type("element", type) unless numo_type
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
              unsupported_type("element", type)
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
          check_status api[:GetValue].call(out_ptr, i, @allocator.read_pointer, seq)
          create_from_onnx_value(seq.read_pointer, output_type)
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
          unsupported_type("element", elem_type)
        end
      else
        unsupported_type("ONNX", type)
      end
    ensure
      api[:ReleaseValue].call(out_ptr) unless out_ptr.null?
    end

    def create_strings_from_onnx_value(out_ptr, output_tensor_size, result)
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

    def read_pointer
      @session.read_pointer
    end

    def check_status(status)
      unless status.null?
        message = api[:GetErrorMessage].call(status).read_string
        api[:ReleaseStatus].call(status)
        raise Error, message
      end
    end

    def node_info(typeinfo)
      onnx_type = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetOnnxTypeFromTypeInfo].call(typeinfo.read_pointer, onnx_type)

      type = FFI::OnnxType[onnx_type.read_int]
      case type
      when :tensor
        tensor_info = ::FFI::MemoryPointer.new(:pointer)
        # don't free tensor_info
        check_status api[:CastTypeInfoToTensorInfo].call(typeinfo.read_pointer, tensor_info)

        type, shape = tensor_type_and_shape(tensor_info)
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
        unsupported_type("ONNX", type)
      end
    ensure
      release :TypeInfo, typeinfo
    end

    def tensor_type_and_shape(tensor_info)
      type = ::FFI::MemoryPointer.new(:int)
      check_status api[:GetTensorElementType].call(tensor_info.read_pointer, type)

      num_dims_ptr = ::FFI::MemoryPointer.new(:size_t)
      check_status api[:GetDimensionsCount].call(tensor_info.read_pointer, num_dims_ptr)
      num_dims = num_dims_ptr.read(:size_t)

      node_dims = ::FFI::MemoryPointer.new(:int64, num_dims)
      check_status api[:GetDimensions].call(tensor_info.read_pointer, node_dims, num_dims)
      dims = node_dims.read_array_of_int64(num_dims)

      # TODO uncomment in 0.8.0
      # symbolic_dims = ::FFI::MemoryPointer.new(:pointer, num_dims)
      # check_status api[:GetSymbolicDimensions].call(tensor_info.read_pointer, symbolic_dims, num_dims)
      # named_dims = num_dims.times.map { |i| symbolic_dims[i].read_pointer.read_string }
      # dims = named_dims.zip(dims).map { |n, d| n.empty? ? d : n }

      [type.read_int, dims]
    end

    def unsupported_type(name, type)
      raise Error, "Unsupported #{name} type: #{type}"
    end

    def tensor_types
      @tensor_types ||= [:float, :uint8, :int8, :uint16, :int16, :int32, :int64, :bool, :double, :uint32, :uint64].map { |v| ["tensor(#{v})", v] }.to_h
    end

    def numo_array?(obj)
      defined?(Numo::NArray) && obj.is_a?(Numo::NArray)
    end

    def numo_types
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

    def api
      self.class.api
    end

    def release(*args)
      self.class.release(*args)
    end

    def allocator_free(ptr)
      api[:AllocatorFree].call(@allocator.read_pointer, ptr.read_pointer)
    end

    def self.api
      @api ||= FFI.OrtGetApiBase[:GetApi].call(FFI::ORT_API_VERSION)
    end

    def self.release(type, pointer)
      api[:"Release#{type}"].call(pointer.read_pointer) if pointer && !pointer.null?
    end

    def self.finalize(session)
      # must use proc instead of stabby lambda
      proc { release :Session, session }
    end

    # wide string on Windows
    # char string on Linux
    # see ORTCHAR_T in onnxruntime_c_api.h
    def ort_string(str)
      if Gem.win_platform?
        max = str.size + 1 # for null byte
        dest = ::FFI::MemoryPointer.new(:wchar_t, max)
        ret = FFI::Libc.mbstowcs(dest, str, max)
        raise Error, "Expected mbstowcs to return #{str.size}, got #{ret}" if ret != str.size
        dest
      else
        str
      end
    end

    def env
      # use mutex for thread-safety
      Utils.mutex.synchronize do
        @@env ||= begin
          env = ::FFI::MemoryPointer.new(:pointer)
          check_status api[:CreateEnv].call(3, "Default", env)
          at_exit { release :Env, env }
          # disable telemetry
          # https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md
          check_status api[:DisableTelemetryEvents].call(env)
          env
        end
      end
    end
  end
end
