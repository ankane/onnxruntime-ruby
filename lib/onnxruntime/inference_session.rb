module OnnxRuntime
  class InferenceSession
    attr_reader :inputs, :outputs

    def initialize(path_or_bytes, enable_cpu_mem_arena: true, enable_mem_pattern: true, enable_profiling: false, execution_mode: nil, free_dimension_overrides_by_denotation: nil, free_dimension_overrides_by_name: nil, graph_optimization_level: nil, inter_op_num_threads: nil, intra_op_num_threads: nil, log_severity_level: nil, log_verbosity_level: nil, logid: nil, optimized_model_filepath: nil, profile_file_prefix: nil, session_config_entries: nil, providers: [])
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
      providers.each do |provider|
        unless self.providers.include?(provider)
          warn "Provider not available: #{provider}"
          next
        end

        case provider
        when "CUDAExecutionProvider"
          cuda_options = ::FFI::MemoryPointer.new(:pointer)
          check_status api[:CreateCUDAProviderOptions].call(cuda_options)
          check_status api[:SessionOptionsAppendExecutionProvider_CUDA_V2].call(session_options.read_pointer, cuda_options.read_pointer)
          release :CUDAProviderOptions, cuda_options
        when "CoreMLExecutionProvider"
          unless FFI.respond_to?(:OrtSessionOptionsAppendExecutionProvider_CoreML)
            raise ArgumentError, "Provider not available: #{provider}"
          end

          coreml_flags = 0
          check_status FFI.OrtSessionOptionsAppendExecutionProvider_CoreML(session_options.read_pointer, coreml_flags)
        when "CPUExecutionProvider"
          break
        else
          raise ArgumentError, "Provider not supported: #{provider}"
        end
      end

      @session = load_session(path_or_bytes, session_options)
      ObjectSpace.define_finalizer(@session, self.class.finalize(read_pointer.to_i))

      @allocator = Utils.allocator
      @inputs = load_inputs
      @outputs = load_outputs
    ensure
      release :SessionOptions, session_options
    end

    def run(output_names, input_feed, log_severity_level: nil, log_verbosity_level: nil, logid: nil, terminate: nil, output_type: :ruby)
      if ![:ruby, :numo].include?(output_type)
        raise ArgumentError, "Invalid output type: #{output_type}"
      end

      ort_values = input_feed.keys.zip(create_input_tensor(input_feed)).to_h

      outputs = run_with_ort_values(output_names, ort_values, log_severity_level: log_severity_level, log_verbosity_level: log_verbosity_level, logid: logid, terminate: terminate)

      outputs.map { |v| output_type == :numo ? v.numo : v.to_a }
    end

    # TODO support logid
    def run_with_ort_values(output_names, input_feed, log_severity_level: nil, log_verbosity_level: nil, logid: nil, terminate: nil)
      # pointer references
      refs = []

      input_tensor = ::FFI::MemoryPointer.new(:pointer, input_feed.size)
      input_feed.each_with_index do |(_, input), i|
        input_tensor[i].write_pointer(input.send(:out_ptr))
      end

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

      output_names.size.times.map { |i| OrtValue.new(output_tensor[i]) }
    ensure
      release :RunOptions, run_options
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
          false
        end

      if from_memory
        check_status api[:CreateSessionFromArray].call(env.read_pointer, path_or_bytes, path_or_bytes.bytesize, session_options.read_pointer, session)
      else
        check_status api[:CreateSession].call(env.read_pointer, ort_string(path_or_bytes), session_options.read_pointer, session)
      end
      session
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
        inputs << {name: name_ptr.read_pointer.read_string}.merge(Utils.node_info(typeinfo))
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
        outputs << {name: name_ptr.read_pointer.read_string}.merge(Utils.node_info(typeinfo))
        allocator_free name_ptr
      end
      outputs
    end

    def create_input_tensor(input_feed)
      input_feed.map.with_index do |(input_name, input), idx|
        # TODO support more types
        inp = @inputs.find { |i| i[:name] == input_name.to_s }
        raise Error, "Unknown input: #{input_name}" unless inp

        input = input.to_a unless input.is_a?(Array) || Utils.numo_array?(input)

        if inp[:type] == "tensor(string)"
          OrtValue.ortvalue_from_array(input, element_type: :string)
        elsif (tensor_type = tensor_types[inp[:type]])
          OrtValue.ortvalue_from_array(input, element_type: tensor_type)
        else
          Utils.unsupported_type("input", inp[:type])
        end
      end
    end

    def create_node_names(names, refs)
      str_ptrs = names.map { |v| ::FFI::MemoryPointer.from_string(v) }
      refs << str_ptrs

      ptr = ::FFI::MemoryPointer.new(:pointer, names.size)
      ptr.write_array_of_pointer(str_ptrs)
      ptr
    end

    def read_pointer
      @session.read_pointer
    end

    def check_status(status)
      Utils.check_status(status)
    end

    def tensor_types
      @tensor_types ||= [:float, :uint8, :int8, :uint16, :int16, :int32, :int64, :bool, :double, :uint32, :uint64].map { |v| ["tensor(#{v})", v] }.to_h
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
      FFI.api
    end

    def self.release(type, pointer)
      Utils.release(type, pointer)
    end

    def self.finalize(addr)
      # must use proc instead of stabby lambda
      proc { api[:ReleaseSession].call(::FFI::Pointer.new(:pointer, addr)) }
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
