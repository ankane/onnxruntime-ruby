module OnnxRuntime
  module FFI
    extend ::FFI::Library

    begin
      ffi_lib OnnxRuntime.ffi_lib
    rescue LoadError => e
      if e.message.include?("Library not loaded: /usr/local/opt/libomp/lib/libomp.dylib") && e.message.include?("Reason: image not found")
        raise LoadError, "OpenMP not found. Run `brew install libomp`"
      else
        raise e
      end
    end

    # https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h
    # keep same order

    # enums
    TensorElementDataType = enum(:undefined, :float, :uint8, :int8, :uint16, :int16, :int32, :int64, :string, :bool, :float16, :double, :uint32, :uint64, :complex64, :complex128, :bfloat16)
    OnnxType = enum(:unknown, :tensor, :sequence, :map, :opaque, :sparsetensor)

    class Api < ::FFI::Struct
      layout \
        :CreateStatus, callback(%i[int string], :pointer),
        :GetErrorCode, callback(%i[pointer], :pointer),
        :GetErrorMessage, callback(%i[pointer], :pointer),
        :CreateEnv, callback(%i[int string pointer], :pointer),
        :CreateEnvWithCustomLogger, callback(%i[], :pointer),
        :EnableTelemetryEvents, callback(%i[pointer], :pointer),
        :DisableTelemetryEvents, callback(%i[pointer], :pointer),
        :CreateSession, callback(%i[pointer pointer pointer pointer], :pointer),
        :CreateSessionFromArray, callback(%i[pointer pointer size_t pointer pointer], :pointer),
        :Run, callback(%i[pointer pointer pointer pointer size_t pointer size_t pointer], :pointer),
        :CreateSessionOptions, callback(%i[pointer], :pointer),
        :SetOptimizedModelFilePath, callback(%i[pointer pointer], :pointer),
        :CloneSessionOptions, callback(%i[], :pointer),
        :SetSessionExecutionMode, callback(%i[pointer int], :pointer),
        :EnableProfiling, callback(%i[pointer pointer], :pointer),
        :DisableProfiling, callback(%i[pointer], :pointer),
        :EnableMemPattern, callback(%i[pointer], :pointer),
        :DisableMemPattern, callback(%i[pointer], :pointer),
        :EnableCpuMemArena, callback(%i[pointer], :pointer),
        :DisableCpuMemArena, callback(%i[pointer], :pointer),
        :SetSessionLogId, callback(%i[pointer string], :pointer),
        :SetSessionLogVerbosityLevel, callback(%i[pointer int], :pointer),
        :SetSessionLogSeverityLevel, callback(%i[pointer int], :pointer),
        :SetSessionGraphOptimizationLevel, callback(%i[pointer int], :pointer),
        :SetIntraOpNumThreads, callback(%i[pointer int], :pointer),
        :SetInterOpNumThreads, callback(%i[pointer int], :pointer),
        :CreateCustomOpDomain, callback(%i[], :pointer),
        :CustomOpDomain_Add, callback(%i[], :pointer),
        :AddCustomOpDomain, callback(%i[], :pointer),
        :RegisterCustomOpsLibrary, callback(%i[], :pointer),
        :SessionGetInputCount, callback(%i[pointer pointer], :pointer),
        :SessionGetOutputCount, callback(%i[pointer pointer], :pointer),
        :SessionGetOverridableInitializerCount, callback(%i[], :pointer),
        :SessionGetInputTypeInfo, callback(%i[pointer size_t pointer], :pointer),
        :SessionGetOutputTypeInfo, callback(%i[pointer size_t pointer], :pointer),
        :SessionGetOverridableInitializerTypeInfo, callback(%i[], :pointer),
        :SessionGetInputName, callback(%i[pointer size_t pointer pointer], :pointer),
        :SessionGetOutputName, callback(%i[pointer size_t pointer pointer], :pointer),
        :SessionGetOverridableInitializerName, callback(%i[], :pointer),
        :CreateRunOptions, callback(%i[pointer], :pointer),
        :RunOptionsSetRunLogVerbosityLevel, callback(%i[pointer int], :pointer),
        :RunOptionsSetRunLogSeverityLevel, callback(%i[pointer int], :pointer),
        :RunOptionsSetRunTag, callback(%i[pointer string], :pointer),
        :RunOptionsGetRunLogVerbosityLevel, callback(%i[], :pointer),
        :RunOptionsGetRunLogSeverityLevel, callback(%i[], :pointer),
        :RunOptionsGetRunTag, callback(%i[], :pointer),
        :RunOptionsSetTerminate, callback(%i[pointer], :pointer),
        :RunOptionsUnsetTerminate, callback(%i[pointer], :pointer),
        :CreateTensorAsOrtValue, callback(%i[pointer pointer size_t int pointer], :pointer),
        :CreateTensorWithDataAsOrtValue, callback(%i[pointer pointer size_t pointer size_t int pointer], :pointer),
        :IsTensor, callback(%i[], :pointer),
        :GetTensorMutableData, callback(%i[pointer pointer], :pointer),
        :FillStringTensor, callback(%i[pointer pointer size_t], :pointer),
        :GetStringTensorDataLength, callback(%i[pointer pointer], :pointer),
        :GetStringTensorContent, callback(%i[pointer pointer size_t pointer size_t], :pointer),
        :CastTypeInfoToTensorInfo, callback(%i[pointer pointer], :pointer),
        :GetOnnxTypeFromTypeInfo, callback(%i[pointer pointer], :pointer),
        :CreateTensorTypeAndShapeInfo, callback(%i[], :pointer),
        :SetTensorElementType, callback(%i[], :pointer),
        :SetDimensions, callback(%i[], :pointer),
        :GetTensorElementType, callback(%i[pointer pointer], :pointer),
        :GetDimensionsCount, callback(%i[pointer pointer], :pointer),
        :GetDimensions, callback(%i[pointer pointer size_t], :pointer),
        :GetSymbolicDimensions, callback(%i[], :pointer),
        :GetTensorShapeElementCount, callback(%i[pointer pointer], :pointer),
        :GetTensorTypeAndShape, callback(%i[pointer pointer], :pointer),
        :GetTypeInfo, callback(%i[pointer pointer], :pointer),
        :GetValueType, callback(%i[pointer pointer], :pointer),
        :CreateMemoryInfo, callback(%i[], :pointer),
        :CreateCpuMemoryInfo, callback(%i[int int pointer], :pointer),
        :CompareMemoryInfo, callback(%i[], :pointer),
        :MemoryInfoGetName, callback(%i[], :pointer),
        :MemoryInfoGetId, callback(%i[], :pointer),
        :MemoryInfoGetMemType, callback(%i[], :pointer),
        :MemoryInfoGetType, callback(%i[], :pointer),
        :AllocatorAlloc, callback(%i[], :pointer),
        :AllocatorFree, callback(%i[pointer pointer], :pointer),
        :AllocatorGetInfo, callback(%i[], :pointer),
        :GetAllocatorWithDefaultOptions, callback(%i[pointer], :pointer),
        :AddFreeDimensionOverride, callback(%i[], :pointer),
        :GetValue, callback(%i[pointer int pointer pointer], :pointer),
        :GetValueCount, callback(%i[pointer pointer], :pointer),
        :CreateValue, callback(%i[], :pointer),
        :CreateOpaqueValue, callback(%i[], :pointer),
        :GetOpaqueValue, callback(%i[], :pointer),
        :KernelInfoGetAttribute_float, callback(%i[], :pointer),
        :KernelInfoGetAttribute_int64, callback(%i[], :pointer),
        :KernelInfoGetAttribute_string, callback(%i[], :pointer),
        :KernelContext_GetInputCount, callback(%i[], :pointer),
        :KernelContext_GetOutputCount, callback(%i[], :pointer),
        :KernelContext_GetInput, callback(%i[], :pointer),
        :KernelContext_GetOutput, callback(%i[], :pointer),
        :ReleaseEnv, callback(%i[pointer], :void),
        :ReleaseStatus, callback(%i[pointer], :void),
        :ReleaseMemoryInfo, callback(%i[pointer], :void),
        :ReleaseSession, callback(%i[pointer], :void),
        :ReleaseValue, callback(%i[pointer], :void),
        :ReleaseRunOptions, callback(%i[pointer], :void),
        :ReleaseTypeInfo, callback(%i[pointer], :void),
        :ReleaseTensorTypeAndShapeInfo, callback(%i[pointer], :void),
        :ReleaseSessionOptions, callback(%i[pointer], :void),
        :ReleaseCustomOpDomain, callback(%i[pointer], :void),
        :GetDenotationFromTypeInfo, callback(%i[], :pointer),
        :CastTypeInfoToMapTypeInfo, callback(%i[pointer pointer], :pointer),
        :CastTypeInfoToSequenceTypeInfo, callback(%i[pointer pointer], :pointer),
        :GetMapKeyType, callback(%i[pointer pointer], :pointer),
        :GetMapValueType, callback(%i[pointer pointer], :pointer),
        :GetSequenceElementType, callback(%i[pointer pointer], :pointer),
        :ReleaseMapTypeInfo, callback(%i[pointer], :void),
        :ReleaseSequenceTypeInfo, callback(%i[pointer], :void),
        :SessionEndProfiling, callback(%i[pointer pointer pointer], :pointer),
        :SessionGetModelMetadata, callback(%i[pointer pointer], :pointer),
        :ModelMetadataGetProducerName, callback(%i[pointer pointer pointer], :pointer),
        :ModelMetadataGetGraphName, callback(%i[pointer pointer pointer], :pointer),
        :ModelMetadataGetDomain, callback(%i[pointer pointer pointer], :pointer),
        :ModelMetadataGetDescription, callback(%i[pointer pointer pointer], :pointer),
        :ModelMetadataLookupCustomMetadataMap, callback(%i[pointer pointer pointer pointer], :pointer),
        :ModelMetadataGetVersion, callback(%i[pointer pointer], :pointer),
        :ReleaseModelMetadata, callback(%i[pointer], :void),
        :CreateEnvWithGlobalThreadPools, callback(%i[], :pointer),
        :DisablePerSessionThreads, callback(%i[], :pointer),
        :CreateThreadingOptions, callback(%i[], :pointer),
        :ReleaseThreadingOptions, callback(%i[], :pointer),
        :ModelMetadataGetCustomMetadataMapKeys, callback(%i[pointer pointer pointer pointer], :pointer),
        :AddFreeDimensionOverrideByName, callback(%i[], :pointer),
        :GetAvailableProviders, callback(%i[pointer pointer], :pointer),
        :ReleaseAvailableProviders, callback(%i[pointer int], :pointer)
    end

    class ApiBase < ::FFI::Struct
      # use uint32 instead of uint32_t
      # to prevent "unable to resolve type" error on Ubuntu
      layout \
        :GetApi, callback(%i[uint32], Api.by_ref),
        :GetVersionString, callback(%i[], :pointer)
    end

    attach_function :OrtGetApiBase, %i[], ApiBase.by_ref

    if Gem.win_platform?
      class Libc
        extend ::FFI::Library
        ffi_lib ::FFI::Library::LIBC
        attach_function :mbstowcs, %i[pointer string size_t], :size_t
      end
    end
  end
end
