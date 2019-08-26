module OnnxRuntime
  module FFI
    extend ::FFI::Library

    begin
      ffi_lib OnnxRuntime.ffi_lib
    rescue LoadError => e
      raise e if ENV["ONNXRUNTIME_DEBUG"]
      raise LoadError, "Could not find ONNX Runtime"
    end

    # https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h
    # keep same order

    # enums
    TensorElementDataType = enum(:undefined, :float, :uint8, :int8, :uint16, :int16, :int32, :int64, :string, :bool, :float16, :double, :uint32, :uint64, :complex64, :complex128, :bfloat16)
    OnnxType = enum(:unknown, :tensor, :sequence, :map, :opaque, :sparsetensor)

    # session
    attach_function :OrtCreateEnv, %i[int string pointer], :pointer
    attach_function :OrtCreateSession, %i[pointer string pointer pointer], :pointer
    attach_function :OrtCreateSessionFromArray, %i[pointer pointer size_t pointer pointer], :pointer
    attach_function :OrtRun, %i[pointer pointer pointer pointer size_t pointer size_t pointer], :pointer
    attach_function :OrtCreateSessionOptions, %i[pointer], :pointer
    attach_function :OrtSetSessionGraphOptimizationLevel, %i[pointer int], :pointer
    attach_function :OrtSetSessionThreadPoolSize, %i[pointer int], :pointer

    # input and output
    attach_function :OrtSessionGetInputCount, %i[pointer pointer], :pointer
    attach_function :OrtSessionGetOutputCount, %i[pointer pointer], :pointer
    attach_function :OrtSessionGetInputTypeInfo, %i[pointer size_t pointer], :pointer
    attach_function :OrtSessionGetOutputTypeInfo, %i[pointer size_t pointer], :pointer
    attach_function :OrtSessionGetInputName, %i[pointer size_t pointer pointer], :pointer
    attach_function :OrtSessionGetOutputName, %i[pointer size_t pointer pointer], :pointer

    # tensor
    attach_function :OrtCreateTensorWithDataAsOrtValue, %i[pointer pointer size_t pointer size_t int pointer], :pointer
    attach_function :OrtGetTensorMutableData, %i[pointer pointer], :pointer
    attach_function :OrtIsTensor, %i[pointer pointer], :pointer
    attach_function :OrtCastTypeInfoToTensorInfo, %i[pointer pointer], :pointer
    attach_function :OrtOnnxTypeFromTypeInfo, %i[pointer pointer], :pointer
    attach_function :OrtGetTensorElementType, %i[pointer pointer], :pointer
    attach_function :OrtGetDimensionsCount, %i[pointer pointer], :pointer
    attach_function :OrtGetDimensions, %i[pointer pointer size_t], :pointer
    attach_function :OrtGetTensorShapeElementCount, %i[pointer pointer], :pointer
    attach_function :OrtGetTensorTypeAndShape, %i[pointer pointer], :pointer

    # value
    attach_function :OrtGetTypeInfo, %i[pointer pointer], :pointer
    attach_function :OrtGetValueType, %i[pointer pointer], :pointer

    # maps and sequences
    attach_function :OrtGetValue, %i[pointer int pointer pointer], :pointer
    attach_function :OrtGetValueCount, %i[pointer pointer], :pointer

    # version
    attach_function :OrtGetVersionString, %i[], :string

    # error
    attach_function :OrtGetErrorMessage, %i[pointer], :string

    # allocator
    attach_function :OrtCreateCpuAllocatorInfo, %i[int int pointer], :pointer
    attach_function :OrtCreateDefaultAllocator, %i[pointer], :pointer

    # release
    attach_function :OrtReleaseEnv, %i[pointer], :pointer
    attach_function :OrtReleaseTypeInfo, %i[pointer], :pointer
    attach_function :OrtReleaseStatus, %i[pointer], :pointer
  end
end
