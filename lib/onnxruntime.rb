# dependencies
require "ffi"

# modules
require "onnxruntime/inference_session"
require "onnxruntime/model"
require "onnxruntime/utils"
require "onnxruntime/version"

module OnnxRuntime
  class Error < StandardError; end

  class << self
    attr_accessor :ffi_lib
  end
  self.ffi_lib = ["onnxruntime"]

  def self.lib_version
    FFI.OrtGetVersionString
  end

  # friendlier error message
  autoload :FFI, "onnxruntime/ffi"
end
