# dependencies
require "ffi"

# modules
require "onnxruntime/datasets"
require "onnxruntime/inference_session"
require "onnxruntime/model"
require "onnxruntime/utils"
require "onnxruntime/version"

module OnnxRuntime
  class Error < StandardError; end

  class << self
    attr_accessor :ffi_lib
  end
  lib_name = ::FFI.map_library_name("onnxruntime")
  vendor_lib = File.expand_path("../vendor/#{lib_name}", __dir__)
  self.ffi_lib = [vendor_lib]

  def self.lib_version
    FFI.OrtGetApiBase[:GetVersionString].call.read_string
  end

  # friendlier error message
  autoload :FFI, "onnxruntime/ffi"
end
