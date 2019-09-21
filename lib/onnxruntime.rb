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
    attr_accessor :ffi_lib, :mutex
  end
  lib_name = ::FFI.map_library_name("onnxruntime")
  vendor_lib = File.expand_path("../vendor/#{lib_name}", __dir__)
  self.ffi_lib = ["onnxruntime", vendor_lib]
  self.mutex = Mutex.new # private

  def self.lib_version
    FFI.OrtGetVersionString
  end

  # friendlier error message
  autoload :FFI, "onnxruntime/ffi"
end
