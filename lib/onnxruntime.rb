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
  lib_name =
    if Gem.win_platform?
      "onnxruntime.dll"
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      if RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i
        "libonnxruntime.arm64.dylib"
      else
        "libonnxruntime.dylib"
      end
    else
      if RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i
        "libonnxruntime.arm64.so"
      else
        "libonnxruntime.so"
      end
    end
  vendor_lib = File.expand_path("../vendor/#{lib_name}", __dir__)
  self.ffi_lib = [vendor_lib]

  def self.lib_version
    FFI.OrtGetApiBase[:GetVersionString].call.read_string
  end

  # friendlier error message
  autoload :FFI, "onnxruntime/ffi"
end
