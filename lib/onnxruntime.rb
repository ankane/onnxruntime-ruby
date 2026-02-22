# dependencies
require "ffi"

# modules
require_relative "onnxruntime/datasets"
require_relative "onnxruntime/inference_session"
require_relative "onnxruntime/model"
require_relative "onnxruntime/ort_value"
require_relative "onnxruntime/pointer"
require_relative "onnxruntime/utils"
require_relative "onnxruntime/version"

module OnnxRuntime
  class Error < StandardError; end

  class << self
    attr_accessor :ffi_lib
  end
  vendor = File.expand_path("../vendor", __dir__)
  lib_name =
    if Gem.win_platform?
      "#{vendor}/onnxruntime.dll"
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      if RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i
        "#{vendor}/libonnxruntime.arm64.dylib"
      else
        "/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib"
      end
    else
      if RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i
        "#{vendor}/libonnxruntime.arm64.so"
      else
        "#{vendor}/libonnxruntime.so"
      end
    end
  self.ffi_lib = [lib_name]

  def self.lib_version
    FFI.OrtGetApiBase[:GetVersionString].call.read_string
  end

  # friendlier error message
  autoload :FFI, "onnxruntime/ffi"
end
