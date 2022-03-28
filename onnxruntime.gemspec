require_relative "lib/onnxruntime/version"

Gem::Specification.new do |spec|
  spec.name          = "onnxruntime"
  spec.version       = OnnxRuntime::VERSION
  spec.summary       = "High performance scoring engine for ML models"
  spec.homepage      = "https://github.com/ankane/onnxruntime-ruby"
  spec.license       = "MIT"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*", "vendor/{LICENSE,*.txt}"]
  spec.require_path  = "lib"

  case spec.platform.to_s
  when "x86_64-linux"
    spec.files << "vendor/libonnxruntime.so"
  when "aarch64-linux"
    spec.files << "vendor/libonnxruntime.arm64.so"
  when "x86_64-darwin"
    spec.files << "vendor/libonnxruntime.dylib"
  when "arm64-darwin"
    spec.files << "vendor/libonnxruntime.arm64.dylib"
  when "x64-mingw"
    spec.files << "vendor/onnxruntime.dll"
  else
    spec.files.concat(Dir["vendor/*.{dll,dylib,so}"])
  end

  spec.required_ruby_version = ">= 2.4"

  spec.add_dependency "ffi"
end
