require_relative "lib/onnxruntime/version"

Gem::Specification.new do |spec|
  spec.name          = "onnxruntime"
  spec.version       = OnnxRuntime::VERSION
  spec.summary       = "High performance scoring engine for ML models"
  spec.homepage      = "https://github.com/ankane/onnxruntime-ruby"
  spec.license       = "MIT"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib,vendor}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.4"

  spec.add_dependency "ffi"
end
