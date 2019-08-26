require_relative "lib/onnxruntime/version"

Gem::Specification.new do |spec|
  spec.name          = "onnxruntime"
  spec.version       = OnnxRuntime::VERSION
  spec.summary       = "High performance scoring engine for ML models"
  spec.homepage      = "https://github.com/ankane/onnxruntime"
  spec.license       = "MIT"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@chartkick.com"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.4"

  spec.add_dependency "ffi"

  spec.add_development_dependency "bundler"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "minitest", ">= 5"
end
