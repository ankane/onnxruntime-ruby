# ONNX Runtime

:fire: [ONNX Runtime](https://github.com/Microsoft/onnxruntime) - the high performance scoring engine for ML models - for Ruby

[![Build Status](https://travis-ci.org/ankane/onnxruntime.svg?branch=master)](https://travis-ci.org/ankane/onnxruntime) [![Build status](https://ci.appveyor.com/api/projects/status/f2bq6ruqjf4jx671/branch/master?svg=true)](https://ci.appveyor.com/project/ankane/onnxruntime/branch/master)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem 'onnxruntime'
```

## Getting Started

Load a model and make predictions

```ruby
model = OnnxRuntime::Model.new("model.onnx")
model.predict(x: [1, 2, 3])
```

Get inputs

```ruby
model.inputs
```

Get outputs

```ruby
model.outputs
```

Load a model from a string

```ruby
byte_str = File.binread("model.onnx")
model = OnnxRuntime::Model.new(byte_str)
```

Get specific outputs

```ruby
model.predict({x: [1, 2, 3]}, output_names: ["label"])
```

## Inference Session API

You can also use the Inference Session API, which follows the [Python API](https://microsoft.github.io/onnxruntime/api_summary.html).

```ruby
session = OnnxRuntime::InferenceSession.new("model.onnx")
session.run(nil, {x: [1, 2, 3]})
```

## History

View the [changelog](https://github.com/ankane/onnxruntime/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/onnxruntime/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/onnxruntime/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features
