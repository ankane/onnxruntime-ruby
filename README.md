# ONNX Runtime

:fire: [ONNX Runtime](https://github.com/Microsoft/onnxruntime) - the high performance scoring engine for ML models - for Ruby

[![Build Status](https://travis-ci.org/ankane/onnxruntime.svg?branch=master)](https://travis-ci.org/ankane/onnxruntime) [![Build status](https://ci.appveyor.com/api/projects/status/f2bq6ruqjf4jx671/branch/master?svg=true)](https://ci.appveyor.com/project/ankane/onnxruntime/branch/master)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem 'onnxruntime'
```

## Getting Started

This project follows the [Python API](https://microsoft.github.io/onnxruntime/api_summary.html).

Load a model and run it

```ruby
session = OnnxRuntime::InferenceSession.new("model.onnx")
session.run(nil, x: [1, 2, 3])
```

Get inputs

```ruby
session.inputs
```

Get outputs

```ruby
session.outputs
```

Load a model from a string

```ruby
byte_str = File.binread("model.onnx")
session = OnnxRuntime::InferenceSession.new(byte_str)
```

Get specific outputs

```ruby
session.run(["label"], x: [1, 2, 3])
```

## History

View the [changelog](https://github.com/ankane/onnxruntime/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/onnxruntime/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/onnxruntime/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features
