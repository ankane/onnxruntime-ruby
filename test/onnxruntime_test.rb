require_relative "test_helper"

class OnnxRuntimeTest < Minitest::Test
  def test_lib_version
    assert OnnxRuntime.lib_version
  end

  def test_basic
    model = OnnxRuntime::Model.new("test/support/model.onnx")

    expected = [{name: "x", type: "tensor(float)", shape: [3, 4, 5]}]
    assert_equal expected, model.inputs

    expected = [{name: "y", type: "tensor(float)", shape: [3, 4, 5]}]
    assert_equal expected, model.outputs

    x = [[[0.5488135,  0.71518934, 0.60276335, 0.5448832,  0.4236548 ],
          [0.6458941,  0.4375872,  0.891773,   0.96366274, 0.3834415 ],
          [0.79172504, 0.5288949,  0.56804454, 0.92559665, 0.07103606],
          [0.0871293,  0.0202184,  0.83261985, 0.77815676, 0.87001216]],

         [[0.9786183,  0.7991586,  0.46147937, 0.7805292,  0.11827443],
          [0.639921,   0.14335328, 0.9446689,  0.5218483,  0.41466194],
          [0.2645556,  0.7742337,  0.45615032, 0.56843394, 0.0187898 ],
          [0.6176355,  0.6120957,  0.616934,   0.94374806, 0.6818203 ]],

         [[0.3595079,  0.43703195, 0.6976312,  0.06022547, 0.6667667 ],
          [0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076],
          [0.57019675, 0.43860152, 0.9883738,  0.10204481, 0.20887676],
          [0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256 ]]]

    output = model.predict({x: x})
    assert_elements_in_delta [0.6338603, 0.6715468, 0.6462883, 0.6329476, 0.6043575], output["y"].first.first
  end

  def test_numo_narray
    skip if RUBY_PLATFORM == "java"

    model = OnnxRuntime::Model.new("test/support/model.onnx")

    x = [[[0.5488135,  0.71518934, 0.60276335, 0.5448832,  0.4236548 ],
          [0.6458941,  0.4375872,  0.891773,   0.96366274, 0.3834415 ],
          [0.79172504, 0.5288949,  0.56804454, 0.92559665, 0.07103606],
          [0.0871293,  0.0202184,  0.83261985, 0.77815676, 0.87001216]],

         [[0.9786183,  0.7991586,  0.46147937, 0.7805292,  0.11827443],
          [0.639921,   0.14335328, 0.9446689,  0.5218483,  0.41466194],
          [0.2645556,  0.7742337,  0.45615032, 0.56843394, 0.0187898 ],
          [0.6176355,  0.6120957,  0.616934,   0.94374806, 0.6818203 ]],

         [[0.3595079,  0.43703195, 0.6976312,  0.06022547, 0.6667667 ],
          [0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076],
          [0.57019675, 0.43860152, 0.9883738,  0.10204481, 0.20887676],
          [0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256 ]]]

    x = Numo::NArray.cast(x)

    output = model.predict({x: x})
    assert_elements_in_delta [0.6338603, 0.6715468, 0.6462883, 0.6329476, 0.6043575], output["y"].first.first
  end

  def test_string
    contents = File.binread("test/support/model.onnx")
    model = OnnxRuntime::Model.new(contents)
    expected = [{name: "x", type: "tensor(float)", shape: [3, 4, 5]}]
    assert_equal expected, model.inputs
  end

  def test_io
    model = File.open("test/support/model.onnx", "rb") { |f| OnnxRuntime::Model.new(f) }
    expected = [{name: "x", type: "tensor(float)", shape: [3, 4, 5]}]
    assert_equal expected, model.inputs
  end

  def test_stringio
    require "stringio"

    contents = StringIO.new(File.binread("test/support/model.onnx"))
    model = OnnxRuntime::Model.new(contents)
    expected = [{name: "x", type: "tensor(float)", shape: [3, 4, 5]}]
    assert_equal expected, model.inputs
  end

  def test_lightgbm
    model = OnnxRuntime::Model.new("test/support/lightgbm.onnx")

    expected = [{name: "input", type: "tensor(float)", shape: [1, 2]}]
    assert_equal expected, model.inputs

    expected = [{name: "label", type: "tensor(int64)", shape: [1]}, {name: "probabilities", type: "seq(map(int64,tensor(float)))", shape: []}]
    assert_equal expected, model.outputs

    x = [[5.8, 2.8]]

    output = model.predict({input: x}) #, output_names: ["label"])
    assert_equal [1], output["label"]
    probabilities = output["probabilities"].first
    assert_equal [0, 1, 2], probabilities.keys
    assert_elements_in_delta [0.2593829035758972, 0.409047931432724, 0.3315691649913788], probabilities.values

    x2 = [[5.8, 2.8],
          [6.0, 2.2],
          [5.5, 4.2],
          [7.3, 2.9],
          [5.0, 3.4]]

    labels = []
    x2.each do |xi|
      output = model.predict({input: [xi]})
      labels << output["label"].first
    end
    assert_equal [1, 1, 0, 2, 0], labels
  end

  def test_random_forest
    model = OnnxRuntime::Model.new("test/support/randomforest.onnx")

    expected = [{name: "float_input", type: "tensor(float)", shape: [1, 4]}]
    assert_equal expected, model.inputs

    expected = [{name: "output_label", type: "tensor(int64)", shape: [1]}, {name: "output_probability", type: "seq(map(int64,tensor(float)))", shape: []}]
    assert_equal expected, model.outputs

    x = [[5.8, 2.8, 5.1, 2.4]]

    output = model.predict({float_input: x})
    assert_equal [2], output["output_label"]
    probabilities = output["output_probability"].first
    assert_equal [0, 1, 2], probabilities.keys
    assert_elements_in_delta [0.0, 0.0, 1.0000001192092896], probabilities.values
  end

  def test_output_names
    model = OnnxRuntime::Model.new("test/support/lightgbm.onnx")
    output = model.predict({input: [[5.8, 2.8]]}, output_names: ["label"])
    assert_equal ["label"], output.keys
  end

  def test_session_options
    optimized_path = "#{Dir.tmpdir}/optimized.onnx"

    session_options = {
      inter_op_num_threads: 1,
      intra_op_num_threads: 1,
      log_severity_level: 4,
      log_verbosity_level: 4,
      logid: "test",
      optimized_model_filepath: optimized_path
    }

    sess = OnnxRuntime::InferenceSession.new("test/support/lightgbm.onnx", **session_options)
    x = [[5.8, 2.8]]
    sess.run(nil, {input: x})

    assert_match "onnx", File.binread(optimized_path)
  end

  def test_run_options
    run_options = {
      log_severity_level: 4,
      log_verbosity_level: 4,
      logid: "test"
    }

    sess = OnnxRuntime::InferenceSession.new("test/support/lightgbm.onnx")
    x = [[5.8, 2.8]]
    sess.run(nil, {input: x}, **run_options)
  end

  def test_profiling
    sess = OnnxRuntime::InferenceSession.new("test/support/model.onnx", enable_profiling: true)
    file = sess.end_profiling
    assert_match ".json", file
    File.unlink(file)
  end

  def test_invalid_rank
    model = OnnxRuntime::Model.new("test/support/model.onnx")
    error = assert_raises(OnnxRuntime::Error) do
      model.predict({x: []})
    end
    assert_match "Invalid rank for input: x", error.message
  end

  def test_invalid_dimensions
    model = OnnxRuntime::Model.new("test/support/model.onnx")
    error = assert_raises(OnnxRuntime::Error) do
      model.predict({x: [[[1]]]})
    end
    assert_match "Got invalid dimensions for input: x", error.message
  end

  def test_missing_input
    model = OnnxRuntime::Model.new("test/support/model.onnx")
    error = assert_raises(OnnxRuntime::Error) do
      model.predict({})
    end
    assert_match "Missing Input: x", error.message
  end

  def test_extra_input
    model = OnnxRuntime::Model.new("test/support/model.onnx")
    error = assert_raises(OnnxRuntime::Error) do
      model.predict({x: [], y: []})
    end
    assert_match "Unknown input: y", error.message
  end

  def test_invalid_output_name
    model = OnnxRuntime::Model.new("test/support/lightgbm.onnx")
    x = [[5.8, 2.8]]
    error = assert_raises(OnnxRuntime::Error) do
      model.predict({input: x}, output_names: ["bad"])
    end
    assert_match "Invalid Output Name:bad", error.message
  end

  def test_examples
    assert_example "logreg_iris.onnx", ["float_input"]
    # assert_example "mul_1.onnx", ["X"]
    assert_example "sigmoid.onnx", ["x"]

    assert_raises ArgumentError do
      OnnxRuntime::Datasets.example("bad.onnx")
    end

    # no path traversal
    assert_raises ArgumentError do
      OnnxRuntime::Datasets.example("../datasets/sigmoid.onnx")
    end
  end

  def test_modelmeta
    sess = OnnxRuntime::InferenceSession.new("test/support/model.onnx")
    metadata = sess.modelmeta
    assert_equal ({"hello" => "world", "test" => "value"}), metadata[:custom_metadata_map]
    assert_equal "", metadata[:description]
    assert_equal "", metadata[:domain]
    assert_equal "test_sigmoid", metadata[:graph_name]
    assert_equal "backend-test", metadata[:producer_name]
    assert_equal 9223372036854775807, metadata[:version]
  end

  def test_providers
    sess = OnnxRuntime::InferenceSession.new("test/support/model.onnx")
    assert_includes sess.providers, "CPUExecutionProvider"
  end

  private

  def assert_example(name, input_names)
    example = OnnxRuntime::Datasets.example(name)
    model = OnnxRuntime::Model.new(example)
    assert_equal input_names, model.inputs.map { |i| i[:name] }
  end
end
