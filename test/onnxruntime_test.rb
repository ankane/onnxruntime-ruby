require_relative "test_helper"

class OnnxRuntimeTest < Minitest::Test
  def test_lib_version
    assert_equal "0.5.0", OnnxRuntime.lib_version
  end

  def test_basic
    model = OnnxRuntime::Model.new("test/support/model.onnx")

    expected = [{:name=>"x", :type=>"tensor(float)", :shape=>[3, 4, 5]}]
    assert_equal expected, model.inputs

    expected = [{:name=>"y", :type=>"tensor(float)", :shape=>[3, 4, 5]}]
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

    output = model.predict(x: x)
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

    output = model.predict(x: x)
    assert_elements_in_delta [0.6338603, 0.6715468, 0.6462883, 0.6329476, 0.6043575], output["y"].first.first
  end

  def test_string
    contents = File.binread("test/support/model.onnx")
    model = OnnxRuntime::Model.new(contents)
    expected = [{:name=>"x", :type=>"tensor(float)", :shape=>[3, 4, 5]}]
    assert_equal expected, model.inputs
  end

  def test_lightgbm
    model = OnnxRuntime::Model.new("test/support/lightgbm.onnx")

    expected = [{:name=>"input", :type=>"tensor(float)", :shape=>[1, 2]}]
    assert_equal expected, model.inputs

    expected = [{:name=>"label", :type=>"tensor(int64)", :shape=>[1]}, {:name=>"probabilities", :type=>"seq", :shape=>[]}]
    assert_equal expected, model.outputs

    x = [[5.8, 2.8],
         [6.0, 2.2],
         [5.5, 4.2],
         [7.3, 2.9],
         [5.0, 3.4]]

    output = model.predict({input: x}) #, output_names: ["label"])
    assert_equal [1, 1, 0, 2, 0], output["label"]
    probabilities = output["probabilities"].first
    assert_equal [0, 1, 2], probabilities.keys
    assert_elements_in_delta [0.2593829035758972, 0.409047931432724, 0.3315691649913788], probabilities.values
  end

  def test_random_forest
    model = OnnxRuntime::Model.new("test/support/randomforest.onnx")

    expected = [{:name=>"float_input", :type=>"tensor(float)", :shape=>[1, 4]}]
    assert_equal expected, model.inputs

    expected = [{:name=>"output_label", :type=>"tensor(int64)", :shape=>[1]}, {:name=>"output_probability", :type=>"seq", :shape=>[]}]
    assert_equal expected, model.outputs

    x = [[5.8, 2.8, 5.1, 2.4]]

    output = model.predict(float_input: x)
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
end
