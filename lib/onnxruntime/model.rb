module OnnxRuntime
  class Model
    def initialize(path_or_bytes, **session_options)
      @session = InferenceSession.new(path_or_bytes, **session_options)
    end

    def predict(input_feed, output_names: nil, **run_options)
      predictions = @session.run(output_names, input_feed, **run_options)
      output_names ||= outputs.map { |o| o[:name] }
      output_names.zip(predictions).to_h { |k, v| [k.to_s, v] }
    end

    def inputs
      @session.inputs
    end

    def outputs
      @session.outputs
    end

    def metadata
      @session.modelmeta
    end
  end
end
