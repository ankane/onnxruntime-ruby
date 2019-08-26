module OnnxRuntime
  module Utils
    def self.reshape(arr, dims)
      dims[1..-1].reverse.each do |dim|
        arr = arr.each_slice(dim)
      end
      arr.to_a
    end
  end
end
