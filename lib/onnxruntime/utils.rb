module OnnxRuntime
  module Utils
    class << self
      attr_accessor :mutex
    end
    self.mutex = Mutex.new

    def self.reshape(arr, dims)
      arr = arr.flatten
      dims[1..-1].reverse.each do |dim|
        arr = arr.each_slice(dim)
      end
      arr.to_a
    end
  end
end
