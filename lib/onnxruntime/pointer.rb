module OnnxRuntime
  class Pointer
    attr_reader :ref

    def initialize(free)
      @ref = ::FFI::MemoryPointer.new(:pointer)
      ObjectSpace.define_finalizer(self, self.class.finalize(@ref, free))
    end

    def self.finalize(ref, free)
      proc do
        ptr = ref.read_pointer
        free.(ptr) unless ptr.null?
      end
    end

    def to_ptr
      @ref.read_pointer
    end

    def read_string(...)
      to_ptr.read_string(...)
    end

    def read_array_of_pointer(...)
      to_ptr.read_array_of_pointer(...)
    end
  end
end
