require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "tmpdir"

class Minitest::Test
  def setup
    # autoload before GC.stress
    OnnxRuntime::FFI if stress?

    GC.stress = true if stress?

    # avoid hanging from too many threads on CI
    GC.start if linux? && ci?
  end

  def teardown
    GC.stress = false if stress?
  end

  private

  def assert_elements_in_delta(expected, actual)
    assert_equal expected.size, actual.size
    expected.zip(actual) do |exp, act|
      assert_in_delta exp, act
    end
  end

  def numo?
    !["jruby", "truffleruby"].include?(RUBY_ENGINE)
  end

  def linux?
    RbConfig::CONFIG["host_os"] =~ /linux/i
  end

  def mac?
    RbConfig::CONFIG["host_os"] =~ /darwin/i
  end

  def ci?
    ENV["CI"]
  end

  def stress?
    ENV["STRESS"]
  end
end
