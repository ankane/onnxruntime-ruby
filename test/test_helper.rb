require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "tmpdir"

class Minitest::Test
  def setup
    GC.stress = true if stress?
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
    RUBY_PLATFORM != "java"
  end

  def mac?
    RbConfig::CONFIG["host_os"] =~ /darwin/i
  end

  def stress?
    ENV["STRESS"]
  end
end
