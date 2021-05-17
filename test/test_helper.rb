require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "tmpdir"

class Minitest::Test
  private

  def setup
    GC.stress = true if stress?
  end

  def teardown
    GC.stress = false
  end

  def assert_elements_in_delta(expected, actual)
    assert_equal expected.size, actual.size
    expected.zip(actual) do |exp, act|
      assert_in_delta exp, act
    end
  end

  def stress?
    ENV["STRESS"]
  end
end
