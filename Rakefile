require "bundler/gem_tasks"
require "rake/testtask"
require "ruby_memcheck"

test_config = lambda do |t|
  t.pattern = "test/**/*_test.rb"
end
Rake::TestTask.new(&test_config)

namespace :test do
  RubyMemcheck::TestTask.new(:valgrind, &test_config)
end

task default: :test

shared_libraries = %w(libonnxruntime.so libonnxruntime.arm64.so libonnxruntime.arm64.dylib onnxruntime.dll)

# ensure vendor files exist
task :ensure_vendor do
  shared_libraries.each do |file|
    raise "Missing file: #{file}" unless File.exist?("vendor/#{file}")
  end
end

Rake::Task["build"].enhance [:ensure_vendor]

platforms = ["x86_64-linux", "aarch64-linux", "x86_64-darwin", "arm64-darwin", "x64-mingw"]

task :build_platform do
  require "fileutils"

  platforms.each do |platform|
    sh "gem", "build", "--platform", platform
  end

  FileUtils.mkdir_p("pkg")
  Dir["*.gem"].each do |file|
    FileUtils.move(file, "pkg")
  end
end

task :release_platform do
  require_relative "lib/onnxruntime/version"

  Dir["pkg/onnxruntime-#{OnnxRuntime::VERSION}-*.gem"].each do |file|
    sh "gem", "push", file
  end
end

def version
  "1.29.0"
end

def download_official(library, remote_lib, file, sha256)
  require "fileutils"
  require "open-uri"
  require "tmpdir"

  url = "https://github.com/microsoft/onnxruntime/releases/download/v#{version}/#{file}"
  puts "Downloading #{file}..."
  contents = URI.parse(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  Dir.chdir(Dir.mktmpdir) do
    File.binwrite(file, contents)
    command = file.end_with?(".zip") ? "unzip -q" : "tar xf"
    system "#{command} #{file}"
    src = file[0..-5]
    dest = File.expand_path("vendor", __dir__)

    FileUtils.cp("#{src}/lib/#{remote_lib}", "#{dest}/#{library}")
    puts "Saved vendor/#{library}"

    if library.end_with?(".so")
      FileUtils.cp("#{src}/LICENSE", "#{dest}/LICENSE")
      puts "Saved vendor/LICENSE"
      FileUtils.cp("#{src}/ThirdPartyNotices.txt", "#{dest}/ThirdPartyNotices.txt")
      puts "Saved vendor/ThirdPartyNotices.txt"
    end
  end
end

# https://github.com/microsoft/onnxruntime/releases
namespace :vendor do
  task :linux do
    download_official("libonnxruntime.so", "libonnxruntime.so.#{version}", "onnxruntime-linux-x64-#{version}.tgz", "c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba")
    download_official("libonnxruntime.arm64.so", "libonnxruntime.so.#{version}", "onnxruntime-linux-aarch64-#{version}.tgz", "e1799098ebc054b370f6176a450f158720f297818c613e5dc99b92e2ec82346f")
  end

  task :mac do
    download_official("libonnxruntime.arm64.dylib", "libonnxruntime.#{version}.dylib", "onnxruntime-osx-arm64-#{version}.tgz", "d0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92")
  end

  task :windows do
    download_official("onnxruntime.dll", "onnxruntime.dll", "onnxruntime-win-x64-#{version}.zip", "c9b4b7086b529ad814f428c1bad028e20a25d7dc0699836775faace4ab5b78b2")
  end

  task all: [:linux, :mac, :windows]

  task :platform do
    if Gem.win_platform?
      Rake::Task["vendor:windows"].invoke
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      Rake::Task["vendor:mac"].invoke
    else
      Rake::Task["vendor:linux"].invoke
    end
  end
end
