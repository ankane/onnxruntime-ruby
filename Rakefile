require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
end

shared_libraries = %w(libonnxruntime.so libonnxruntime.arm64.so libonnxruntime.dylib libonnxruntime.arm64.dylib onnxruntime.dll)

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
  "1.15.1"
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
    download_official("libonnxruntime.so", "libonnxruntime.so.#{version}", "onnxruntime-linux-x64-#{version}.tgz", "5492f9065f87538a286fb04c8542e9ff7950abb2ea6f8c24993a940006787d87")
    download_official("libonnxruntime.arm64.so", "libonnxruntime.so.#{version}", "onnxruntime-linux-aarch64-#{version}.tgz", "85272e75d8dd841138de4b774a9672ea93c1be108d96038c6c34a62d7f976aee")
  end

  task :mac do
    download_official("libonnxruntime.dylib", "libonnxruntime.#{version}.dylib", "onnxruntime-osx-x86_64-#{version}.tgz", "4b66ebbca24b8b96f6b74655fee3610a7e529b4e01f6790632f24ee82b778e5a")
    download_official("libonnxruntime.arm64.dylib", "libonnxruntime.#{version}.dylib", "onnxruntime-osx-arm64-#{version}.tgz", "df97832fc7907c6677a6da437f92339d84a462becb74b1d65217fcb859ee9460")
  end

  task :windows do
    download_official("onnxruntime.dll", "onnxruntime.dll", "onnxruntime-win-x64-#{version}.zip", "261308ee5526dfd3f405ce8863e43d624a2e0bcd16b2d33cdea8c120ab3534d3")
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
