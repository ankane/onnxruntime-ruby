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
  "1.15.0"
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
    download_official("libonnxruntime.so", "libonnxruntime.so.#{version}", "onnxruntime-linux-x64-#{version}.tgz", "1b3e88c0ea8e2770e5c9b11a36886097954a642c054b341992dcef268d8bb902")
    download_official("libonnxruntime.arm64.so", "libonnxruntime.so.#{version}", "onnxruntime-linux-aarch64-#{version}.tgz", "69656d8dd35633f52d79dbe5b82475a8c896c39a74eddaa8b95ae7daac0099c4")
  end

  task :mac do
    download_official("libonnxruntime.dylib", "libonnxruntime.#{version}.dylib", "onnxruntime-osx-x86_64-#{version}.tgz", "9d932b8a5b13e0f422ac6b9a23d845728e127fd4652e6872bc1e82969cac0af8")
    download_official("libonnxruntime.arm64.dylib", "libonnxruntime.#{version}.dylib", "onnxruntime-osx-arm64-#{version}.tgz", "beed38a4768cc0d0dadf879b76bd44f5c68c3a57bfaf02dea876ea5e364003ed")
  end

  task :windows do
    download_official("onnxruntime.dll", "onnxruntime.dll", "onnxruntime-win-x64-#{version}.zip", "577081480887e18d9d1baf4f38e61f416dbd54b3b585cb6267354c76ba47e741")
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
