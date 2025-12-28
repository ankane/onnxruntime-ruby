source "https://rubygems.org"

gemspec

gem "rake"
gem "minitest"

# TODO remove when numo-narray > 0.9.2.1 is released
if Gem.win_platform?
  gem "numo-narray", github: "ruby-numo/numo-narray", ref: "421feddb46cac5145d69067fc1ac3ba3c434f668"
else
  gem "numo-narray", platform: [:mri, :windows]
end
