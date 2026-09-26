# typed: strict
# frozen_string_literal: true

# Installs the prebuilt qwen-image-cplus CLI, library, and app for Apple Silicon.
class QwenImageCplus < Formula
  desc "Native Qwen-Image-2.1 inference for Apple Silicon"
  homepage "https://github.com/netdur/qwen-image-cplus"
  url "https://github.com/netdur/qwen-image-cplus/releases/download/v0.2.2/qwen-image-cplus-aarch64-apple-darwin.tar.gz"
  version "0.2.2"
  sha256 "62ef677a2e5ed64b36c68e6725e4fbf6fe1f1c1588be97126886bbe39b2ba781"
  license "MIT"

  depends_on arch: :arm64
  depends_on macos: :sonoma

  def install
    bin.install "bin/qwen-image-cplus"
    include.install "include/qwen_image.h"
    lib.install "lib/libqwen_image.a", "lib/libqwen_image.dylib"
    prefix.install "Qwen Image.app"
    bin.install_symlink prefix/"Qwen Image.app/Contents/MacOS/gui" => "qwen-image-gui"
  end

  def caveats
    <<~EOS
      Model weights are not included in this package. Follow the model setup
      instructions at:
        https://github.com/netdur/qwen-image-cplus#model-files

      Launch the app with `qwen-image-gui` or:
        open "#{prefix}/Qwen Image.app"
    EOS
  end

  test do
    assert_match "usage: qwen-image-cplus", shell_output("#{bin}/qwen-image-cplus 2>&1")
    assert_path_exists include/"qwen_image.h"
    assert_path_exists lib/"libqwen_image.a"
    assert_path_exists lib/"libqwen_image.dylib"
    assert_path_exists prefix/"Qwen Image.app/Contents/MacOS/gui"
    assert_path_exists bin/"qwen-image-gui"
  end
end
