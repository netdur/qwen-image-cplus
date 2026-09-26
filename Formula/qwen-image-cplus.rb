# typed: strict
# frozen_string_literal: true

# Installs the prebuilt qwen-image-cplus runtime for Apple Silicon.
class QwenImageCplus < Formula
  desc "Native Qwen-Image-2.1 inference for Apple Silicon"
  homepage "https://github.com/netdur/qwen-image-cplus"
  url "https://github.com/netdur/qwen-image-cplus/releases/download/v0.2.1/qwen-image-cplus-aarch64-apple-darwin.tar.gz"
  version "0.2.1"
  sha256 "1bd274fb4ed87e57568f1c4c042860de503ef4eb21d6b2d12d8903172ea64cf7"
  license "MIT"

  depends_on arch: :arm64
  depends_on macos: :sonoma

  def install
    bin.install "bin/qwen-image-cplus"
    include.install "include/qwen_image.h"
    lib.install "lib/libqwen_image.a", "lib/libqwen_image.dylib"
  end

  def caveats
    <<~EOS
      Model weights are not included in this package. Follow the model setup
      instructions at:
        https://github.com/netdur/qwen-image-cplus#model-files
    EOS
  end

  test do
    assert_match "usage: qwen-image-cplus", shell_output("#{bin}/qwen-image-cplus 2>&1")
    assert_path_exists include/"qwen_image.h"
    assert_path_exists lib/"libqwen_image.a"
    assert_path_exists lib/"libqwen_image.dylib"
  end
end
