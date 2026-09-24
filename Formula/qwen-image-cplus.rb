# typed: strict
# frozen_string_literal: true

# Installs the prebuilt qwen-image-cplus runtime for Apple Silicon.
class QwenImageCplus < Formula
  desc "Native Qwen-Image-2.1 inference for Apple Silicon"
  homepage "https://github.com/netdur/qwen-image-cplus"
  url "https://github.com/netdur/qwen-image-cplus/releases/download/v0.0.1/qwen-image-cplus-aarch64-apple-darwin.tar.gz"
  version "0.0.1"
  sha256 "01646520545207c810849db5eeefd254a9d688ad4daeef91efbef364b3471388"
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
