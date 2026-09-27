#!/bin/sh
# Compiles the Linux engine's CUDA kernels into the static library the
# qwen_image_cuda package links (see Cplus.toml [linux.link]).
#
#   CUDA_HOME=/usr/local/cuda-12.6 CUDA_ARCH=sm_75 qwen_image_cuda/build_cuda.sh
set -eu

here=$(cd "$(dirname "$0")" && pwd)
cuda_home=${CUDA_HOME:-/usr/local/cuda}
arch=${CUDA_ARCH:-sm_75}
source_dir="$here/cuda"
output_dir="$source_dir/target"
mkdir -p "$output_dir"

objects=""
for source in "$source_dir"/*.cu; do
    object="$output_dir/$(basename "$source" .cu).o"
    "$cuda_home/bin/nvcc" -O3 -arch="$arch" -std=c++17 -Xcompiler -fPIC -c "$source" -o "$object"
    objects="$objects $object"
done
# Host-side C helpers (image decoding and resampling).
for source in "$here"/native/*.c; do
    object="$output_dir/$(basename "$source" .c).o"
    cc -O2 -fPIC -c "$source" -o "$object"
    objects="$objects $object"
done
rm -f "$output_dir/libqwen_image_cuda.a"
# shellcheck disable=SC2086
ar rcs "$output_dir/libqwen_image_cuda.a" $objects
echo "built $output_dir/libqwen_image_cuda.a ($arch)"
