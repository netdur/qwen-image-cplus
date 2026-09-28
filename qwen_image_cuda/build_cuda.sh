#!/bin/sh
# Compiles the Linux engine's CUDA kernels into the static library the
# qwen_image_cuda package links (see Cplus.toml [linux.link]).
#
#   CUDA_HOME=/usr/local/cuda-12.6 CUDA_ARCH=sm_75 qwen_image_cuda/build_cuda.sh
#
# CUDA_ARCHS builds one fat library instead, for distribution: native code for
# every listed compute capability plus PTX for the last, which newer GPUs JIT.
#
#   CUDA_ARCHS="75 80 86 89" qwen_image_cuda/build_cuda.sh
set -eu

here=$(cd "$(dirname "$0")" && pwd)
cuda_home=${CUDA_HOME:-/usr/local/cuda}
if [ -n "${CUDA_ARCHS:-}" ]; then
    arch_flags=""
    last=""
    for cc in $CUDA_ARCHS; do
        arch_flags="$arch_flags -gencode=arch=compute_$cc,code=sm_$cc"
        last=$cc
    done
    arch_flags="$arch_flags -gencode=arch=compute_$last,code=compute_$last"
    arch="sm_$(echo "$CUDA_ARCHS" | sed 's/ \{1,\}/,sm_/g') + compute_$last"
else
    arch=${CUDA_ARCH:-sm_75}
    arch_flags="-arch=$arch"
fi
source_dir="$here/cuda"
output_dir="$source_dir/target"
mkdir -p "$output_dir"

objects=""
for source in "$source_dir"/*.cu; do
    object="$output_dir/$(basename "$source" .cu).o"
    # shellcheck disable=SC2086
    "$cuda_home/bin/nvcc" -O3 $arch_flags -std=c++17 -Xcompiler -fPIC -c "$source" -o "$object"
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
