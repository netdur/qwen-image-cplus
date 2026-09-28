#!/bin/sh
# Builds the Linux distribution into dist/: the CLI, the GUI, the C library,
# and the CUDA runtime libraries they load, laid out as an install prefix.
#
#   dist/bin/qwen-image-cplus                    CLI
#   dist/bin/qwen-image-gui                      desktop app (GTK 4)
#   dist/include/qwen_image.h                    C API
#   dist/lib/libqwen_image.so                    C API, shared
#   dist/lib/libqwen_image.a                     C API, static (link the CUDA libs yourself)
#   dist/lib/qwen-image-cplus/                   bundled CUDA 12 / cuDNN 8 runtime, libjpeg
#   dist/share/facet/assets/                     the GUI's icon font
#
# The binaries find the bundled CUDA libraries through an $ORIGIN-relative
# RUNPATH, so the prefix can be moved as a whole (Homebrew keg, snap). The GPU
# driver (libcuda.so.1) and GTK 4 come from the system.
#
# Environment:
#   CPC          C+ compiler (default: cpc)
#   BUILD_MODE   release | debug (default: release)
#   CUDA_HOME    CUDA toolkit (default: /usr/local/cuda)
#   CUDA_ARCHS   GPU architectures to compile for (default: "75 80 86 89")
#   CUDNN_LIB    directory holding libcudnn*.so.8 (default: found through ldd)
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cpc_bin=${CPC:-cpc}
build_mode=${BUILD_MODE:-release}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_ARCHS=${CUDA_ARCHS:-75 80 86 89}

case "$build_mode" in
    release) cpc_flags=--release ;;
    debug) cpc_flags= ;;
    *) echo "BUILD_MODE must be release or debug" >&2; exit 2 ;;
esac

build_package() {
    # shellcheck disable=SC2086
    (cd "$project_root/$1" && "$cpc_bin" build $cpc_flags)
}

"$project_root/qwen_image_cuda/build_cuda.sh"

# The qwen_image* packages sit side by side at the repository root and
# resolve as sibling directories; vendor/ carries only third-party packages.
build_package qwen_image
build_package cli
build_package ffi
build_package gui

dist="$project_root/dist"
private_lib="$dist/lib/qwen-image-cplus"
rm -rf "$dist"
mkdir -p "$dist/bin" "$dist/include" "$private_lib" "$dist/share/facet/assets"

cp "$project_root/cli/target/$build_mode/qwen-image-cplus" "$dist/bin/qwen-image-cplus"
cp "$project_root/gui/target/$build_mode/gui" "$dist/bin/qwen-image-gui"
cp "$project_root/ffi/target/$build_mode/qwen_image.h" "$dist/include/qwen_image.h"
cp "$project_root/ffi/target/$build_mode/libqwen_image.so" "$dist/lib/libqwen_image.so"
cp "$project_root/gui/vendor/facet/assets/MaterialSymbolsOutlined.ttf" "$dist/share/facet/assets/"

# The static library: the FFI objects plus every archive the link line names.
# A consumer adds -lcudart -lcublas -l:libcudnn.so.8 -ljpeg -lstdc++.
# shellcheck disable=SC2086
link_args=$(cd "$project_root/ffi" && "$cpc_bin" build $cpc_flags --print-link-args)
mri=$(mktemp)
trap 'rm -f "$mri"' EXIT HUP INT TERM
{
    echo "create $dist/lib/libqwen_image.a"
    echo "addlib $project_root/ffi/target/$build_mode/libqwen_image.a"
    printf '%s\n' "$link_args" | while IFS= read -r arg; do
        case "$arg" in
            *.a) echo "addlib $arg" ;;
        esac
    done
    echo "save"
    echo "end"
} > "$mri"
ar -M < "$mri"

# The CUDA runtime (and libjpeg, which not every desktop install carries),
# found where the build resolved it. cuDNN 8 is a small
# dispatcher that dlopens its sub-libraries by soname from the same search
# path; the engine's convolutions need only the inference pair.
resolve() {
    ldd "$dist/bin/qwen-image-cplus" | awk -v lib="$1" '$1 == lib { print $3 }'
}
for soname in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcudnn.so.8 libjpeg.so.8; do
    path=$(resolve "$soname")
    if [ -z "$path" ] || [ ! -f "$path" ]; then
        echo "cannot find $soname (set CUDA_HOME / CUDNN_LIB or LD_LIBRARY_PATH)" >&2
        exit 1
    fi
    cp -L "$path" "$private_lib/$soname"
done
cudnn_dir=${CUDNN_LIB:-$(dirname "$(resolve libcudnn.so.8)")}
for soname in libcudnn_ops_infer.so.8 libcudnn_cnn_infer.so.8; do
    cp -L "$cudnn_dir/$soname" "$private_lib/$soname"
done

# cpc links with the build machine's CUDA path as RUNPATH; the distribution
# looks beside itself instead.
patchelf --set-rpath '$ORIGIN/../lib/qwen-image-cplus' "$dist/bin/qwen-image-cplus"
patchelf --set-rpath '$ORIGIN/../lib/qwen-image-cplus' "$dist/bin/qwen-image-gui"
patchelf --set-rpath '$ORIGIN/qwen-image-cplus' "$dist/lib/libqwen_image.so"
for lib in "$private_lib"/*.so.*; do
    patchelf --set-rpath '$ORIGIN' "$lib"
done

# Every binary resolves every library from the bundle or the system.
for binary in "$dist/bin/qwen-image-cplus" "$dist/bin/qwen-image-gui" "$dist/lib/libqwen_image.so"; do
    if env -u LD_LIBRARY_PATH ldd "$binary" | grep -q "not found"; then
        env -u LD_LIBRARY_PATH ldd "$binary" | grep "not found" >&2
        exit 1
    fi
    for soname in libcudart.so.12 libcublas.so.12 libcudnn.so.8; do
        if ! env -u LD_LIBRARY_PATH ldd "$binary" | grep "$soname =>" | grep -q "/qwen-image-cplus/$soname "; then
            echo "$binary does not load $soname from the bundle" >&2
            exit 1
        fi
    done
done

# The C API links and answers, shared and static. No GPU needed: the smoke
# test only exercises validation.
smoke=$(mktemp -d)
trap 'rm -f "$mri"; rm -rf "$smoke"' EXIT HUP INT TERM
cc -std=c11 -Wall -Wextra -Werror "$project_root/tests/ffi_smoke.c" \
    -I "$dist/include" -L "$dist/lib" -lqwen_image -Wl,-rpath,"$dist/lib" \
    -o "$smoke/shared"
"$smoke/shared"
cc -std=c11 -Wall -Wextra -Werror "$project_root/tests/ffi_smoke.c" \
    -I "$dist/include" "$dist/lib/libqwen_image.a" \
    -L "$private_lib" -Wl,-rpath,"$private_lib" \
    -l:libcudart.so.12 -l:libcublas.so.12 -l:libcudnn.so.8 -ljpeg -lstdc++ -lm -lpthread -ldl \
    -o "$smoke/static"
"$smoke/static"

"$dist/bin/qwen-image-cplus" 2>&1 | grep -q "usage: qwen-image-cplus"

echo "distribution ready: $dist"
