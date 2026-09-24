#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cpc_bin=${CPC:-cpc}
build_mode=${BUILD_MODE:-release}

ensure_engine_link() {
    consumer=$1
    link="$project_root/$consumer/vendor/qwen_image"
    mkdir -p "$project_root/$consumer/vendor"
    if [ ! -e "$link" ] && [ ! -L "$link" ]; then
        ln -s ../../qwen_image "$link"
    fi
}

build_package() {
    package_dir=$1
    if [ "$build_mode" = "release" ]; then
        (cd "$project_root/$package_dir" && "$cpc_bin" build --release)
    elif [ "$build_mode" = "debug" ]; then
        (cd "$project_root/$package_dir" && "$cpc_bin" build)
    else
        echo "BUILD_MODE must be release or debug" >&2
        exit 2
    fi
}

ensure_engine_link cli
ensure_engine_link ffi

build_package qwen_image
build_package cli
build_package ffi

artifact_dir="$project_root/dist"
mode_dir="$build_mode"
ffi_object="$project_root/ffi/target/$mode_dir/qwen_image.objs/qwen_image_ffi.src.ffi.o"
ffi_header="$project_root/ffi/target/$mode_dir/qwen_image.h"
cli_binary="$project_root/cli/target/$mode_dir/qwen-image-cplus"

mkdir -p "$artifact_dir/bin" "$artifact_dir/include" "$artifact_dir/lib"
cp "$cli_binary" "$artifact_dir/bin/qwen-image-cplus"
cp "$ffi_header" "$artifact_dir/include/qwen_image.h"

if [ "$build_mode" = "release" ]; then
    dependency_link_args=$(cd "$project_root/ffi" && "$cpc_bin" build --release --print-link-args)
else
    dependency_link_args=$(cd "$project_root/ffi" && "$cpc_bin" build --print-link-args)
fi

set -- "$ffi_object"
while IFS= read -r arg; do
    case "$arg" in
        *.a) set -- "$@" "$arg" ;;
    esac
done <<EOF
$dependency_link_args
EOF
xcrun libtool -static -o "$artifact_dir/lib/libqwen_image.a" "$@"

set -- "$ffi_object"
while IFS= read -r arg; do
    if [ -n "$arg" ]; then
        set -- "$@" "$arg"
    fi
done <<EOF
$dependency_link_args
EOF
clang -dynamiclib -o "$artifact_dir/lib/libqwen_image.dylib" "$@" \
    -Wl,-install_name,@rpath/libqwen_image.dylib

smoke_binary=$(mktemp "${TMPDIR:-/tmp}/qwen-image-ffi-smoke.XXXXXX")
trap 'rm -f "$smoke_binary"' EXIT HUP INT TERM
clang -std=c11 -Wall -Wextra -Werror \
    "$project_root/tests/ffi_smoke.c" \
    -I "$artifact_dir/include" \
    "$artifact_dir/lib/libqwen_image.a" \
    -framework CoreFoundation -framework CoreGraphics -framework Foundation \
    -framework ImageIO -framework Metal -lobjc \
    -o "$smoke_binary"
"$smoke_binary"

echo "distribution ready: $artifact_dir"
