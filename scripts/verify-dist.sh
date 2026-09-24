#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dist="$project_root/dist"

test -x "$dist/bin/qwen-image-cplus"
test -f "$dist/include/qwen_image.h"
test -f "$dist/lib/libqwen_image.a"
test -f "$dist/lib/libqwen_image.dylib"

lipo "$dist/bin/qwen-image-cplus" -verify_arch arm64
lipo "$dist/lib/libqwen_image.dylib" -verify_arch arm64

if lipo "$dist/bin/qwen-image-cplus" -archs | grep -q x86_64; then
    echo "distribution unexpectedly contains an Intel CLI slice" >&2
    exit 1
fi
if lipo "$dist/lib/libqwen_image.dylib" -archs | grep -q x86_64; then
    echo "distribution unexpectedly contains an Intel library slice" >&2
    exit 1
fi

for binary in "$dist/bin/qwen-image-cplus" "$dist/lib/libqwen_image.dylib"; do
    if ! xcrun vtool -show-build "$binary" | grep -q 'minos 14\.0'; then
        echo "$binary does not target macOS 14.0" >&2
        xcrun vtool -show-build "$binary" >&2
        exit 1
    fi
done

smoke_binary=$(mktemp "${TMPDIR:-/tmp}/qwen-image-dylib-smoke.XXXXXX")
trap 'rm -f "$smoke_binary"' EXIT HUP INT TERM
clang -std=c11 -Wall -Wextra -Werror \
    "$project_root/tests/ffi_smoke.c" \
    -I "$dist/include" \
    -L "$dist/lib" -lqwen_image \
    -framework Foundation -framework Metal -lobjc \
    -o "$smoke_binary"
DYLD_LIBRARY_PATH="$dist/lib" "$smoke_binary"

usage=$($dist/bin/qwen-image-cplus 2>&1)
case "$usage" in
    'usage: qwen-image-cplus '*) ;;
    *)
        echo "installed CLI did not print its usage" >&2
        exit 1
        ;;
esac

echo "verified ARM64 macOS 14 distribution"
