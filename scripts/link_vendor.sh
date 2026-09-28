#!/bin/sh
# Point every package's vendor/ at the C+ vendor folder, so third-party
# packages (stdlib, json, objc, metal, facet, ...) are shared rather than
# installed per package. The project's own packages (qwen_image,
# qwen_image_metal, qwen_image_cuda, qwen_image_runtime) need no link: they sit
# side by side at the repository root and cpc resolves them as siblings.
#
#   CPLUS_VENDOR=/path/to/cplus/vendor scripts/link_vendor.sh
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cplus_vendor=${CPLUS_VENDOR:-$root/../cplus/vendor}
if [ ! -d "$cplus_vendor" ]; then
    echo "C+ vendor folder not found: $cplus_vendor (set CPLUS_VENDOR)" >&2
    exit 1
fi

# Git Bash's `ln -s` copies the whole tree instead of linking it, so on Windows
# the link is a directory junction, which needs no administrator.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) windows=true ;;
    *) windows=false ;;
esac

for package in qwen_image qwen_image_metal qwen_image_cuda qwen_image_runtime \
               qwen_image_metal_dev qwen_image_cuda_dev qwen_image_quantize cli ffi gui; do
    vendor="$root/$package/vendor"
    if [ "$windows" = true ]; then
        if [ -e "$vendor" ]; then
            echo "leaving $package/vendor (already present; remove it to relink)" >&2
        else
            cmd //c mklink //J "$(cygpath -w "$vendor")" "$(cygpath -w "$cplus_vendor")" >/dev/null
        fi
    elif [ -L "$vendor" ] || [ ! -e "$vendor" ]; then
        ln -sfn "$cplus_vendor" "$vendor"
    else
        echo "leaving $package/vendor (a real directory; remove it to link)" >&2
    fi
done
echo "vendor folders linked to $cplus_vendor"
