#!/bin/sh
# Link each client package's vendor/ to the C+ vendor folder and to the engine
# for this platform. Clients always import `qwen_image/...`; which directory
# answers to that name is decided here, not in their sources.
#
#   CPLUS_VENDOR=/path/to/cplus/vendor scripts/link_vendor.sh
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cplus_vendor=${CPLUS_VENDOR:-$root/../cplus/vendor}
if [ ! -d "$cplus_vendor" ]; then
    echo "C+ vendor folder not found: $cplus_vendor (set CPLUS_VENDOR)" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin) engine=qwen_image ;;
    Linux) engine=linux/qwen_image ;;
    *) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

# Engines resolve their own dependencies straight from the C+ vendor folder.
for package in qwen_image linux/qwen_image; do
    ln -sfn "$cplus_vendor" "$root/$package/vendor"
done

for client in cli ffi gui; do
    vendor="$root/$client/vendor"
    if [ -L "$vendor" ]; then rm "$vendor"; fi
    mkdir -p "$vendor"
    for package in "$cplus_vendor"/*; do
        ln -sfn "$package" "$vendor/$(basename "$package")"
    done
    ln -sfn "$root/$engine" "$vendor/qwen_image"
done
echo "clients linked to $engine"
