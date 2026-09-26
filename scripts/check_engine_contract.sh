#!/bin/sh
# Every engine package must expose the same public contract. The contract
# modules are byte-identical copies of the macOS originals.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
status=0
for module in api generation_control; do
    for engine in linux/qwen_image; do
        if ! cmp -s "$root/qwen_image/src/$module.cplus" "$root/$engine/src/$module.cplus"; then
            echo "contract drift: $engine/src/$module.cplus differs from qwen_image/src/$module.cplus" >&2
            status=1
        fi
    done
done
exit $status
