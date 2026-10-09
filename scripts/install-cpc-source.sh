#!/bin/sh
# Builds the C+ compiler from a pinned commit of cplus main, for the
# macOS, Linux and Windows releases. The checkout's vendor/ folder is the
# package set that matches it; scripts/link_vendor.sh points every package at
# it.
#
#   scripts/install-cpc-source.sh DESTINATION
#   CPC=DESTINATION/target/release/cpc CPLUS_VENDOR=DESTINATION/vendor ...
#
# A commit rather than a release because the engines and GUIs need compiler,
# facet (button symbols), facet_gtk and facet_win32 changes newer than the
# last release, 0.0.29 (which install-cpc.sh installs).
set -eu

ref=368b4a15b6d688b251dde857bd41bf06c475923f
repository=https://github.com/netdur/cplus
destination=${1:?usage: install-cpc-source.sh DESTINATION}

if [ ! -d "$destination/.git" ]; then
    git clone --filter=blob:none "$repository" "$destination"
fi
git -C "$destination" fetch --quiet origin "$ref"
git -C "$destination" checkout --quiet --detach "$ref"
cargo build --release --locked -p cpc --manifest-path "$destination/Cargo.toml"

echo "built $("$destination/target/release/cpc" --version) at $ref in $destination"
