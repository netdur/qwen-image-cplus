#!/bin/sh
# Builds the C+ compiler from a pinned commit of the cplus repository, for the
# Linux release. The checkout's vendor/ folder is the package set that
# matches it; scripts/link_vendor.sh points every package at it.
#
#   scripts/install-cpc-source.sh DESTINATION
#   CPC=DESTINATION/target/release/cpc CPLUS_VENDOR=DESTINATION/vendor ...
#
# The macOS release installs a released toolchain instead (install-cpc.sh);
# this one pins a commit because the Linux engine and GUI need compiler and
# facet_gtk fixes newer than the last release.
set -eu

ref=b30eeb0c1d22108d924a38dee321c37748db95f9
repository=https://github.com/netdur/cplus
destination=${1:?usage: install-cpc-source.sh DESTINATION}

if [ ! -d "$destination/.git" ]; then
    git clone --filter=blob:none "$repository" "$destination"
fi
git -C "$destination" fetch --quiet origin "$ref"
git -C "$destination" checkout --quiet --detach "$ref"
cargo build --release --locked -p cpc --manifest-path "$destination/Cargo.toml"

echo "built $("$destination/target/release/cpc" --version) at $ref in $destination"
