#!/bin/sh
# Builds the C+ compiler from a pinned commit of the cplus repository, for the
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

# The older checkout passed the Windows distribution build. The newer one is
# needed for the macOS agent libraries, but fails the Windows build in CI.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) ref=0c48e8011249defe0619600ce4da22a1f0f1ff48 ;;
    *) ref=b30eeb0c1d22108d924a38dee321c37748db95f9 ;;
esac
repository=https://github.com/netdur/cplus
destination=${1:?usage: install-cpc-source.sh DESTINATION}

if [ ! -d "$destination/.git" ]; then
    git clone --filter=blob:none "$repository" "$destination"
fi
git -C "$destination" fetch --quiet origin "$ref"
git -C "$destination" checkout --quiet --detach "$ref"
cargo build --release --locked -p cpc --manifest-path "$destination/Cargo.toml"

echo "built $("$destination/target/release/cpc" --version) at $ref in $destination"
