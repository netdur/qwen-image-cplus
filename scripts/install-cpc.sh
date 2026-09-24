#!/bin/sh
set -eu

version=0.0.29
sha256=fe2f358e4cf880b4dc58f88620c38e716bc7be1f14b981e5b158846d957729e6
asset=cplus-aarch64-apple-darwin.tar.gz
url="https://github.com/netdur/cplus/releases/download/v${version}/${asset}"
destination=${1:?usage: install-cpc.sh DESTINATION}

mkdir -p "$destination"
archive="$destination/$asset"

curl --fail --location --retry 3 --output "$archive" "$url"
printf '%s  %s\n' "$sha256" "$archive" | shasum -a 256 -c -
tar -xzf "$archive" -C "$destination"
rm "$archive"

actual_version=$("$destination/cpc" --version)
if [ "$actual_version" != "cpc $version" ]; then
    echo "expected cpc $version, got $actual_version" >&2
    exit 1
fi

echo "installed $actual_version in $destination"
