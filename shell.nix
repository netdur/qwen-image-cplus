# Development tools for the native Apple Silicon build.
# Enter with `nix-shell`, then follow docs/build-and-verify.md.
{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShellNoCC {
  packages = with pkgs; [
    cargo
    rustc
    git
    curl
    uv
    jq
    nixfmt
    shellcheck
  ];

  # C+ emits Apple object files and links system frameworks with Xcode tools.
  # Keep Apple's clang ahead of Nix's compiler wrappers.
  shellHook = ''
    export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
    export MACOSX_DEPLOYMENT_TARGET=14.0
    # macOS 27 rejects some stripped Rust proc-macro dylibs (rust#157750).
    export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_DEBUG=true
    export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
    export CPC="$PWD/../cplus/target/release/cpc"
    export CPLUS_VENDOR="$PWD/../cplus/vendor"
    export PATH="$(dirname "$CPC"):$PATH"
  '';
}
