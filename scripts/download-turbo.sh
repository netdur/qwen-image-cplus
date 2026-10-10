#!/bin/sh
# Download and SHA-256-check the official eight-step Turbo QIPACK and support
# files, without loading model code. Run inside nix-shell:
#   ./scripts/download-turbo.sh         # models/
#   ./scripts/download-turbo.sh /path/to/models
set -eu

revision=20c1633c1e69406aa53cef8a1fd5a332d4f1c1c9
repository=netdur/Qwen-Image-2.1-QIPACK
destination=${1:-models}
pack=qwen-image-2.1-turbo-8step-fp16-v4.qipack

uvx --from huggingface-hub==2.2.0 hf download "$repository" \
    --revision "$revision" --local-dir "$destination" \
    "$pack" manifest.json \
    processor/vocab.json processor/merges.txt \
    text_encoder/model-00001-of-00004.safetensors \
    text_encoder/model-00002-of-00004.safetensors \
    text_encoder/model-00003-of-00004.safetensors \
    text_encoder/model-00004-of-00004.safetensors \
    vae/diffusion_pytorch_model.safetensors

# Only the requested pack and its seven support files belong in this check.
# Emit shasum's data format; model metadata is never evaluated as shell code.
(
    cd "$destination"
    jq -er --arg pack "$pack" \
        '([.artifacts[] | select(.path == $pack)] + .support_files) |
         if length != 8 then error("expected one pack and seven support files")
         else .[] end |
         "\(.sha256)  \(.path)"' manifest.json | shasum -a 256 -c -
)
