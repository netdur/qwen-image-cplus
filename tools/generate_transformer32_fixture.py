#!/usr/bin/env python3
"""Generate a compact 32-block FP32 transformer oracle from pinned BF16 weights.

The four-token layout deliberately matches the existing block-0 fixture so the
native runtime can validate the transition from dense blocks to the mixed Q8
tail without requiring a prompt encoder or VAE.  This is an offline development
tool; generated binary fixtures, not NumPy, are consumed by the C+ executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_block0_fixture import (
    DIFFUSERS_COMMIT,
    HEAD_DIM,
    HEADS,
    MLP,
    MODEL_SNAPSHOT,
    ROWS,
    WIDTH,
    SafeTensorFile,
    apply_rope,
    attention,
    layer_norm,
    rms_norm,
    rope_table,
)


CAPTURE_BLOCKS = (0, 23, 24, 27, 28, 31)


class TransformerWeights:
    def __init__(self, model: Path) -> None:
        root = model / "transformer"
        self.shards = (
            SafeTensorFile(root / "diffusion_pytorch_model-00001-of-00002.safetensors"),
            SafeTensorFile(root / "diffusion_pytorch_model-00002-of-00002.safetensors"),
        )

    def bf16(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        for shard in self.shards:
            if name in shard.header:
                return shard.bf16(name, shape)
        raise KeyError(name)

    def linear(self, name: str, inputs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        weights = self.bf16(name, shape)
        result = inputs @ weights.T
        del weights
        return np.asarray(result, dtype=np.float32)


def write_tensor(directory: Path, name: str, value: np.ndarray, records: dict) -> None:
    value = np.asarray(value, dtype="<f4", order="C")
    raw = value.tobytes()
    filename = f"{name}.f32"
    (directory / filename).write_bytes(raw)
    records[name] = {
        "file": filename,
        "dtype": "F32",
        "shape": list(value.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def generate(model: Path, output: Path) -> None:
    weights = TransformerWeights(model)
    index = np.arange(ROWS * WIDTH, dtype=np.float32).reshape(ROWS, WIDTH)
    hidden = (np.sin(index * np.float32(0.0013)) * np.float32(0.25)).astype(np.float32)
    modulation_index = np.arange(2 * 4 * WIDTH, dtype=np.float32).reshape(2, 4, WIDTH)
    modulation = (
        np.sin(modulation_index * np.float32(0.0007)) * np.float32(0.08)
    ).astype(np.float32)
    target_mask = np.array([False, False, True, True])
    selected = np.where(target_mask[:, None, None], modulation[0:1], modulation[1:2])
    rope = rope_table()
    captures: dict[int, np.ndarray] = {}

    for block in range(32):
        prefix = f"transformer_blocks.{block}."
        norm1 = layer_norm(hidden) * (np.float32(1.0) + selected[:, 0])
        query = weights.linear(prefix + "attn.to_q.weight", norm1, (WIDTH, WIDTH))
        key = weights.linear(prefix + "attn.to_k.weight", norm1, (WIDTH, WIDTH))
        value = weights.linear(prefix + "attn.to_v.weight", norm1, (WIDTH, WIDTH))
        query = query.reshape(ROWS, HEADS, HEAD_DIM)
        key = key.reshape(ROWS, HEADS, HEAD_DIM)
        value = value.reshape(ROWS, HEADS, HEAD_DIM)
        query = apply_rope(
            rms_norm(query, weights.bf16(prefix + "attn.norm_q.weight", (HEAD_DIM,))),
            rope,
        )
        key = apply_rope(
            rms_norm(key, weights.bf16(prefix + "attn.norm_k.weight", (HEAD_DIM,))),
            rope,
        )
        attended = attention(query, key, value).reshape(ROWS, WIDTH)
        attention_projected = weights.linear(
            prefix + "attn.to_out.0.weight", attended, (WIDTH, WIDTH)
        )
        residual = hidden + np.tanh(selected[:, 1], dtype=np.float32) * attention_projected
        norm2 = layer_norm(residual) * (np.float32(1.0) + selected[:, 2])
        gate = weights.linear(prefix + "img_mlp.gate_layer.weight", norm2, (MLP, WIDTH))
        projected = weights.linear(prefix + "img_mlp.proj.weight", norm2, (MLP, WIDTH))
        swiglu = gate / (np.float32(1.0) + np.exp(-gate, dtype=np.float32)) * projected
        mlp = weights.linear(prefix + "img_mlp.out.weight", swiglu, (WIDTH, MLP))
        hidden = residual + np.tanh(selected[:, 3], dtype=np.float32) * mlp
        if block in CAPTURE_BLOCKS:
            captures[block] = hidden.copy()
        print(f"reference transformer block {block + 1}/32", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    write_tensor(output, "hidden", (np.sin(index * np.float32(0.0013)) * np.float32(0.25)), records)
    write_tensor(output, "modulation", modulation, records)
    write_tensor(output, "rope", rope, records)
    for block in CAPTURE_BLOCKS:
        write_tensor(output, f"block_{block:02d}_output", captures[block], records)

    metadata = {
        "schema_version": 1,
        "operation": "QwenImage21TransformerBlock chain, blocks 0-31, FP32 reference",
        "model_snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "reference": "Pinned Diffusers equations transcribed to NumPy; BF16 checkpoint weights decoded to FP32",
        "shape": {
            "rows": ROWS,
            "width": WIDTH,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "mlp": MLP,
            "blocks": 32,
        },
        "tokens": {
            "image_ids": [-1, -1, 0, 0],
            "target_mask": [False, False, True, True],
        },
        "capture_blocks": list(CAPTURE_BLOCKS),
        "acceptance": {
            "metric": "normalized RMS versus the dense FP32 oracle",
            "final_limit": 0.01,
        },
        "tensors": records,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/transformer32_fp32")
    )
    arguments = parser.parse_args()
    generate(arguments.model.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
