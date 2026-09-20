#!/usr/bin/env python3
"""Generate a small complete Qwen-Image-2.1 transformer oracle.

The layout contains three text positions, one 2x2 condition image, and one
2x4 target image.  One text key is invalid.  It therefore exercises the real
four-fold VLM image-slot expansion, image substitution, block-causal mask,
causal-condition timestep split, all 32 blocks, and the final adaptive norm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from generate_block0_fixture import (
    DIFFUSERS_COMMIT,
    EPSILON,
    HEAD_DIM,
    HEADS,
    MLP,
    MODEL_SNAPSHOT,
    WIDTH,
    layer_norm,
    rms_norm,
)
from generate_transformer32_fixture import TransformerWeights


TEXT_ROWS = 4
IMAGE_ROWS = 12
ROWS = 15
OUT_CHANNELS = 64
CAPTURE_BLOCKS = (0, 23, 24, 27, 28, 31)
IMG_MASK = np.array([False, False, True, False, True, True])
IMAGE_PAD_MASK = np.repeat(IMG_MASK, np.where(IMG_MASK, 4, 1))
IMAGE_IDS = np.array([-1, -1, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
TARGET_MASK = IMAGE_IDS == 1
KEY_VALID = np.array([True, True, True, True, True, True, False, True, True, True, True, True, True, True, True])


def silu(value: np.ndarray) -> np.ndarray:
    return value / (np.float32(1.0) + np.exp(-value, dtype=np.float32))


def gelu_tanh(value: np.ndarray) -> np.ndarray:
    coefficient = np.float32(math.sqrt(2.0 / math.pi))
    cubic = np.float32(0.044715) * value * value * value
    return np.float32(0.5) * value * (
        np.float32(1.0) + np.tanh(coefficient * (value + cubic), dtype=np.float32)
    )


def rope_table() -> np.ndarray:
    # Exact QwenImage21Rope cursor/position construction for image shapes
    # [(1,2,2), (1,2,4)] and IMAGE_PAD_MASK above.
    frame = np.array([0, 1, 2, 2, 2, 2, 4, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int32)
    height = frame.copy()
    width = frame.copy()
    height[IMAGE_PAD_MASK] = np.array([-1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0])
    width[IMAGE_PAD_MASK] = np.array([-1, 0, -1, 0, -2, -1, 0, 1, -2, -1, 0, 1])
    table_parts = []
    for positions, dimension in zip((frame, height, width), (16, 56, 56)):
        frequency = np.power(
            np.float32(10000.0),
            -np.arange(0, dimension, 2, dtype=np.float32) / np.float32(dimension),
            dtype=np.float32,
        )
        table_parts.append(positions[:, None].astype(np.float32) * frequency[None])
    angles = np.concatenate(table_parts, axis=1)
    return np.concatenate(
        [np.cos(angles, dtype=np.float32), np.sin(angles, dtype=np.float32)], axis=1
    )


def apply_rope(value: np.ndarray, rope: np.ndarray) -> np.ndarray:
    paired = value.reshape(ROWS, HEADS, HEAD_DIM // 2, 2)
    cosine = rope[:, None, : HEAD_DIM // 2]
    sine = rope[:, None, HEAD_DIM // 2 :]
    output = np.empty_like(paired)
    output[..., 0] = paired[..., 0] * cosine - paired[..., 1] * sine
    output[..., 1] = paired[..., 0] * sine + paired[..., 1] * cosine
    return output.reshape(ROWS, HEADS, HEAD_DIM)


def attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = np.empty_like(query)
    scale = np.float32(1.0 / math.sqrt(HEAD_DIM))
    for row in range(ROWS):
        allowed = KEY_VALID & np.array(
            [row >= column or (IMAGE_IDS[row] >= 0 and IMAGE_IDS[row] == IMAGE_IDS[column]) for column in range(ROWS)]
        )
        scores = np.einsum("hd,khd->hk", query[row], key, dtype=np.float32) * scale
        scores[:, ~allowed] = -np.inf
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores, dtype=np.float32)
        probabilities /= probabilities.sum(axis=-1, keepdims=True, dtype=np.float32)
        result[row] = np.einsum("hk,khd->hd", probabilities, value, dtype=np.float32)
    return result


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
    image_index = np.arange(IMAGE_ROWS * OUT_CHANNELS, dtype=np.float32).reshape(IMAGE_ROWS, OUT_CHANNELS)
    text_index = np.arange(TEXT_ROWS * WIDTH, dtype=np.float32).reshape(TEXT_ROWS, WIDTH)
    image_input = (np.sin(image_index * np.float32(0.017)) * np.float32(0.35)).astype(np.float32)
    text_input = (np.cos(text_index * np.float32(0.0011)) * np.float32(0.2)).astype(np.float32)

    image_projected = weights.linear("img_in.weight", image_input, (WIDTH, OUT_CHANNELS))
    text_weight = weights.bf16("txt_in.text_norm.weight", (WIDTH,)) + np.float32(1.0)
    text_normalized = rms_norm(text_input, text_weight)
    text_hidden = weights.linear("txt_in.in_layer.weight", text_normalized, (WIDTH, WIDTH))
    text_activated = gelu_tanh(text_hidden)
    text_projected = weights.linear("txt_in.out_layer.weight", text_activated, (WIDTH, WIDTH))

    timestep = np.array([np.float32(0.623013), np.float32(0.0)], dtype=np.float32)
    frequency = np.exp(
        -np.float32(math.log(10000.0)) * np.arange(128, dtype=np.float32) / np.float32(128.0),
        dtype=np.float32,
    )
    arguments = timestep[:, None] * np.float32(1000.0) * frequency[None]
    time_projection = np.concatenate(
        [np.cos(arguments, dtype=np.float32), np.sin(arguments, dtype=np.float32)], axis=1
    )
    time_hidden = weights.linear(
        "time_text_embed.timestep_embedder.linear_1.weight", time_projection, (WIDTH, 256)
    )
    time_embedding = weights.linear(
        "time_text_embed.timestep_embedder.linear_2.weight", silu(time_hidden), (WIDTH, WIDTH)
    )
    activated_time = silu(time_embedding)
    modulation = weights.linear("modulation.1.weight", activated_time, (4 * WIDTH, WIDTH)).reshape(2, 4, WIDTH)

    base = np.concatenate([text_projected, np.zeros((2, WIDTH), dtype=np.float32)], axis=0)
    joint = np.repeat(base, np.where(IMG_MASK, 4, 1), axis=0)
    joint[IMAGE_PAD_MASK] = image_projected
    rope = rope_table()
    selected = np.where(TARGET_MASK[:, None, None], modulation[0:1], modulation[1:2])
    captures: dict[int, np.ndarray] = {}

    hidden = joint
    for block in range(32):
        prefix = f"transformer_blocks.{block}."
        norm1 = layer_norm(hidden) * (np.float32(1.0) + selected[:, 0])
        query = weights.linear(prefix + "attn.to_q.weight", norm1, (WIDTH, WIDTH)).reshape(ROWS, HEADS, HEAD_DIM)
        key = weights.linear(prefix + "attn.to_k.weight", norm1, (WIDTH, WIDTH)).reshape(ROWS, HEADS, HEAD_DIM)
        value = weights.linear(prefix + "attn.to_v.weight", norm1, (WIDTH, WIDTH)).reshape(ROWS, HEADS, HEAD_DIM)
        query = apply_rope(rms_norm(query, weights.bf16(prefix + "attn.norm_q.weight", (HEAD_DIM,))), rope)
        key = apply_rope(rms_norm(key, weights.bf16(prefix + "attn.norm_k.weight", (HEAD_DIM,))), rope)
        attended = attention(query, key, value).reshape(ROWS, WIDTH)
        attention_projected = weights.linear(prefix + "attn.to_out.0.weight", attended, (WIDTH, WIDTH))
        residual = hidden + np.tanh(selected[:, 1], dtype=np.float32) * attention_projected
        norm2 = layer_norm(residual) * (np.float32(1.0) + selected[:, 2])
        gate = weights.linear(prefix + "img_mlp.gate_layer.weight", norm2, (MLP, WIDTH))
        projected = weights.linear(prefix + "img_mlp.proj.weight", norm2, (MLP, WIDTH))
        mlp = weights.linear(prefix + "img_mlp.out.weight", silu(gate) * projected, (WIDTH, MLP))
        hidden = residual + np.tanh(selected[:, 3], dtype=np.float32) * mlp
        if block in CAPTURE_BLOCKS:
            captures[block] = hidden.copy()
        print(f"reference complete transformer block {block + 1}/32", flush=True)

    final_scale = weights.linear("norm_out.linear.weight", activated_time, (WIDTH, WIDTH))
    selected_final_scale = np.where(TARGET_MASK[:, None], final_scale[0:1], final_scale[1:2])
    final_normalized = layer_norm(hidden) * (np.float32(1.0) + selected_final_scale)
    output_value = weights.linear("proj_out.weight", final_normalized, (OUT_CHANNELS, WIDTH))

    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for name, value in {
        "image_input": image_input,
        "text_input": text_input,
        "timestep": timestep,
        "image_projected": image_projected,
        "text_projected": text_projected,
        "time_projection": time_projection,
        "time_embedding": time_embedding,
        "modulation": modulation,
        "joint_input": joint,
        "rope": rope,
        "final_normalized": final_normalized,
        "output": output_value,
    }.items():
        write_tensor(output, name, value, records)
    for block in CAPTURE_BLOCKS:
        write_tensor(output, f"block_{block:02d}_output", captures[block], records)

    metadata = {
        "schema_version": 1,
        "operation": "Complete QwenImage21Transformer2DModel FP32 reference",
        "model_snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "reference": "Pinned Diffusers equations transcribed to NumPy; BF16 checkpoint weights decoded to FP32",
        "shape": {"text_rows": TEXT_ROWS, "image_rows": IMAGE_ROWS, "joint_rows": ROWS, "width": WIDTH},
        "layout": {
            "img_mask_before_expansion": IMG_MASK.tolist(),
            "image_pad_mask": IMAGE_PAD_MASK.tolist(),
            "image_shapes": [[1, 2, 2], [1, 2, 4]],
            "image_ids": IMAGE_IDS.tolist(),
            "target_mask": TARGET_MASK.tolist(),
            "key_valid": KEY_VALID.tolist(),
        },
        "timestep": float(timestep[0]),
        "capture_blocks": list(CAPTURE_BLOCKS),
        "acceptance": {
            "metric": "target-output normalized RMS versus dense FP32 oracle",
            "target_rows": [7, 15],
            "final_limit": 0.012,
        },
        "tensors": records,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/transformer_model_fp32"))
    arguments = parser.parse_args()
    generate(arguments.model.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
