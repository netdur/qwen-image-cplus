#!/usr/bin/env python3
"""Generate the model-width block-0 FP32 oracle without a product dependency.

The equations and tensor ordering are transcribed from the pinned Diffusers
Qwen-Image-2.1 implementation. Checkpoint BF16 values are decoded to FP32 and
all reference activations remain FP32; this deliberately provides a
higher-precision oracle for the Metal correctness path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


DIFFUSERS_COMMIT = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"
MODEL_SNAPSHOT = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
ROWS, WIDTH, HEADS, HEAD_DIM, MLP = 4, 4096, 32, 128, 12288
EPSILON = np.float32(1e-6)


class SafeTensorFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            self.header = json.loads(stream.read(header_length))
        self.data_start = 8 + header_length
        self.mapping = np.memmap(path, mode="r", dtype=np.uint8)

    def bf16(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        bits = self.bf16_bits(name, shape)
        return (bits.astype(np.uint32) << np.uint32(16)).view("<f4").reshape(shape)

    def bf16_bits(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        info = self.header[name]
        if info["dtype"] != "BF16" or tuple(info["shape"]) != shape:
            raise ValueError(f"unexpected tensor metadata for {name}: {info}")
        begin, end = info["data_offsets"]
        return self.mapping[self.data_start + begin : self.data_start + end].view("<u2").reshape(shape)


def layer_norm(value: np.ndarray) -> np.ndarray:
    mean = value.mean(axis=-1, keepdims=True, dtype=np.float32)
    variance = np.square(value - mean, dtype=np.float32).mean(
        axis=-1, keepdims=True, dtype=np.float32
    )
    return (value - mean) / np.sqrt(variance + EPSILON, dtype=np.float32)


def rms_norm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    variance = np.square(value, dtype=np.float32).mean(
        axis=-1, keepdims=True, dtype=np.float32
    )
    return value * (np.float32(1.0) / np.sqrt(variance + EPSILON, dtype=np.float32)) * weight


def rope_table() -> np.ndarray:
    # Two causal text tokens followed by a 1x1x2 target-image block. This is
    # QwenImage21Rope's exact three-axis construction for axes (16, 56, 56).
    positions = [(0, 0, 0), (1, 1, 1), (2, -1, -1), (2, -1, 0)]
    axes = (16, 56, 56)
    table = np.empty((ROWS, HEAD_DIM), dtype="<f4")
    for row, coordinate in enumerate(positions):
        angles = []
        for axis_position, dimension in zip(coordinate, axes):
            frequency = np.power(
                np.float32(10000.0),
                -np.arange(0, dimension, 2, dtype=np.float32) / np.float32(dimension),
                dtype=np.float32,
            )
            angles.append(np.float32(axis_position) * frequency)
        angle = np.concatenate(angles)
        table[row, : HEAD_DIM // 2] = np.cos(angle, dtype=np.float32)
        table[row, HEAD_DIM // 2 :] = np.sin(angle, dtype=np.float32)
    return table


def apply_rope(value: np.ndarray, rope: np.ndarray) -> np.ndarray:
    paired = value.reshape(ROWS, HEADS, HEAD_DIM // 2, 2)
    real, imaginary = paired[..., 0], paired[..., 1]
    cosine = rope[:, None, : HEAD_DIM // 2]
    sine = rope[:, None, HEAD_DIM // 2 :]
    output = np.empty_like(paired)
    output[..., 0] = real * cosine - imaginary * sine
    output[..., 1] = real * sine + imaginary * cosine
    return output.reshape(ROWS, HEADS, HEAD_DIM)


def attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    image_ids = np.array([-1, -1, 0, 0], dtype=np.int32)
    result = np.empty_like(query)
    scale = np.float32(1.0 / math.sqrt(HEAD_DIM))
    for row in range(ROWS):
        allowed = np.array(
            [row >= column or (image_ids[row] >= 0 and image_ids[row] == image_ids[column]) for column in range(ROWS)]
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
    shard = SafeTensorFile(model / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors")
    prefix = "transformer_blocks.0."
    weight = lambda name, shape: shard.bf16(prefix + name, shape)

    index = np.arange(ROWS * WIDTH, dtype=np.float32).reshape(ROWS, WIDTH)
    hidden = (np.sin(index * np.float32(0.0013)) * np.float32(0.25)).astype(np.float32)
    modulation_index = np.arange(2 * 4 * WIDTH, dtype=np.float32).reshape(2, 4, WIDTH)
    modulation = (np.sin(modulation_index * np.float32(0.0007)) * np.float32(0.08)).astype(np.float32)
    target_mask = np.array([False, False, True, True])
    selected = np.where(target_mask[:, None, None], modulation[0:1], modulation[1:2])

    norm1 = layer_norm(hidden) * (np.float32(1.0) + selected[:, 0])
    query = norm1 @ weight("attn.to_q.weight", (WIDTH, WIDTH)).T
    key = norm1 @ weight("attn.to_k.weight", (WIDTH, WIDTH)).T
    value = norm1 @ weight("attn.to_v.weight", (WIDTH, WIDTH)).T
    query = query.reshape(ROWS, HEADS, HEAD_DIM)
    key = key.reshape(ROWS, HEADS, HEAD_DIM)
    value_heads = value.reshape(ROWS, HEADS, HEAD_DIM)
    rope = rope_table()
    query_normalized = rms_norm(query, weight("attn.norm_q.weight", (HEAD_DIM,)))
    key_normalized = rms_norm(key, weight("attn.norm_k.weight", (HEAD_DIM,)))
    query_rope = apply_rope(query_normalized, rope)
    key_rope = apply_rope(key_normalized, rope)
    attended = attention(query_rope, key_rope, value_heads).reshape(ROWS, WIDTH)
    attention_projected = attended @ weight("attn.to_out.0.weight", (WIDTH, WIDTH)).T
    residual1 = hidden + np.tanh(selected[:, 1], dtype=np.float32) * attention_projected

    norm2 = layer_norm(residual1) * (np.float32(1.0) + selected[:, 2])
    gate = norm2 @ weight("img_mlp.gate_layer.weight", (MLP, WIDTH)).T
    projected = norm2 @ weight("img_mlp.proj.weight", (MLP, WIDTH)).T
    swiglu = (gate / (np.float32(1.0) + np.exp(-gate, dtype=np.float32))) * projected
    mlp_projected = swiglu @ weight("img_mlp.out.weight", (WIDTH, MLP)).T
    output_value = residual1 + np.tanh(selected[:, 3], dtype=np.float32) * mlp_projected

    # Prove that this compact input is discriminative for the common porting
    # errors called out by the milestone. These alternatives are not stored;
    # their distance from the oracle is recorded in metadata.
    causal_attention = np.empty_like(query_rope)
    for row in range(ROWS):
        allowed = np.arange(ROWS) <= row
        scores = np.einsum("hd,khd->hk", query_rope[row], key_rope, dtype=np.float32) * np.float32(
            1.0 / math.sqrt(HEAD_DIM)
        )
        scores[:, ~allowed] = -np.inf
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores, dtype=np.float32)
        probabilities /= probabilities.sum(axis=-1, keepdims=True, dtype=np.float32)
        causal_attention[row] = np.einsum("hk,khd->hd", probabilities, value_heads, dtype=np.float32)
    no_rope_attention = attention(query_normalized, key_normalized, value_heads)
    no_qk_norm_attention = attention(query, key, value_heads)
    transposed_query = norm1 @ weight("attn.to_q.weight", (WIDTH, WIDTH))
    linear_gate_output = residual1 + selected[:, 3] * mlp_projected

    def maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.max(np.abs(left - right)))

    diagnostics = {
        "strict_causal_instead_of_block_causal": maximum_difference(attended, causal_attention.reshape(ROWS, WIDTH)),
        "rope_omitted": maximum_difference(attended, no_rope_attention.reshape(ROWS, WIDTH)),
        "qk_rmsnorm_omitted": maximum_difference(attended, no_qk_norm_attention.reshape(ROWS, WIDTH)),
        "q_weight_transposed_wrong_way": maximum_difference(query.reshape(ROWS, WIDTH), transposed_query),
        "residual_gate_without_tanh": maximum_difference(output_value, linear_gate_output),
    }
    if any(value <= 1e-4 for value in diagnostics.values()):
        raise AssertionError(f"block fixture is not discriminative enough: {diagnostics}")

    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for name, tensor in {
        "hidden": hidden,
        "modulation": modulation,
        "rope": rope,
        "norm1": norm1,
        "query": query.reshape(ROWS, WIDTH),
        "key": key.reshape(ROWS, WIDTH),
        "value": value,
        "query_rope": query_rope.reshape(ROWS, WIDTH),
        "key_rope": key_rope.reshape(ROWS, WIDTH),
        "attention": attended,
        "attention_projected": attention_projected,
        "residual1": residual1,
        "norm2": norm2,
        "gate": gate,
        "projected": projected,
        "swiglu": swiglu,
        "mlp_projected": mlp_projected,
        "output": output_value,
    }.items():
        write_tensor(output, name, tensor, records)

    metadata = {
        "schema_version": 1,
        "operation": "QwenImage21TransformerBlock block 0, FP32 reference",
        "model_snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "reference": "Pinned Diffusers equations transcribed to NumPy; BF16 checkpoint weights decoded to FP32",
        "shape": {"rows": ROWS, "width": WIDTH, "heads": HEADS, "head_dim": HEAD_DIM, "mlp": MLP},
        "tokens": {"image_ids": [-1, -1, 0, 0], "target_mask": [False, False, True, True]},
        "tolerance": {"max_abs": 0.003, "max_rel": 0.003},
        "wrong_equation_max_abs_separation": diagnostics,
        "tensors": records,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/block0_fp32"))
    arguments = parser.parse_args()
    generate(arguments.model.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
