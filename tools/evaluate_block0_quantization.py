#!/usr/bin/env python3
"""Measure candidate INT4 and INT8 policies on real Qwen-Image-2.1 block-0 weights.

This is an offline engineering tool, not part of the C+ runtime. It evaluates
stored-FP16 per-group scales, reports each linear role independently, and then
propagates every uniform candidate through the complete four-token block.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from generate_block0_fixture import (
    DIFFUSERS_COMMIT,
    EPSILON,
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
)


PREFIX = "transformer_blocks.0."
ROLES = {
    "q": ("attn.to_q.weight", WIDTH, WIDTH, "norm1", "query"),
    "k": ("attn.to_k.weight", WIDTH, WIDTH, "norm1", "key"),
    "v": ("attn.to_v.weight", WIDTH, WIDTH, "norm1", "value"),
    "attention_output": ("attn.to_out.0.weight", WIDTH, WIDTH, "attention", "attention_projected"),
    "mlp_gate": ("img_mlp.gate_layer.weight", MLP, WIDTH, "norm2", "gate"),
    "mlp_projection": ("img_mlp.proj.weight", MLP, WIDTH, "norm2", "projected"),
    "mlp_output": ("img_mlp.out.weight", WIDTH, MLP, "swiglu", "mlp_projected"),
}


def load_f32(directory: Path, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(directory / f"{name}.f32", dtype="<f4")
    return value.reshape(shape)


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(difference)
    rms = math.sqrt(float(np.mean(np.square(difference))))
    reference_rms = math.sqrt(float(np.mean(np.square(expected.astype(np.float64)))))
    return {
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "rms": rms,
        "reference_rms": reference_rms,
        "normalized_rms": rms / max(reference_rms, 1e-12),
    }


def packed_bytes(outputs: int, inputs: int, group_size: int, affine: bool, bits: int) -> int:
    groups = outputs * (inputs // group_size)
    return outputs * inputs * bits // 8 + groups * (3 if affine else 2)


def quantized_linear(
    shard: SafeTensorFile,
    tensor_name: str,
    shape: tuple[int, int],
    inputs: np.ndarray,
    mode: str,
    group_size: int,
    bits_per_weight: int,
    chunk_rows: int = 128,
) -> np.ndarray:
    outputs, input_width = shape
    bits = shard.bf16_bits(PREFIX + tensor_name, shape)
    result = np.empty((inputs.shape[0], outputs), dtype=np.float32)
    group_count = input_width // group_size
    for start in range(0, outputs, chunk_rows):
        end = min(start + chunk_rows, outputs)
        chunk_bits = bits[start:end]
        weights = (chunk_bits.astype(np.uint32) << np.uint32(16)).view("<f4")
        grouped = weights.reshape(end - start, group_count, group_size)
        if mode == "symmetric":
            maximum_level = (1 << (bits_per_weight - 1)) - 1
            scale = np.max(np.abs(grouped), axis=-1) / np.float32(maximum_level)
            scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float16).astype(np.float32)
            quantized = np.clip(
                np.rint(grouped / scale[..., None]), -maximum_level, maximum_level
            ).astype(np.int8)
            dequantized = quantized.astype(np.float32) * scale[..., None]
        elif mode == "affine":
            maximum_level = (1 << bits_per_weight) - 1
            minimum = grouped.min(axis=-1)
            maximum = grouped.max(axis=-1)
            scale = (maximum - minimum) / np.float32(maximum_level)
            scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float16).astype(np.float32)
            zero = np.clip(np.rint(-minimum / scale), 0, maximum_level).astype(np.uint8)
            quantized = np.clip(
                np.rint(grouped / scale[..., None]) + zero[..., None], 0, maximum_level
            ).astype(np.uint8)
            dequantized = (quantized.astype(np.float32) - zero[..., None].astype(np.float32)) * scale[..., None]
        else:
            raise ValueError(mode)
        result[:, start:end] = inputs @ dequantized.reshape(end - start, input_width).T
    return result


def dense_linear(
    shard: SafeTensorFile,
    tensor_name: str,
    shape: tuple[int, int],
    inputs: np.ndarray,
    chunk_rows: int = 128,
) -> np.ndarray:
    outputs, input_width = shape
    bits = shard.bf16_bits(PREFIX + tensor_name, shape)
    result = np.empty((inputs.shape[0], outputs), dtype=np.float32)
    for start in range(0, outputs, chunk_rows):
        end = min(start + chunk_rows, outputs)
        weights = (bits[start:end].astype(np.uint32) << np.uint32(16)).view("<f4")
        result[:, start:end] = inputs @ weights.T
    return result


def role_linear(
    shard: SafeTensorFile,
    role: str,
    inputs: np.ndarray,
    policies: dict[str, tuple[str, int, int] | None],
) -> np.ndarray:
    tensor_name, outputs, input_width, _, _ = ROLES[role]
    policy = policies.get(role)
    if policy is None:
        return dense_linear(shard, tensor_name, (outputs, input_width), inputs)
    mode, group_size, bits_per_weight = policy
    return quantized_linear(
        shard,
        tensor_name,
        (outputs, input_width),
        inputs,
        mode,
        group_size,
        bits_per_weight,
    )


def block_forward(
    shard: SafeTensorFile,
    fixtures: dict[str, np.ndarray],
    policies: dict[str, tuple[str, int, int] | None],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    hidden = fixtures["hidden"]
    modulation = fixtures["modulation"]
    selected = np.where(
        np.array([False, False, True, True])[:, None, None], modulation[0:1], modulation[1:2]
    )
    norm1 = layer_norm(hidden) * (np.float32(1.0) + selected[:, 0])
    query = role_linear(shard, "q", norm1, policies)
    key = role_linear(shard, "k", norm1, policies)
    value = role_linear(shard, "v", norm1, policies)
    query = query.reshape(ROWS, HEADS, HEAD_DIM)
    key = key.reshape(ROWS, HEADS, HEAD_DIM)
    value_heads = value.reshape(ROWS, HEADS, HEAD_DIM)
    query_rope = apply_rope(
        rms_norm(query, shard.bf16(PREFIX + "attn.norm_q.weight", (HEAD_DIM,))), fixtures["rope"]
    )
    key_rope = apply_rope(
        rms_norm(key, shard.bf16(PREFIX + "attn.norm_k.weight", (HEAD_DIM,))), fixtures["rope"]
    )
    attended = attention(query_rope, key_rope, value_heads).reshape(ROWS, WIDTH)
    attention_projected = role_linear(shard, "attention_output", attended, policies)
    residual1 = hidden + np.tanh(selected[:, 1], dtype=np.float32) * attention_projected
    norm2 = layer_norm(residual1) * (np.float32(1.0) + selected[:, 2])
    gate = role_linear(shard, "mlp_gate", norm2, policies)
    projected = role_linear(shard, "mlp_projection", norm2, policies)
    swiglu = (gate / (np.float32(1.0) + np.exp(-gate, dtype=np.float32))) * projected
    mlp_projected = role_linear(shard, "mlp_output", swiglu, policies)
    output = residual1 + np.tanh(selected[:, 3], dtype=np.float32) * mlp_projected
    boundaries = {
        "attention_projected": error_metrics(attention_projected, fixtures["attention_projected"]),
        "residual1": error_metrics(residual1, fixtures["residual1"]),
        "mlp_projected": error_metrics(mlp_projected, fixtures["mlp_projected"]),
        "output": error_metrics(output, fixtures["output"]),
    }
    return output, boundaries


def evaluate(model: Path, fixture_directory: Path) -> dict:
    shard = SafeTensorFile(model / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors")
    fixtures = {
        "hidden": load_f32(fixture_directory, "hidden", (ROWS, WIDTH)),
        "modulation": load_f32(fixture_directory, "modulation", (2, 4, WIDTH)),
        "rope": load_f32(fixture_directory, "rope", (ROWS, HEAD_DIM)),
    }
    for _, (_, outputs, _, input_name, output_name) in ROLES.items():
        if input_name not in fixtures:
            width = MLP if input_name == "swiglu" else WIDTH
            fixtures[input_name] = load_f32(fixture_directory, input_name, (ROWS, width))
        if output_name not in fixtures:
            width = MLP if output_name in ("gate", "projected") else WIDTH
            fixtures[output_name] = load_f32(fixture_directory, output_name, (ROWS, width))
    fixtures.update(
        {
            "attention_projected": load_f32(fixture_directory, "attention_projected", (ROWS, WIDTH)),
            "residual1": load_f32(fixture_directory, "residual1", (ROWS, WIDTH)),
            "mlp_projected": load_f32(fixture_directory, "mlp_projected", (ROWS, WIDTH)),
            "output": load_f32(fixture_directory, "output", (ROWS, WIDTH)),
        }
    )

    candidates = [
        (mode, group, bits)
        for bits in (4, 8)
        for mode in ("symmetric", "affine")
        for group in (32, 64, 128)
    ]
    direct: dict[str, dict] = {}
    full_block: dict[str, dict] = {}
    started = time.perf_counter()
    candidate_policies: dict[str, tuple[str, int, int]] = {}
    for mode, group_size, bits_per_weight in candidates:
        candidate = f"int{bits_per_weight}_{mode}_g{group_size}"
        candidate_policies[candidate] = (mode, group_size, bits_per_weight)
        print(f"evaluating {candidate}", flush=True)
        role_results = {}
        total_bytes = 0
        for role, (tensor_name, outputs, inputs, input_name, output_name) in ROLES.items():
            actual = quantized_linear(
                shard,
                tensor_name,
                (outputs, inputs),
                fixtures[input_name],
                mode,
                group_size,
                bits_per_weight,
            )
            affine = mode == "affine"
            role_results[role] = {
                **error_metrics(actual, fixtures[output_name]),
                "packed_bytes": packed_bytes(outputs, inputs, group_size, affine, bits_per_weight),
                "compression_vs_bf16": (outputs * inputs * 2)
                / packed_bytes(outputs, inputs, group_size, affine, bits_per_weight),
            }
            total_bytes += role_results[role]["packed_bytes"]
        direct[candidate] = {"roles": role_results, "block_weight_bytes": total_bytes}
        _, boundaries = block_forward(
            shard,
            fixtures,
            {role: (mode, group_size, bits_per_weight) for role in ROLES},
        )
        full_block[candidate] = {"boundaries": boundaries}

    isolated_q4 = {}
    q4_policy = ("affine", 32, 4)
    for role in ROLES:
        print(f"evaluating isolated Q4 role {role}", flush=True)
        _, boundaries = block_forward(shard, fixtures, {role: q4_policy})
        isolated_q4[role] = boundaries["output"]

    # Build a mixed candidate: Q4 only where its isolated block error is under
    # 1%, and the best passing uniform fallback everywhere else.
    passing_uniform = [
        name
        for name, result in full_block.items()
        if result["boundaries"]["output"]["normalized_rms"] <= 0.01
    ]
    fallback = (
        min(passing_uniform, key=lambda name: direct[name]["block_weight_bytes"])
        if passing_uniform
        else None
    )
    mixed = None
    if fallback is not None:
        fallback_policy = candidate_policies[fallback]
        mixed_policies = {
            role: q4_policy if isolated_q4[role]["normalized_rms"] <= 0.01 else fallback_policy
            for role in ROLES
        }
        _, mixed_boundaries = block_forward(shard, fixtures, mixed_policies)
        mixed_bytes = 0
        for role, policy in mixed_policies.items():
            _, outputs, inputs, _, _ = ROLES[role]
            mode, group, bits = policy
            mixed_bytes += packed_bytes(outputs, inputs, group, mode == "affine", bits)
        mixed = {
            "policies": {
                role: f"int{policy[2]}_{policy[0]}_g{policy[1]}"
                for role, policy in mixed_policies.items()
            },
            "block_weight_bytes": mixed_bytes,
            "boundaries": mixed_boundaries,
        }
    eligible = [(name, direct[name]["block_weight_bytes"]) for name in passing_uniform]
    if mixed is not None and mixed["boundaries"]["output"]["normalized_rms"] <= 0.01:
        eligible.append(("mixed", mixed["block_weight_bytes"]))
    chosen = min(eligible, key=lambda item: item[1])[0] if eligible else None
    return {
        "schema_version": 1,
        "model": "Qwen/Qwen-Image-2.1",
        "snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "calibration": {
            "block": 0,
            "tokens": ROWS,
            "fixture": "tests/fixtures/block0_fp32",
            "limitation": "One deterministic four-token model-width block fixture; policy is provisional until prompt/timestep/resolution calibration is available.",
        },
        "storage": {
            "nibbles": "two little-endian logical K values per byte, low nibble first",
            "scale": "FP16 per output-row K-group",
            "symmetric_levels": [-7, 7],
            "affine_levels": [0, 15],
            "affine_zero_point": "U8 per output-row K-group",
            "int8_candidates": "same scale/zero-point representation with one byte per weight",
        },
        "selection": {
            "criterion": "smallest block storage with final normalized RMS <= 0.01",
            "provisional_candidate": chosen,
        },
        "direct_linear": direct,
        "complete_block": full_block,
        "isolated_q4_affine_g32": isolated_q4,
        "mixed_candidate": mixed,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/block0_fp32"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/m1-max-quantization.json"))
    arguments = parser.parse_args()
    result = evaluate(arguments.model.resolve(), arguments.fixtures.resolve())
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
