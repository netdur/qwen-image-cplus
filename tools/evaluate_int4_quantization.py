#!/usr/bin/env python3
"""Measure INT4 transformer quantization against the FP16 QIPACK.

Runs the complete 32-block transformer in NumPy/FP32 on the committed 256px
trajectory inputs (real prompt embeddings, noise, and RoPE), once with the
FP16 weights and once per candidate quantization, and reports the noise
prediction error. The block equations are the same transcription used by
generate_transformer_model_fixture.py.

Candidates are emulated from the FP16 pack by default. With --packed, the
candidate weights are read from an INT4 QIPACK instead, which checks that a
written pack reproduces its emulation.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_block0_fixture import HEAD_DIM, HEADS, MLP, WIDTH, layer_norm, rms_norm  # noqa: E402
from generate_transformer_model_fixture import gelu_tanh, silu  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests/fixtures/trajectory_256_fp32"
OUT_CHANNELS = 64
BLOCK_MATRICES = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out",
)
CAPTURE_BLOCKS = (0, 7, 15, 23, 31)

SCHEME_BF16 = 0
SCHEME_FP16 = 1
SCHEME_W4A16_AFFINE = 5
SCHEME_W4A4_H256 = 6


# MARK: - QIPACK reading


class Pack:
    def __init__(self, path: Path) -> None:
        self.mapping = np.memmap(path, mode="r", dtype=np.uint8)
        header = bytes(self.mapping[:256])
        magic, _version, _header_bytes, count, _scope, directory = struct.unpack_from("<8sIIIIQ", header, 0)
        if magic != b"QIPACK1\x00":
            raise ValueError(f"{path} is not a QIPACK")
        self.policy = header[136:168].rstrip(b"\0").decode()
        self.records = {}
        for index in range(count):
            record = bytes(self.mapping[directory + 256 * index : directory + 256 * (index + 1)])
            name = record[:128].rstrip(b"\0").decode()
            scheme, _rank = struct.unpack_from("<II", record, 128)
            dimensions = struct.unpack_from("<4Q", record, 136)
            _axis, group_size, _scale_type, _zero_type = struct.unpack_from("<IIII", record, 168)
            offsets = struct.unpack_from("<6Q", record, 184)
            self.records[name] = (scheme, dimensions[0], dimensions[1], group_size, offsets)

    def _bytes(self, offset: int, length: int) -> np.ndarray:
        return self.mapping[offset : offset + length]

    def vector(self, name: str) -> np.ndarray:
        scheme, output, _input, _group, offsets = self.records[name]
        bits = self._bytes(offsets[0], offsets[1]).view("<u2")
        if scheme == SCHEME_BF16:
            return (bits.astype(np.uint32) << np.uint32(16)).view("<f4")
        return bits.view("<f2").astype(np.float32)

    def matrix(self, name: str) -> np.ndarray:
        """Dense FP32 weights as stored (INT4 weights are dequantized, still rotated)."""
        scheme, output, inputs, group, offsets = self.records[name]
        weights_offset, weights_length, scales_offset, scales_length, zeros_offset, zeros_length = offsets
        if scheme in (SCHEME_BF16, SCHEME_FP16):
            return self.vector(name).reshape(output, inputs)
        packed = self._bytes(weights_offset, weights_length)
        codes = np.empty((output, inputs), dtype=np.int8)
        codes.reshape(-1)[0::2] = packed & 0x0F
        codes.reshape(-1)[1::2] = packed >> 4
        scales = self._bytes(scales_offset, scales_length).view("<f2").astype(np.float32).reshape(output, inputs // group)
        grouped = codes.reshape(output, inputs // group, group).astype(np.float32)
        if scheme == SCHEME_W4A16_AFFINE:
            minimum = self._bytes(zeros_offset, zeros_length).view("<f2").astype(np.float32)
            minimum = minimum.reshape(output, inputs // group)
            return (grouped * scales[..., None] + minimum[..., None]).reshape(output, inputs)
        if scheme == SCHEME_W4A4_H256:
            grouped = np.where(grouped > 7, grouped - 16, grouped)
            return (grouped * scales[..., None]).reshape(output, inputs)
        raise ValueError(f"unsupported scheme {scheme} for {name}")


# MARK: - Quantizers (these define the INT4 formats)


HADAMARD_4 = np.float32(0.5) * np.array(
    [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32
)


def hadamard_256(values: np.ndarray) -> np.ndarray:
    """The QIPACK v5 H256 butterfly (radix 4, strides 1/4/16/64) on the last axis in 256-chunks."""
    shape = values.shape
    result = values.reshape(-1, 256).astype(np.float32, copy=True)
    stride = 1
    while stride < 256:
        blocks = result.reshape(result.shape[0], 256 // (4 * stride), 4, stride)
        result = np.einsum("ij,rbjs->rbis", HADAMARD_4, blocks, optimize=True).reshape(-1, 256)
        stride *= 4
    return result.reshape(shape)


def fp16_round(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).astype(np.float32)


CLIP_CANDIDATES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)
# Smallest positive FP16 value, so a rounded scale is never zero.
SCALE_FLOOR = np.float32(2.0 ** -24)


def quantize_affine(weights: np.ndarray, group: int, clip: bool):
    """Unsigned 4-bit codes 0..15 with an FP16 scale and FP16 minimum per group: w = q*s + m."""
    output, inputs = weights.shape
    grouped = weights.reshape(output, inputs // group, group)
    low = grouped.min(axis=-1)
    high = grouped.max(axis=-1)
    best = None
    for shrink in CLIP_CANDIDATES if clip else (1.0,):
        center = (high + low) * np.float32(0.5)
        half = (high - low) * np.float32(0.5 * shrink)
        minimum = fp16_round(center - half)
        scale = fp16_round(np.maximum((center + half - minimum) / np.float32(15.0), SCALE_FLOOR))
        codes = np.clip(np.rint((grouped - minimum[..., None]) / scale[..., None]), 0, 15)
        error = np.square(codes * scale[..., None] + minimum[..., None] - grouped).sum(axis=-1)
        if best is None:
            best = [error, codes, scale, minimum]
        else:
            better = error < best[0]
            best[0] = np.where(better, error, best[0])
            best[1] = np.where(better[..., None], codes, best[1])
            best[2] = np.where(better, scale, best[2])
            best[3] = np.where(better, minimum, best[3])
    return best[1].astype(np.uint8).reshape(output, inputs), best[2], best[3]


def quantize_symmetric(values: np.ndarray, group: int, clip: bool):
    """Signed 4-bit codes -7..7 with one FP16 scale per group: w = q*s."""
    rows, inputs = values.shape
    grouped = values.reshape(rows, inputs // group, group)
    peak = np.abs(grouped).max(axis=-1)
    best = None
    for shrink in CLIP_CANDIDATES if clip else (1.0,):
        scale = fp16_round(np.maximum(peak * np.float32(shrink / 7.0), SCALE_FLOOR))
        codes = np.clip(np.rint(grouped / scale[..., None]), -7, 7)
        if not clip:
            return codes.astype(np.int8).reshape(rows, inputs), scale
        error = np.square(codes * scale[..., None] - grouped).sum(axis=-1)
        if best is None:
            best = [error, codes, scale]
        else:
            better = error < best[0]
            best[0] = np.where(better, error, best[0])
            best[1] = np.where(better[..., None], codes, best[1])
            best[2] = np.where(better, scale, best[2])
    return best[1].astype(np.int8).reshape(rows, inputs), best[2]


def dequantize_symmetric(codes: np.ndarray, scale: np.ndarray, group: int) -> np.ndarray:
    rows, inputs = codes.shape
    return (codes.reshape(rows, inputs // group, group).astype(np.float32) * scale[..., None]).reshape(rows, inputs)


class Mode:
    """name: fp16 | w4a16-g<G>[-clip] | w4a4-g<G>[-clip] | w4a4-h256-g<G>[-clip]"""

    def __init__(self, name: str) -> None:
        self.name = name
        parts = name.split("-")
        self.kind = parts[0]
        self.rotate = "h256" in parts
        self.clip = "clip" in parts
        groups = [int(part[1:]) for part in parts if part.startswith("g") and part[1:].isdigit()]
        self.group = groups[0] if groups else 64
        if self.kind not in ("fp16", "w4a16", "w4a4"):
            raise ValueError(f"unknown mode {name}")

    def prepare(self, weights: np.ndarray) -> np.ndarray:
        """FP32 weights as the kernel will see them (rotated for H256 modes)."""
        if self.kind == "fp16":
            return weights
        if self.kind == "w4a16":
            codes, scale, minimum = quantize_affine(weights, self.group, self.clip)
            output, inputs = weights.shape
            grouped = codes.reshape(output, inputs // self.group, self.group).astype(np.float32)
            return (grouped * scale[..., None] + minimum[..., None]).reshape(output, inputs)
        source = hadamard_256(weights) if self.rotate else weights
        codes, scale = quantize_symmetric(source, self.group, self.clip)
        return dequantize_symmetric(codes, scale, self.group)

    def activations(self, inputs: np.ndarray) -> np.ndarray:
        if self.kind != "w4a4":
            return fp16_round(inputs) if self.kind == "w4a16" else inputs
        source = hadamard_256(inputs) if self.rotate else inputs
        codes, scale = quantize_symmetric(source, self.group, clip=False)
        return dequantize_symmetric(codes, scale, self.group)


# MARK: - Transformer


class Model:
    def __init__(self, source: Pack, mode: Mode, candidate: Pack | None = None) -> None:
        self.source = source
        self.mode = mode
        self.candidate = candidate
        self.quantize_seconds = 0.0

    def global_linear(self, name: str, inputs: np.ndarray) -> np.ndarray:
        return inputs @ self.source.matrix(name).T

    def block_linear(self, name: str, inputs: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        if self.candidate is not None:
            weights = self.candidate.matrix(name)
        else:
            weights = self.mode.prepare(self.source.matrix(name))
        self.quantize_seconds += time.perf_counter() - started
        return self.mode.activations(inputs) @ weights.T


def load_fixture():
    def tensor(name: str, shape: tuple[int, ...]) -> np.ndarray:
        return np.fromfile(FIXTURE / f"{name}.f32", dtype="<f4").reshape(shape)

    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    text_rows = metadata["shape"]["text_rows"]
    target_rows = metadata["shape"]["target_rows"]
    return {
        "text": tensor("text_input", (text_rows, WIDTH)),
        "rope": tensor("rope", (text_rows + target_rows, HEAD_DIM)),
        "timesteps": np.fromfile(FIXTURE / "timesteps.f32", dtype="<f4"),
        "latents": {
            "noise": tensor("image_input", (target_rows, OUT_CHANNELS)),
            "step_02": tensor("latent_step_02", (target_rows, OUT_CHANNELS)),
            "final": tensor("latent_step_40", (target_rows, OUT_CHANNELS)),
        },
        "text_rows": text_rows,
    }


def apply_rope(value: np.ndarray, rope: np.ndarray) -> np.ndarray:
    rows = value.shape[0]
    paired = value.reshape(rows, HEADS, HEAD_DIM // 2, 2)
    cosine = rope[:, None, : HEAD_DIM // 2]
    sine = rope[:, None, HEAD_DIM // 2 :]
    output = np.empty_like(paired)
    output[..., 0] = paired[..., 0] * cosine - paired[..., 1] * sine
    output[..., 1] = paired[..., 0] * sine + paired[..., 1] * cosine
    return output.reshape(rows, HEADS, HEAD_DIM)


def attention(query: np.ndarray, key: np.ndarray, value: np.ndarray, text_rows: int) -> np.ndarray:
    # Text rows are causal over text; the single target image block sees every key.
    rows = query.shape[0]
    allowed = np.ones((rows, rows), dtype=bool)
    allowed[:text_rows, :] = np.tril(np.ones((text_rows, rows), dtype=bool))
    scores = np.einsum("qhd,khd->hqk", query, key, optimize=True) * np.float32(1.0 / math.sqrt(HEAD_DIM))
    scores = np.where(allowed[None], scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probabilities, value, optimize=True).astype(np.float32)


def forward(model: Model, fixture: dict, latent: np.ndarray, timestep: float) -> tuple[np.ndarray, dict]:
    text_rows = fixture["text_rows"]
    image = model.global_linear("img_in.weight", latent)
    text_weight = model.source.vector("txt_in.text_norm.weight") + np.float32(1.0)
    text = model.global_linear("txt_in.in_layer.weight", rms_norm(fixture["text"], text_weight))
    text = model.global_linear("txt_in.out_layer.weight", gelu_tanh(text))

    steps = np.array([np.float32(timestep / 1000.0), np.float32(0.0)], dtype=np.float32)
    frequency = np.exp(
        -np.float32(math.log(10000.0)) * np.arange(128, dtype=np.float32) / np.float32(128.0)
    ).astype(np.float32)
    arguments = steps[:, None] * np.float32(1000.0) * frequency[None]
    projection = np.concatenate([np.cos(arguments), np.sin(arguments)], axis=1).astype(np.float32)
    time_hidden = model.global_linear("time_text_embed.timestep_embedder.linear_1.weight", projection)
    time_embedding = model.global_linear("time_text_embed.timestep_embedder.linear_2.weight", silu(time_hidden))
    activated_time = silu(time_embedding)
    modulation = model.global_linear("modulation.1.weight", activated_time).reshape(2, 4, WIDTH)

    hidden = np.concatenate([text, image], axis=0).astype(np.float32)
    rows = hidden.shape[0]
    target = np.arange(rows) >= text_rows
    selected = np.where(target[:, None, None], modulation[0:1], modulation[1:2])
    rope = fixture["rope"]
    captures = {}
    for block in range(32):
        prefix = f"transformer_blocks.{block}."
        norm1 = layer_norm(hidden) * (np.float32(1.0) + selected[:, 0])
        query = model.block_linear(prefix + "attn.to_q.weight", norm1).reshape(rows, HEADS, HEAD_DIM)
        key = model.block_linear(prefix + "attn.to_k.weight", norm1).reshape(rows, HEADS, HEAD_DIM)
        value = model.block_linear(prefix + "attn.to_v.weight", norm1).reshape(rows, HEADS, HEAD_DIM)
        query = apply_rope(rms_norm(query, model.source.vector(prefix + "attn.norm_q.weight")), rope)
        key = apply_rope(rms_norm(key, model.source.vector(prefix + "attn.norm_k.weight")), rope)
        attended = attention(query, key, value, text_rows).reshape(rows, WIDTH)
        projected = model.block_linear(prefix + "attn.to_out.0.weight", attended)
        residual = hidden + np.tanh(selected[:, 1]) * projected
        norm2 = layer_norm(residual) * (np.float32(1.0) + selected[:, 2])
        gate = model.block_linear(prefix + "img_mlp.gate_layer.weight", norm2)
        up = model.block_linear(prefix + "img_mlp.proj.weight", norm2)
        mlp = model.block_linear(prefix + "img_mlp.out.weight", silu(gate) * up)
        hidden = (residual + np.tanh(selected[:, 3]) * mlp).astype(np.float32)
        if block in CAPTURE_BLOCKS:
            captures[block] = hidden[target].copy()

    final_scale = model.global_linear("norm_out.linear.weight", activated_time)
    selected_final = np.where(target[:, None], final_scale[0:1], final_scale[1:2])
    normalized = layer_norm(hidden) * (np.float32(1.0) + selected_final)
    output = model.global_linear("proj_out.weight", normalized)
    return output[target], captures


def nrmse(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - expected))) / np.sqrt(np.mean(np.square(expected))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="FP16 v4 QIPACK")
    parser.add_argument("--modes", nargs="+", default=["w4a16-g64-clip", "w4a4-h256-g64-clip"])
    parser.add_argument("--packed", type=Path, help="INT4 QIPACK to read instead of emulating (one mode)")
    parser.add_argument("--cases", nargs="+", default=["noise:0", "blend:0.4"],
                        help="latent:timestep-index pairs from the 256px trajectory fixture, or blend:SIGMA "
                             "for the flow-matching point (1-SIGMA)*final latent + SIGMA*noise at t=1000*SIGMA")
    parser.add_argument("--json", type=Path, help="write results here")
    parser.add_argument("--dump-dir", type=Path, help="write each noise prediction here as raw FP32")
    arguments = parser.parse_args()

    source = Pack(arguments.source)
    fixture = load_fixture()
    cases = []
    for case in arguments.cases:
        name, value = case.split(":")
        if name == "blend":
            sigma = float(value)
            latents = fixture["latents"]
            latent = ((1.0 - sigma) * latents["final"] + sigma * latents["noise"]).astype(np.float32)
            cases.append((case, latent, 1000.0 * sigma))
        else:
            cases.append((name, fixture["latents"][name], float(fixture["timesteps"][int(value)])))

    references = {}
    for latent_name, latent, timestep in cases:
        started = time.perf_counter()
        references[latent_name] = forward(Model(source, Mode("fp16")), fixture, latent, timestep)
        print(f"reference {latent_name} t={timestep:.2f}: {time.perf_counter() - started:.1f} s", flush=True)
        if arguments.dump_dir:
            arguments.dump_dir.mkdir(parents=True, exist_ok=True)
            references[latent_name][0].astype("<f4").tofile(arguments.dump_dir / f"fp16_{latent_name}.f32")

    candidate = Pack(arguments.packed) if arguments.packed else None
    results = []
    for mode_name in arguments.modes:
        mode = Mode(mode_name)
        for latent_name, latent, timestep in cases:
            started = time.perf_counter()
            model = Model(source, mode, candidate)
            output, captures = forward(model, fixture, latent, timestep)
            expected_output, expected_captures = references[latent_name]
            if arguments.dump_dir:
                suffix = "packed" if candidate else "emulated"
                output.astype("<f4").tofile(arguments.dump_dir / f"{mode_name}_{suffix}_{latent_name}.f32")
            result = {
                "mode": mode_name,
                "packed": str(arguments.packed) if candidate else None,
                "latent": latent_name,
                "timestep": timestep,
                "output_nrmse": nrmse(output, expected_output),
                "block_nrmse": {str(block): nrmse(captures[block], expected_captures[block]) for block in CAPTURE_BLOCKS},
                "seconds": time.perf_counter() - started,
                "quantize_seconds": model.quantize_seconds,
            }
            results.append(result)
            blocks = " ".join(f"b{block}={value:.4f}" for block, value in result["block_nrmse"].items())
            print(f"{mode_name:24s} {latent_name} t={timestep:7.2f}  output nRMSE {result['output_nrmse']:.5f}  "
                  f"[{blocks}]  {result['seconds']:.0f} s", flush=True)
    if arguments.json:
        arguments.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
