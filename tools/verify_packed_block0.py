#!/usr/bin/env python3
"""Independently execute the C+ QIPACK1 block-0 artifact in NumPy."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from evaluate_block0_quantization import error_metrics, load_f32
from generate_block0_fixture import (
    HEAD_DIM,
    HEADS,
    ROWS,
    WIDTH,
    apply_rope,
    attention,
    layer_norm,
    rms_norm,
)


class PackedBlock:
    def __init__(self, path: Path) -> None:
        self.mapping = np.memmap(path, mode="r", dtype=np.uint8)
        if bytes(self.mapping[:8]) != b"QIPACK1\0":
            raise ValueError("wrong packed magic")
        version, header_bytes, tensor_count = struct.unpack_from("<III", self.mapping, 8)
        if version != 1 or header_bytes != 256 or tensor_count != 9:
            raise ValueError("unsupported packed header")
        self.records = {}
        for index in range(tensor_count):
            offset = 256 + index * 256
            record = self.mapping[offset : offset + 256]
            name = bytes(record[:128]).split(b"\0", 1)[0].decode("utf-8")
            fields = struct.unpack_from("<II4QIIII8Q", record, 128)
            self.records[name] = {
                "scheme": fields[0],
                "rank": fields[1],
                "shape": tuple(fields[2 : 2 + fields[1]]),
                "group_axis": fields[6],
                "group_size": fields[7],
                "weights_offset": fields[10],
                "weights_length": fields[11],
                "scales_offset": fields[12],
                "scales_length": fields[13],
                "zeros_offset": fields[14],
                "zeros_length": fields[15],
            }

    def bf16(self, name: str) -> np.ndarray:
        record = self.records[name]
        begin = record["weights_offset"]
        end = begin + record["weights_length"]
        bits = self.mapping[begin:end].view("<u2")
        return (bits.astype(np.uint32) << np.uint32(16)).view("<f4").reshape(record["shape"])

    def linear(self, name: str, inputs: np.ndarray, chunk_rows: int = 128) -> np.ndarray:
        record = self.records[name]
        if record["scheme"] != 3 or record["group_axis"] != 1:
            raise ValueError(f"{name} is not affine INT8")
        outputs, input_width = record["shape"]
        group_size = record["group_size"]
        groups_per_row = input_width // group_size
        weights = self.mapping[
            record["weights_offset"] : record["weights_offset"] + record["weights_length"]
        ].reshape(outputs, input_width)
        scales = self.mapping[
            record["scales_offset"] : record["scales_offset"] + record["scales_length"]
        ].view("<f2").reshape(outputs, groups_per_row)
        zeros = self.mapping[
            record["zeros_offset"] : record["zeros_offset"] + record["zeros_length"]
        ].reshape(outputs, groups_per_row)
        result = np.empty((inputs.shape[0], outputs), dtype=np.float32)
        for start in range(0, outputs, chunk_rows):
            end = min(start + chunk_rows, outputs)
            dequantized = (
                weights[start:end].reshape(end - start, groups_per_row, group_size).astype(np.float32)
                - zeros[start:end, :, None].astype(np.float32)
            ) * scales[start:end, :, None].astype(np.float32)
            result[:, start:end] = inputs @ dequantized.reshape(end - start, input_width).T
        return result


def execute(packed: PackedBlock, fixtures: Path) -> dict[str, float]:
    hidden = load_f32(fixtures, "hidden", (ROWS, WIDTH))
    modulation = load_f32(fixtures, "modulation", (2, 4, WIDTH))
    rope = load_f32(fixtures, "rope", (ROWS, HEAD_DIM))
    target = np.array([False, False, True, True])
    selected = np.where(target[:, None, None], modulation[0:1], modulation[1:2])
    norm1 = layer_norm(hidden) * (np.float32(1.0) + selected[:, 0])
    prefix = "transformer_blocks.0."
    query = packed.linear(prefix + "attn.to_q.weight", norm1).reshape(ROWS, HEADS, HEAD_DIM)
    key = packed.linear(prefix + "attn.to_k.weight", norm1).reshape(ROWS, HEADS, HEAD_DIM)
    value = packed.linear(prefix + "attn.to_v.weight", norm1).reshape(ROWS, HEADS, HEAD_DIM)
    query = apply_rope(rms_norm(query, packed.bf16(prefix + "attn.norm_q.weight")), rope)
    key = apply_rope(rms_norm(key, packed.bf16(prefix + "attn.norm_k.weight")), rope)
    attended = attention(query, key, value).reshape(ROWS, WIDTH)
    attention_projected = packed.linear(prefix + "attn.to_out.0.weight", attended)
    residual1 = hidden + np.tanh(selected[:, 1], dtype=np.float32) * attention_projected
    norm2 = layer_norm(residual1) * (np.float32(1.0) + selected[:, 2])
    gate = packed.linear(prefix + "img_mlp.gate_layer.weight", norm2)
    projected = packed.linear(prefix + "img_mlp.proj.weight", norm2)
    swiglu = gate / (np.float32(1.0) + np.exp(-gate, dtype=np.float32)) * projected
    mlp = packed.linear(prefix + "img_mlp.out.weight", swiglu)
    output = residual1 + np.tanh(selected[:, 3], dtype=np.float32) * mlp
    expected = load_f32(fixtures, "output", (ROWS, WIDTH))
    return error_metrics(output, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packed", type=Path)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/block0_fp32"))
    arguments = parser.parse_args()
    result = execute(PackedBlock(arguments.packed), arguments.fixtures)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["normalized_rms"] > 0.01:
        raise SystemExit("packed block exceeds the provisional normalized-RMS budget")


if __name__ == "__main__":
    main()
