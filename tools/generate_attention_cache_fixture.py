#!/usr/bin/env python3
"""Generate a dense FP32 oracle for block-causal prefill and cached decode."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROWS, PREFIX_ROWS, HEADS, HEAD_DIM = 7, 5, 32, 128
WIDTH = HEADS * HEAD_DIM
# Two adjacent condition-image blocks (0 and 1), then padded text, then target.
IMAGE_IDS = np.array([0, 0, 1, 1, -1, 2, 2], dtype=np.int32)
KEY_VALID = np.array([True, True, True, True, False, True, True])


def dense_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = np.empty_like(query)
    scale = np.float32(1.0 / math.sqrt(HEAD_DIM))
    for row in range(ROWS):
        allowed = KEY_VALID & np.array(
            [row >= column or (IMAGE_IDS[row] >= 0 and IMAGE_IDS[row] == IMAGE_IDS[column]) for column in range(ROWS)]
        )
        scores = np.einsum("hd,khd->hk", query[row], key, dtype=np.float32) * scale
        scores[:, ~allowed] = -np.inf
        scores -= scores.max(axis=-1, keepdims=True)
        probability = np.exp(scores, dtype=np.float32)
        probability /= probability.sum(axis=-1, keepdims=True, dtype=np.float32)
        result[row] = np.einsum("hk,khd->hd", probability, value, dtype=np.float32)
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


def main() -> None:
    output = Path("tests/fixtures/attention_cache_fp32")
    output.mkdir(parents=True, exist_ok=True)
    index = np.arange(ROWS * HEADS * HEAD_DIM, dtype=np.float32).reshape(ROWS, HEADS, HEAD_DIM)
    query = (np.sin(index * np.float32(0.0017)) * np.float32(0.4)).astype(np.float32)
    key = (np.cos(index * np.float32(0.0013) + np.float32(0.2)) * np.float32(0.35)).astype(np.float32)
    value = (np.sin(index * np.float32(0.0009) - np.float32(0.4)) * np.float32(0.3)).astype(np.float32)
    expected = dense_attention(query, key, value)

    # The target block is bidirectional and follows every prefix token, so a
    # cached decode over [cached prefix, current target] must equal these rows.
    records: dict[str, dict] = {}
    for name, tensor in {
        "query": query,
        "key": key,
        "value": value,
        "output": expected,
        "target_query": query[PREFIX_ROWS:],
        "target_key": key[PREFIX_ROWS:],
        "target_value": value[PREFIX_ROWS:],
        "target_output": expected[PREFIX_ROWS:],
        "prefix_key": key[:PREFIX_ROWS],
        "prefix_value": value[:PREFIX_ROWS],
    }.items():
        write_tensor(output, name, tensor, records)

    metadata = {
        "schema_version": 1,
        "operation": "Tiled online-softmax block-causal prefill and cached target decode",
        "shape": {"rows": ROWS, "prefix_rows": PREFIX_ROWS, "heads": HEADS, "head_dim": HEAD_DIM},
        "image_ids": IMAGE_IDS.tolist(),
        "key_valid": KEY_VALID.tolist(),
        "adjacent_condition_blocks": [[0, 2], [2, 4]],
        "cache_point": "post-RoPE K and raw V",
        "tolerance": {"max_abs": 0.00001, "max_rel": 0.00001},
        "tensors": records,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
