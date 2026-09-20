#!/usr/bin/env python3
"""Generate small, source-controlled oracle fixtures from pinned Diffusers.

This is a development tool. It is never imported or executed by the C+ product.
It intentionally refuses any Diffusers checkout other than the pinned commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


DIFFUSERS_COMMIT = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"
MODEL_SNAPSHOT = "b3179ad355be050328e483a9dfdd9e60cd62adfa"


def require_pinned_checkout(source: Path) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != DIFFUSERS_COMMIT:
        raise SystemExit(
            f"Diffusers checkout is {actual}; expected pinned commit {DIFFUSERS_COMMIT}"
        )
    if not (source / "src" / "diffusers").is_dir():
        raise SystemExit(f"not a Diffusers source checkout: {source}")


def tensor_bytes(torch, tensor) -> tuple[bytes, str]:
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(), "BF16"
    encoding = {
        torch.float32: ("<f4", "F32"),
        torch.float16: ("<f2", "F16"),
        torch.int32: ("<i4", "I32"),
        torch.int64: ("<i8", "I64"),
        torch.bool: ("u1", "BOOL"),
    }.get(value.dtype)
    if encoding is None:
        raise TypeError(f"unsupported fixture dtype: {value.dtype}")
    numpy_dtype, label = encoding
    return value.numpy().astype(numpy_dtype, copy=False).tobytes(), label


def write_fixture(root: Path, name: str, operation: str, tensors: dict, tolerance: dict) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for tensor_name, tensor in tensors.items():
        raw, dtype = tensor_bytes(sys.modules["torch"], tensor)
        filename = f"{tensor_name}.bin"
        (directory / filename).write_bytes(raw)
        records[tensor_name] = {
            "file": filename,
            "dtype": dtype,
            "shape": list(tensor.shape),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    metadata = {
        "schema_version": 1,
        "operation": operation,
        "model_snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "seed": 20260920,
        "tolerance": tolerance,
        "tensors": records,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate(source: Path, output: Path) -> None:
    require_pinned_checkout(source)
    sys.path.insert(0, str(source / "src"))

    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21TemporalTimesteps,
            QwenImage21ZeroCenterRMSNorm,
            apply_rotary_emb_qwen,
        )
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
    except ImportError as error:
        raise SystemExit(
            "fixture generation requires a development Python environment with "
            "PyTorch and the pinned Diffusers checkout's dependencies"
        ) from error

    torch.manual_seed(20260920)
    tolerance = {"max_abs": 1e-6, "max_rel": 1e-6}

    source_values = torch.tensor(
        [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 7.5], dtype=torch.float32
    )
    bf16 = source_values.to(torch.bfloat16)
    write_fixture(
        output,
        "bf16_decode",
        "IEEE BF16 storage decoded to FP32",
        {"input": bf16, "output": bf16.float()},
        {"exact": True},
    )

    hidden = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32).reshape(2, 8)
    norm = QwenImage21ZeroCenterRMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.2, 0.2, 8))
        normalized = norm(hidden)
    write_fixture(
        output,
        "zero_centered_rms_norm",
        "QwenImage21ZeroCenterRMSNorm",
        {"input": hidden, "weight": norm.weight, "output": normalized},
        tolerance,
    )

    write_fixture(
        output,
        "affine_free_layer_norm",
        "torch.nn.functional.layer_norm without affine parameters",
        {"input": hidden, "output": functional.layer_norm(hidden, (8,), eps=1e-6)},
        tolerance,
    )

    gate = torch.linspace(-3.0, 3.0, 16, dtype=torch.float32).reshape(2, 8)
    projected = torch.linspace(1.0, -1.0, 16, dtype=torch.float32).reshape(2, 8)
    write_fixture(
        output,
        "activations",
        "tanh GELU, SiLU, and QwenImage21 SwiGLU product",
        {
            "input": gate,
            "projected": projected,
            "gelu": functional.gelu(gate, approximate="tanh"),
            "silu": functional.silu(gate),
            "swiglu": functional.silu(gate) * projected,
        },
        tolerance,
    )

    timesteps = torch.tensor([0.0, 0.001, 0.5, 1.0], dtype=torch.float32)
    temporal = QwenImage21TemporalTimesteps(timestep_dim=256)
    write_fixture(
        output,
        "timestep_embedding",
        "QwenImage21TemporalTimesteps with time_factor=1000",
        {"timestep": timesteps, "output": temporal(timesteps)},
        tolerance,
    )

    rope_input = torch.linspace(-1.0, 1.0, 48, dtype=torch.float32).reshape(1, 3, 2, 8)
    angles = torch.linspace(0.0, 1.25, 12, dtype=torch.float32).reshape(3, 4)
    frequencies = torch.polar(torch.ones_like(angles), angles)
    rope_output = apply_rotary_emb_qwen(rope_input, frequencies, use_real=False)
    write_fixture(
        output,
        "complex_rope",
        "apply_rotary_emb_qwen complex-pair path",
        {
            "input": rope_input,
            "cosine": frequencies.real,
            "sine": frequencies.imag,
            "output": rope_output,
        },
        tolerance,
    )

    image_ids = torch.tensor([-1, -1, 0, 0, 0, 1, 1], dtype=torch.int32)
    indices = torch.arange(image_ids.numel())
    query = indices[:, None]
    key = indices[None, :]
    same_image = (image_ids[:, None] == image_ids[None, :]) & (image_ids[:, None] >= 0)
    mask = (query >= key) | same_image
    write_fixture(
        output,
        "block_causal_mask",
        "build_qwenimage21_block_causal_mask predicate before 128-token padding",
        {"image_ids": image_ids, "output": mask},
        {"exact": True},
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=0.9,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        shift_terminal=0.02,
        time_shift_type="exponential",
        stochastic_sampling=False,
    )
    image_sequence_length = 4096
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = image_sequence_length * slope + (0.5 - slope * 256)
    scheduler.set_timesteps(
        sigmas=np.linspace(1.0, 1.0 / 40.0, 40),
        mu=mu,
    )
    write_fixture(
        output,
        "flowmatch_schedule_1024",
        "FlowMatchEulerDiscreteScheduler for a 4096-token target and 40 steps",
        {"sigmas": scheduler.sigmas, "timesteps": scheduler.timesteps},
        tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusers-source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/small"))
    arguments = parser.parse_args()
    generate(arguments.diffusers_source.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
