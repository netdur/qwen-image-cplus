#!/usr/bin/env python3
"""Generate compact production-scale Qwen-Image-2.1 transformer oracles.

This development-only tool runs the pinned Diffusers transformer with real
prompt embeddings and deterministic latent noise. It stores complete inputs
and target noise predictions, but only 32 rows at selected block boundaries so
the 256/512/1024-pixel fixtures remain suitable for source control.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import __version__ as accelerate_version
from diffusers import __version__ as diffusers_version
from transformers import __version__ as transformers_version

from calibrate_transformer_quantization import (
    CASES,
    DIFFUSERS_COMMIT,
    MODEL_SNAPSHOT,
    PROMPTS,
    encode_prompts,
    make_inputs,
    schedule_timestep,
    validate_model_root,
)


CAPTURE_BLOCKS = (0, 1, 7, 15, 31)
SAMPLED_ROWS = 32
SELECTED_CASES = ("short-256-early", "spatial-512-early", "text-1024-middle")


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


def write_i32(directory: Path, name: str, value: np.ndarray, records: dict) -> None:
    value = np.asarray(value, dtype="<i4", order="C")
    raw = value.tobytes()
    filename = f"{name}.i32"
    (directory / filename).write_bytes(raw)
    records[name] = {
        "file": filename,
        "dtype": "I32",
        "shape": list(value.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def sampled_indices(rows: int) -> torch.Tensor:
    count = min(SAMPLED_ROWS, rows)
    return torch.linspace(0, rows - 1, count).round().long()


def generate_case(model, case, prompt: torch.Tensor, timestep: float, output_root: Path) -> None:
    inputs = make_inputs(case, prompt, timestep, torch.device("mps"))
    target_rows = inputs["hidden_states"].shape[1]
    text_rows = inputs["encoder_hidden_states"].shape[1]
    joint_rows = text_rows + target_rows
    indices = sampled_indices(joint_rows)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for block_index in CAPTURE_BLOCKS:
        handles.append(
            model.transformer_blocks[block_index].register_forward_hook(
                lambda _module, _args, value, index=block_index: captures.__setitem__(
                    index, value[:, indices].detach().cpu().float()
                )
            )
        )

    # With no condition images the expanded joint sequence is simply the text
    # prefix followed by the complete target grid.
    image_pad_mask = torch.cat(
        [
            torch.zeros(text_rows, dtype=torch.bool, device="mps"),
            torch.ones(target_rows, dtype=torch.bool, device="mps"),
        ]
    )
    side = case.pixels // 16
    rotary = model.pos_embed([(1, side, side)], image_pad_mask, device=torch.device("mps"))
    rope = torch.cat([rotary.real, rotary.imag], dim=-1).detach().cpu().float()

    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model(**inputs)[0]
        torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - started

    directory = output_root / str(case.pixels)
    directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    write_tensor(directory, "image_input", inputs["hidden_states"].detach().cpu().float().numpy(), records)
    write_tensor(
        directory,
        "text_input",
        inputs["encoder_hidden_states"].detach().cpu().float().numpy(),
        records,
    )
    write_tensor(
        directory,
        "timestep",
        np.array([float(inputs["timestep"].cpu()[0]), 0.0], dtype=np.float32),
        records,
    )
    write_tensor(directory, "rope", rope.numpy(), records)
    write_tensor(directory, "output", output[:, -target_rows:].detach().cpu().float().numpy(), records)
    write_i32(directory, "sample_rows", indices.numpy(), records)
    for block_index in CAPTURE_BLOCKS:
        write_tensor(
            directory,
            f"block_{block_index:02d}_sample",
            captures[block_index].numpy(),
            records,
        )

    metadata = {
        "schema_version": 1,
        "operation": "Production-scale QwenImage21Transformer2DModel BF16 reference",
        "model_snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "environment": {
            "torch": torch.__version__,
            "diffusers": diffusers_version,
            "transformers": transformers_version,
            "accelerate": accelerate_version,
            "device": "mps",
        },
        "case": case.name,
        "prompt": PROMPTS[case.prompt_index],
        "latent_seed": case.seed,
        "timestep_index": case.timestep_index,
        "scheduler_timestep": timestep,
        "normalized_timestep": float(inputs["timestep"].cpu()[0]),
        "pixels": [case.pixels, case.pixels],
        "shape": {
            "text_rows": text_rows,
            "target_rows": target_rows,
            "joint_rows": joint_rows,
            "width": 4096,
            "channels": 64,
            "heads": 32,
            "head_dimension": 128,
        },
        "layout": {
            "image_shapes": [[1, side, side]],
            "image_ids": "-1 for the text prefix, 0 for every target row",
            "target_rows": [text_rows, joint_rows],
            "key_valid": "all rows",
        },
        "capture_blocks": list(CAPTURE_BLOCKS),
        "sampled_rows": indices.tolist(),
        "reference_seconds": elapsed,
        "reference_dtype": "BF16 model and inputs; fixture tensors exported as F32",
        "acceptance": {
            "metric": "target-output normalized RMS versus pinned BF16 Diffusers",
            "final_limit": 0.015,
        },
        "tensors": records,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {case.name}: text={text_rows}, target={target_rows}, "
        f"joint={joint_rows}, reference={elapsed:.2f}s",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/transformer_scale_fp32")
    )
    parser.add_argument("--pixels", type=int, choices=(256, 512, 1024), action="append")
    args = parser.parse_args()
    model_root = args.model.resolve()
    validate_model_root(model_root)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for production-scale fixture generation")

    selected_pixels = set(args.pixels or (256, 512, 1024))
    selected = [
        case for case in CASES if case.name in SELECTED_CASES and case.pixels in selected_pixels
    ]
    prompts = encode_prompts(model_root, torch.device("mps"))
    timesteps = {
        case.name: schedule_timestep(
            model_root, (case.pixels // 16) ** 2, case.timestep_index, torch.device("mps")
        )
        for case in selected
    }

    from diffusers import QwenImage21Transformer2DModel

    started = time.perf_counter()
    model = QwenImage21Transformer2DModel.from_pretrained(
        model_root / "transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("mps").eval()
    print(f"loaded pinned transformer in {time.perf_counter() - started:.2f}s", flush=True)
    for case in selected:
        generate_case(model, case, prompts[case.prompt_index], timesteps[case.name], args.output)
    del model, prompts
    gc.collect()
    torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
