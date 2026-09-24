#!/usr/bin/env python3
"""Generate a compact oracle for the pinned Qwen-Image-2.1 VAE encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21

from calibrate_transformer_quantization import DIFFUSERS_COMMIT, MODEL_SNAPSHOT, validate_model_root


def hwc(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[2] != 1:
        raise ValueError(f"expected [1,C,1,H,W], got {tuple(value.shape)}")
    return value[0, :, 0].permute(1, 2, 0).detach().cpu().float().numpy()


def write_tensor(root: Path, name: str, value: np.ndarray, records: dict) -> None:
    value = np.asarray(value, dtype="<f4", order="C")
    raw = value.tobytes()
    path = root / f"{name}.f32"
    path.write_bytes(raw)
    records[name] = {
        "file": path.name,
        "shape": list(value.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/vae_encoder_fp32/small_32"),
    )
    args = parser.parse_args()
    model_root = args.model.resolve()
    validate_model_root(model_root)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the VAE encoder oracle")

    device = torch.device("mps")
    model = AutoencoderKLQwenImage21.from_pretrained(
        model_root / "vae",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()

    side = 32
    axis = np.linspace(0.0, 1.0, side, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    rgba = np.stack(
        [xx * 2 - 1, yy * 2 - 1, np.sin(xx * np.pi) * 2 - 1, (xx + yy) - 1],
        axis=-1,
    ).astype(np.float32)
    input_tensor = torch.from_numpy(rgba).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device)

    captures: dict[str, torch.Tensor] = {}
    modules = [
        ("conv_in", model.encoder.conv_in),
        *[(f"down_block_{index:02d}", block) for index, block in enumerate(model.encoder.down_blocks)],
        ("mid", model.encoder.mid_block),
        ("norm_out", model.encoder.norm_out),
        ("conv_out", model.encoder.conv_out),
        ("quant_conv", model.quant_conv),
    ]
    handles = [
        module.register_forward_hook(
            lambda _module, _args, output, label=label: captures.__setitem__(label, output.detach())
        )
        for label, module in modules
    ]
    try:
        with torch.inference_mode():
            posterior = model.encode(input_tensor).latent_dist
            mean = posterior.mode()
        torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    mean_values = torch.tensor(model.config.latents_mean, device=device).view(1, 64, 1, 1, 1)
    std_values = torch.tensor(model.config.latents_std, device=device).view(1, 64, 1, 1, 1)
    normalized = (mean - mean_values) / std_values

    args.output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    write_tensor(args.output, "input_rgba", rgba, records)
    for label, _module in modules:
        write_tensor(args.output, label, hwc(captures[label]), records)
    write_tensor(args.output, "mean", hwc(mean), records)
    write_tensor(args.output, "normalized", hwc(normalized), records)
    metadata = {
        "schema_version": 1,
        "operation": "Qwen-Image-2.1 still-image VAE encode mode",
        "model_snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "input_layout": "HWC RGBA, [-1, 1]",
        "output_layout": "HWC, 64 channels",
        "tensors": records,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote VAE encoder oracle to {args.output}: input={rgba.shape} latent={tuple(hwc(normalized).shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
