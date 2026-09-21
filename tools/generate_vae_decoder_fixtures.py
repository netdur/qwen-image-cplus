#!/usr/bin/env python3
"""Generate pinned Qwen-Image-2.1 still-image VAE decoder fixtures.

The small 1x1-latent case retains every major decoder boundary.  The 16x16
case decodes the committed 40-step trajectory latent to a complete 256x256
four-channel sample, proving the real pipeline handoff without committing
large intermediate feature maps.  Files use NHWC order for direct native
Metal consumption even though the official module executes NCTHW tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import __version__ as accelerate_version
from diffusers import AutoencoderKLQwenImage21
from diffusers import __version__ as diffusers_version
from PIL import Image
from transformers import __version__ as transformers_version

from calibrate_transformer_quantization import DIFFUSERS_COMMIT, MODEL_SNAPSHOT, validate_model_root


def write_tensor(directory: Path, name: str, value: np.ndarray, records: dict) -> None:
    value = np.asarray(value, dtype="<f4", order="C")
    raw = value.tobytes()
    filename = f"{name}.f32"
    (directory / filename).write_bytes(raw)
    records[name] = {
        "file": filename,
        "dtype": "F32",
        "layout": "HWC",
        "shape": list(value.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_rgba(directory: Path, value: torch.Tensor, records: dict) -> dict:
    pixels = np.clip(hwc(value) * 0.5 + 0.5, 0.0, 1.0)
    pixels = np.rint(pixels * 255.0).astype(np.uint8)
    raw = pixels.tobytes()
    filename = "rgba_u8.bin"
    (directory / filename).write_bytes(raw)
    png_path = directory / "reference.png"
    Image.fromarray(pixels, mode="RGBA").save(png_path)
    records["rgba_u8"] = {
        "file": filename,
        "dtype": "U8",
        "layout": "RGBA",
        "shape": list(pixels.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    png = png_path.read_bytes()
    return {
        "file": png_path.name,
        "format": "PNG",
        "mode": "RGBA",
        "bytes": len(png),
        "sha256": hashlib.sha256(png).hexdigest(),
    }


def hwc(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[2] != 1:
        raise ValueError(f"expected [1,C,1,H,W], got {tuple(value.shape)}")
    return value[0, :, 0].permute(1, 2, 0).detach().cpu().float().numpy()


def summary(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().cpu().float()
    return {
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
        "rms": float(value.square().mean().sqrt()),
    }


def denormalize(model: AutoencoderKLQwenImage21, normalized: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(model.config.latents_mean, device=normalized.device).view(1, 64, 1, 1, 1)
    std = torch.tensor(model.config.latents_std, device=normalized.device).view(1, 64, 1, 1, 1)
    return normalized.float() * std + mean


def run_case(
    model: AutoencoderKLQwenImage21,
    name: str,
    normalized: torch.Tensor,
    output_root: Path,
    retain_boundaries: bool,
) -> None:
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def capture(label: str):
        return lambda _module, _args, output: captures.__setitem__(label, output.detach())

    modules = [
        ("post_quant", model.post_quant_conv),
        ("conv_in", model.decoder.conv_in),
        ("mid", model.decoder.mid_block),
        *[(f"up_block_{index:02d}", block) for index, block in enumerate(model.decoder.up_blocks)],
        ("norm_out", model.decoder.norm_out),
        ("conv_out", model.decoder.conv_out),
    ]
    if retain_boundaries:
        for block_index, block in enumerate(model.decoder.up_blocks):
            for resnet_index, resnet in enumerate(block.resnets):
                modules.append((f"up_block_{block_index:02d}_resnet_{resnet_index}", resnet))
            if block.upsampler is not None:
                modules.append((f"up_block_{block_index:02d}_main_upsample", block.upsampler))
            if block.avg_shortcut is not None:
                modules.append((f"up_block_{block_index:02d}_shortcut", block.avg_shortcut))
    for label, module in modules:
        handles.append(module.register_forward_hook(capture(label)))

    decoded_input = denormalize(model, normalized)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model.decode(decoded_input, return_dict=False)[0]
        torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - started

    directory = output_root / name
    directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    write_tensor(directory, "normalized_latent", hwc(normalized), records)
    write_tensor(directory, "denormalized_latent", hwc(decoded_input), records)
    if retain_boundaries:
        for label, _module in modules:
            write_tensor(directory, label, hwc(captures[label]), records)
    write_tensor(directory, "output", hwc(output), records)
    reference_png = write_rgba(directory, output, records)

    metadata = {
        "schema_version": 1,
        "operation": "Qwen-Image-2.1 still-image causal-3D VAE decode",
        "model_snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "environment": {
            "torch": torch.__version__,
            "diffusers": diffusers_version,
            "transformers": transformers_version,
            "accelerate": accelerate_version,
            "device": "mps",
        },
        "case": name,
        "reference_seconds": elapsed,
        "normalized_latent_shape_ncthw": list(normalized.shape),
        "decoded_output_shape_ncthw": list(output.shape),
        "fixture_layout": "HWC contiguous",
        "latent_handoff": "BF16-rounded scheduler latent converted to FP32, then latent * per-channel std + mean",
        "still_image_specialization": {
            "frames": 1,
            "first_chunk": True,
            "causal_conv": "squeeze time and execute checkpoint Conv2d",
            "upsample3d_main_path": "first cache visit records Rep and skips time_conv",
            "upsample_shortcut": "DupUp3D followed by first_chunk temporal crop",
        },
        "summaries": {label: summary(value) for label, value in captures.items()},
        "output_summary": summary(output),
        "reference_png": reference_png,
        "tensors": records,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {name}: latent={tuple(normalized.shape)} output={tuple(output.shape)} "
        f"reference={elapsed:.2f}s",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--trajectory-fixture",
        type=Path,
        default=Path("tests/fixtures/trajectory_256_fp32/latent_step_40.f32"),
    )
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/vae_decoder_fp32"))
    args = parser.parse_args()
    model_root = args.model.resolve()
    validate_model_root(model_root)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for VAE fixture generation")

    device = torch.device("mps")
    started = time.perf_counter()
    model = AutoencoderKLQwenImage21.from_pretrained(
        model_root / "vae",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    print(f"loaded pinned VAE in {time.perf_counter() - started:.2f}s", flush=True)

    generator = torch.Generator(device="cpu").manual_seed(2101)
    small = torch.randn((1, 64, 1, 1, 1), generator=generator, dtype=torch.float32)
    small = small.to(torch.bfloat16).float().to(device)
    run_case(model, "small_1x1", small, args.output, retain_boundaries=True)

    flat = np.fromfile(args.trajectory_fixture, dtype="<f4")
    if flat.size != 256 * 64:
        raise ValueError(f"trajectory fixture has {flat.size} values, expected {256 * 64}")
    trajectory_hwc = torch.from_numpy(flat.copy().reshape(16, 16, 64))
    trajectory = trajectory_hwc.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device)
    run_case(model, "trajectory_256", trajectory, args.output, retain_boundaries=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
