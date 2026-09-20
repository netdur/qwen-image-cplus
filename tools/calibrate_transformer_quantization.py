#!/usr/bin/env python3
"""Calibrate the block-matrix Q8 policy on the real Qwen-Image-2.1 transformer.

This is an offline development tool, not part of the C+ runtime.  It uses the
pinned Diffusers implementation and local model snapshot to compare the BF16
reference transformer with the packed runtime's affine-INT8/group-64 policy.
Quantized weights and activations are rounded to FP16 before each target
linear, matching the Metal kernel's storage and input path; the surrounding
reference model remains BF16. PyTorch's MPS matmul is not claimed to be a
bit-exact emulation of the Metal kernel's per-product rounding.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn


MODEL_SNAPSHOT = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
DIFFUSERS_COMMIT = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"
GROUP_SIZE = 64
SAMPLED_BLOCKS = (0, 7, 15, 23, 31)
SAMPLED_ROWS = 32
NUM_INFERENCE_STEPS = 40
FINAL_NRMSE_LIMIT = 0.01

PROMPTS = (
    "A red fox asleep beside a mossy stone in soft morning light.",
    "A vintage travel poster for CASABLANCA reading 'MEET ME AT SUNSET', bold geometric lettering.",
    (
        "An isometric cutaway of a crowded orbital greenhouse: citrus trees on the upper deck, "
        "a blue maintenance robot beside the central stairs, three gardeners at the lower-left "
        "workbench, warm sunlight entering from windows on the right, precise architectural detail."
    ),
)


@dataclass(frozen=True)
class Case:
    name: str
    prompt_index: int
    pixels: int
    timestep_index: int
    seed: int


CASES = (
    Case("short-256-early", 0, 256, 0, 1101),
    Case("text-256-late", 1, 256, 39, 1102),
    Case("spatial-512-early", 2, 512, 0, 1201),
    Case("short-512-middle", 0, 512, 20, 1202),
    Case("text-1024-middle", 1, 1024, 20, 1301),
    Case("spatial-1024-late", 2, 1024, 39, 1302),
)

LINEAR_ROLES = {
    "q": ("attn", "to_q"),
    "k": ("attn", "to_k"),
    "v": ("attn", "to_v"),
    "attention_output": ("attn", "to_out", 0),
    "mlp_gate": ("img_mlp", "gate_layer"),
    "mlp_projection": ("img_mlp", "proj"),
    "mlp_output": ("img_mlp", "out"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local Qwen-Image-2.1 snapshot")
    parser.add_argument("output", type=Path, help="JSON report to write")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--quick", action="store_true", help="run the first 256px case only")
    return parser.parse_args()


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.detach().to(device="cpu", dtype=torch.float64)
    expected = expected.detach().to(device="cpu", dtype=torch.float64)
    difference = actual - expected
    rms = float(difference.square().mean().sqrt())
    reference_rms = float(expected.square().mean().sqrt())
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rms": rms,
        "reference_rms": reference_rms,
        "normalized_rms": rms / max(reference_rms, 1e-12),
    }


def get_child(root: nn.Module, path: tuple[str | int, ...]) -> nn.Module:
    value = root
    for part in path:
        value = value[part] if isinstance(part, int) else getattr(value, part)
    return value


def set_child(root: nn.Module, path: tuple[str | int, ...], value: nn.Module) -> None:
    parent = get_child(root, path[:-1]) if len(path) > 1 else root
    final = path[-1]
    if isinstance(final, int):
        parent[final] = value
    else:
        setattr(parent, final, value)


class AffineInt8LinearEmulation(nn.Module):
    """Dense FP16 emulation of one packed affine-Q8 Metal linear.

    Keeping the dequantized values avoids benchmarking thousands of tiny
    dequantization dispatches in Python.  The values themselves are produced
    from U8 weights, FP16 scales, and U8 zero points exactly as the packed
    format specifies.
    """

    def __init__(self, weight: torch.Tensor, chunk_rows: int = 128):
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE:
            raise ValueError(f"unsupported Q8 weight shape {tuple(weight.shape)}")
        output_width, input_width = weight.shape
        group_count = input_width // GROUP_SIZE
        dequantized = torch.empty_like(weight, dtype=torch.float16)
        with torch.no_grad():
            for start in range(0, output_width, chunk_rows):
                end = min(start + chunk_rows, output_width)
                grouped = weight[start:end].float().reshape(end - start, group_count, GROUP_SIZE)
                minimum = grouped.amin(dim=-1)
                maximum = grouped.amax(dim=-1)
                scale = (maximum - minimum) / 255.0
                scale = torch.where(scale == 0, torch.ones_like(scale), scale).to(torch.float16)
                scale_f32 = scale.float()
                zero = torch.round(-minimum / scale_f32).clamp_(0, 255).to(torch.uint8)
                quantized = torch.round(grouped / scale_f32.unsqueeze(-1))
                quantized.add_(zero.float().unsqueeze(-1)).clamp_(0, 255)
                restored = (quantized - zero.float().unsqueeze(-1)) * scale_f32.unsqueeze(-1)
                dequantized[start:end].copy_(restored.reshape(end - start, input_width))
        self.register_buffer("weight", dequantized)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional.linear(inputs.to(torch.float16), self.weight).to(inputs.dtype)


def encode_prompts(model_root: Path, device: torch.device) -> list[torch.Tensor]:
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    processor_root = model_root / "processor"
    tokenizer = AutoTokenizer.from_pretrained(
        processor_root, local_files_only=True, padding_side="left"
    )
    system_prompt = "Comprehend and analyze the provided prompt."
    template = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    system_message = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    system_tokens = tokenizer.apply_chat_template(system_message, tokenize=True, return_dict=False)
    drop_index = len(system_tokens[0]) if system_tokens and isinstance(system_tokens[0], list) else len(system_tokens)
    tokenized = tokenizer(
        [template.format(prompt) for prompt in PROMPTS],
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).to(device)

    started = time.perf_counter()
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_root / "text_encoder",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    text_model = getattr(encoder.model, "language_model", encoder.model)
    handle = text_model.norm.register_forward_hook(lambda _module, args, _output: args[0])
    try:
        with torch.inference_mode():
            output = encoder(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                output_hidden_states=True,
            )
    finally:
        handle.remove()

    hidden = output.hidden_states[-1]
    result = [
        sample[mask.bool()][drop_index:].detach().cpu()
        for sample, mask in zip(hidden, tokenized.attention_mask)
    ]
    print(
        f"encoded {len(result)} prompts in {time.perf_counter() - started:.2f}s: "
        + ", ".join(str(value.shape[0]) for value in result)
        + " tokens",
        flush=True,
    )
    del output, hidden, encoder, text_model, tokenized
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def schedule_timestep(model_root: Path, image_tokens: int, index: int, device: torch.device) -> float:
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_root / "scheduler", local_files_only=True
    )
    config = scheduler.config
    mu = calculate_shift(
        image_tokens,
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    sigmas = np.linspace(1.0, 1 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sigmas, mu=mu)
    return float(timesteps[index].cpu())


def make_inputs(
    case: Case,
    prompt: torch.Tensor,
    timestep: float,
    device: torch.device,
) -> dict:
    side = case.pixels // 16
    image_tokens = side * side
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    latents = torch.randn((1, image_tokens, 64), generator=generator, dtype=torch.float32)
    image_mask = torch.cat(
        [
            torch.zeros((1, prompt.shape[0]), dtype=torch.bool),
            torch.ones((1, image_tokens // 4), dtype=torch.bool),
        ],
        dim=1,
    )
    return {
        "hidden_states": latents.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": prompt.unsqueeze(0).to(device),
        "timestep": torch.tensor([timestep / 1000.0], device=device, dtype=torch.bfloat16),
        "img_shapes": [[(1, side, side)]],
        "img_mask": image_mask.to(device),
        "return_dict": False,
    }


def sampled_rows(value: torch.Tensor) -> torch.Tensor:
    count = min(SAMPLED_ROWS, value.shape[1])
    indices = torch.linspace(0, value.shape[1] - 1, count, device=value.device).round().long()
    return value[:, indices].detach().cpu()


def run_case(model: nn.Module, inputs: dict) -> tuple[torch.Tensor, dict[int, torch.Tensor], float]:
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for block_index in SAMPLED_BLOCKS:
        handles.append(
            model.transformer_blocks[block_index].register_forward_hook(
                lambda _module, _args, output, index=block_index: captures.__setitem__(index, sampled_rows(output))
            )
        )
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model(**inputs)[0]
        if inputs["hidden_states"].device.type == "mps":
            torch.mps.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - started
    image_tokens = inputs["hidden_states"].shape[1]
    return output[:, -image_tokens:].detach().cpu(), captures, elapsed


def quantize_transformer_blocks(model: nn.Module) -> dict[str, int | float]:
    source_bytes = 0
    packed_bytes = 0
    matrix_count = 0
    started = time.perf_counter()
    for block_index, block in enumerate(model.transformer_blocks):
        for role, path in LINEAR_ROLES.items():
            dense = get_child(block, path)
            if not isinstance(dense, nn.Linear) or dense.bias is not None:
                raise TypeError(f"unexpected {block_index}:{role} module {dense!r}")
            elements = dense.weight.numel()
            groups = dense.weight.shape[0] * (dense.weight.shape[1] // GROUP_SIZE)
            replacement = AffineInt8LinearEmulation(dense.weight)
            set_child(block, path, replacement)
            source_bytes += elements * 2
            packed_bytes += elements + groups * 3
            matrix_count += 1
        print(f"quantized block {block_index + 1}/32", flush=True)
    gc.collect()
    if next(model.parameters()).device.type == "mps":
        torch.mps.empty_cache()
    return {
        "matrix_count": matrix_count,
        "source_bf16_bytes": source_bytes,
        "packed_bytes": packed_bytes,
        "compression_vs_bf16": source_bytes / packed_bytes,
        "elapsed_seconds": time.perf_counter() - started,
    }


def validate_model_root(model_root: Path) -> None:
    manifest = model_root / "model_index.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing model manifest: {manifest}")
    resolved = model_root.resolve()
    if MODEL_SNAPSHOT not in str(resolved):
        print(
            f"warning: path does not contain pinned snapshot {MODEL_SNAPSHOT}; "
            "the report still records the requested path",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    validate_model_root(args.model)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    cases = CASES[:1] if args.quick else CASES
    prompts = encode_prompts(args.model, device)
    case_timesteps = {
        case.name: schedule_timestep(
            args.model, (case.pixels // 16) ** 2, case.timestep_index, device
        )
        for case in cases
    }

    from diffusers import QwenImage21Transformer2DModel, __version__ as diffusers_version
    from transformers import __version__ as transformers_version

    started = time.perf_counter()
    model = QwenImage21Transformer2DModel.from_pretrained(
        args.model / "transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    print(f"loaded transformer in {time.perf_counter() - started:.2f}s", flush=True)

    references = {}
    case_inputs = {}
    for case in cases:
        inputs = make_inputs(case, prompts[case.prompt_index], case_timesteps[case.name], device)
        final, blocks, elapsed = run_case(model, inputs)
        references[case.name] = (final, blocks, elapsed)
        case_inputs[case.name] = inputs
        print(f"reference {case.name}: {elapsed:.2f}s", flush=True)

    quantization = quantize_transformer_blocks(model)
    print(
        f"quantized {quantization['matrix_count']} matrices in "
        f"{quantization['elapsed_seconds']:.2f}s",
        flush=True,
    )

    results = []
    final_errors = []
    for case in cases:
        actual, actual_blocks, elapsed = run_case(model, case_inputs[case.name])
        expected, expected_blocks, reference_elapsed = references[case.name]
        final_metrics = error_metrics(actual, expected)
        final_errors.append(final_metrics["normalized_rms"])
        block_metrics = {
            str(index): error_metrics(actual_blocks[index], expected_blocks[index])
            for index in SAMPLED_BLOCKS
        }
        results.append(
            {
                "name": case.name,
                "prompt_index": case.prompt_index,
                "pixels": [case.pixels, case.pixels],
                "image_tokens": (case.pixels // 16) ** 2,
                "timestep_index": case.timestep_index,
                "scheduler_timestep": case_timesteps[case.name],
                "latent_seed": case.seed,
                "prompt_tokens": prompts[case.prompt_index].shape[0],
                "reference_seconds": reference_elapsed,
                "quantized_seconds": elapsed,
                "sampled_blocks": block_metrics,
                "final_noise_prediction": final_metrics,
            }
        )
        print(
            f"Q8 {case.name}: nRMSE={final_metrics['normalized_rms']:.6%}, {elapsed:.2f}s",
            flush=True,
        )

    worst_error = max(final_errors)
    report = {
        "schema_version": 1,
        "model": "Qwen/Qwen-Image-2.1",
        "snapshot": MODEL_SNAPSHOT,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "environment": {
            "torch": torch.__version__,
            "diffusers": diffusers_version,
            "transformers": transformers_version,
            "device": str(device),
        },
        "method": {
            "prompt_encoding": "official raw T2I template; last decoder layer before final RMSNorm",
            "latents": "deterministic Gaussian BF16 inputs",
            "scheduler_steps": NUM_INFERENCE_STEPS,
            "sampled_blocks": list(SAMPLED_BLOCKS),
            "sampled_rows_per_block": SAMPLED_ROWS,
            "quantization": {
                "scheme": "affine_int8",
                "group_axis": 1,
                "group_size": GROUP_SIZE,
                "scale_dtype": "F16",
                "zero_point_dtype": "U8",
                "linear_input_dtype": "F16",
                "linear_weight_dtype": "F16 dequantized from packed representation",
                "linear_arithmetic": "PyTorch MPS FP16 matmul; not bit-exact Metal product rounding",
                "scope": "seven matrix roles in all 32 transformer blocks",
            },
        },
        "prompts": list(PROMPTS),
        "quantization_storage": quantization,
        "cases": results,
        "acceptance": {
            "metric": "worst per-case final-noise normalized RMS",
            "limit": FINAL_NRMSE_LIMIT,
            "measured": worst_error,
            "passed": worst_error <= FINAL_NRMSE_LIMIT,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {args.output}; worst final nRMSE={worst_error:.6%}; "
        f"gate={'PASS' if worst_error <= FINAL_NRMSE_LIMIT else 'FAIL'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
