#!/usr/bin/env python3
"""Measure affine-Q8/group-64 error through the complete Qwen3-VL text stack.

This is an offline development tool. It replaces selected decoder-layer linear
weights with the exact U8 + FP16-scale + U8-zero representation proposed for
QIPACK, while leaving embeddings and norm vectors in BF16. Each forward pass
dequantizes a bounded output-row chunk to FP32 and uses FP32 inputs and
accumulation, matching the viable custom-Metal path without retaining a dense
dequantized copy.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers import __version__ as transformers_version

from calibrate_transformer_quantization import MODEL_SNAPSHOT, error_metrics


GROUP_SIZE = 64
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPTS = (
    "A red fox asleep beside a mossy stone in soft morning light.",
    "A blue ceramic teapot on a windowsill.",
    "An isometric orbital greenhouse with citrus trees, gardeners, and a blue maintenance robot.",
)
LINEAR_PATHS = {
    "q": ("self_attn", "q_proj"),
    "k": ("self_attn", "k_proj"),
    "v": ("self_attn", "v_proj"),
    "o": ("self_attn", "o_proj"),
    "gate": ("mlp", "gate_proj"),
    "up": ("mlp", "up_proj"),
    "down": ("mlp", "down_proj"),
}


class AffineQ8Linear(nn.Module):
    """Bounded-memory FP32 emulation of affine U8/group-64 execution."""

    def __init__(self, source: nn.Linear, chunk_rows: int = 1024):
        super().__init__()
        weight = source.weight.detach()
        if weight.ndim != 2 or weight.shape[1] % GROUP_SIZE:
            raise ValueError(f"unsupported Q8 weight shape {tuple(weight.shape)}")
        output_width, input_width = weight.shape
        group_count = input_width // GROUP_SIZE
        quantized = torch.empty_like(weight, dtype=torch.uint8).reshape(
            output_width, group_count, GROUP_SIZE
        )
        scales = torch.empty(
            (output_width, group_count), dtype=torch.float16, device=weight.device
        )
        zeros = torch.empty(
            (output_width, group_count), dtype=torch.uint8, device=weight.device
        )
        with torch.no_grad():
            for start in range(0, output_width, chunk_rows):
                end = min(start + chunk_rows, output_width)
                grouped = weight[start:end].float().reshape(
                    end - start, group_count, GROUP_SIZE
                )
                minimum = grouped.amin(dim=-1)
                maximum = grouped.amax(dim=-1)
                scale = ((maximum - minimum) / 255.0).to(torch.float16)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                scale_f32 = scale.float()
                zero = torch.round(-minimum / scale_f32).clamp_(0, 255).to(torch.uint8)
                values = torch.round(grouped / scale_f32.unsqueeze(-1))
                values.add_(zero.float().unsqueeze(-1)).clamp_(0, 255)
                quantized[start:end].copy_(values.to(torch.uint8))
                scales[start:end].copy_(scale)
                zeros[start:end].copy_(zero)
        self.output_width = output_width
        self.input_width = input_width
        self.chunk_rows = chunk_rows
        self.register_buffer("quantized", quantized)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        if source.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", source.bias.detach().float())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        source_dtype = inputs.dtype
        inputs_f32 = inputs.float()
        chunks = []
        for start in range(0, self.output_width, self.chunk_rows):
            end = min(start + self.chunk_rows, self.output_width)
            weight = (
                self.quantized[start:end].float()
                - self.zeros[start:end].float().unsqueeze(-1)
            ) * self.scales[start:end].float().unsqueeze(-1)
            bias = None if self.bias is None else self.bias[start:end]
            chunks.append(
                functional.linear(
                    inputs_f32,
                    weight.reshape(end - start, self.input_width),
                    bias,
                )
            )
        return torch.cat(chunks, dim=-1).to(source_dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--scan",
        default="36",
        help="comma-separated cumulative counts of early decoder layers to quantize",
    )
    parser.add_argument(
        "--roles",
        default=",".join(LINEAR_PATHS),
        help="comma-separated linear roles: q,k,v,o,gate,up,down",
    )
    parser.add_argument(
        "--export-first-embedding",
        type=Path,
        help="write the final candidate's first prompt embedding as little-endian F32",
    )
    return parser.parse_args()


def tokenize(tokenizer: AutoTokenizer, prompt: str, device: torch.device):
    template = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return tokenizer(template.format(prompt), return_tensors="pt").to(device)


def run_prompts(
    encoder: Qwen3VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    prompts: tuple[str, ...],
    device: torch.device,
) -> list[torch.Tensor]:
    text_model = encoder.model.language_model
    # Native Qwen-Image consumes the last decoder-layer output before the
    # language model's final RMSNorm. A forward hook may replace module output,
    # so return the norm input exactly as the fixture generator does.
    handle = text_model.norm.register_forward_hook(
        lambda _module, hook_args, _output: hook_args[0]
    )
    outputs: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for prompt in prompts:
                inputs = tokenize(tokenizer, prompt, device)
                result = encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )
                outputs.append(result.hidden_states[-1][0, 14:].float().cpu())
                del result, inputs
    finally:
        handle.remove()
    return outputs


def replace_linears(
    text_model: nn.Module,
    device: torch.device,
    start_layer: int,
    end_layer: int,
    roles: tuple[str, ...],
) -> dict[str, object]:
    started = time.perf_counter()
    matrix_count = 0
    parameter_count = 0
    packed_bytes = 0
    for layer_index in range(start_layer, end_layer):
        layer = text_model.layers[layer_index]
        for role in roles:
            parent_name, child_name = LINEAR_PATHS[role]
            parent = getattr(layer, parent_name)
            source = getattr(parent, child_name)
            if not isinstance(source, nn.Linear):
                raise TypeError(
                    f"layer {layer_index} {parent_name}.{child_name} is {type(source)}"
                )
            output_width, input_width = source.weight.shape
            groups = output_width * (input_width // GROUP_SIZE)
            parameter_count += source.weight.numel()
            packed_bytes += source.weight.numel() + groups * 3
            setattr(parent, child_name, AffineQ8Linear(source))
            matrix_count += 1
            del source
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        print(
            f"quantized text layer {layer_index + 1}/{len(text_model.layers)}",
            flush=True,
        )
    return {
        "matrix_count": matrix_count,
        "parameter_count": parameter_count,
        "packed_bytes": packed_bytes,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if args.model.name != MODEL_SNAPSHOT:
        raise ValueError(f"expected pinned snapshot {MODEL_SNAPSHOT}")
    device = torch.device(args.device)
    prompts = PROMPTS[:1] if args.quick else PROMPTS
    scan = sorted({int(value) for value in args.scan.split(",")})
    if not scan or scan[0] < 1 or scan[-1] > 36:
        raise ValueError("--scan values must be cumulative layer counts from 1 through 36")
    roles = tuple(value.strip() for value in args.roles.split(",") if value.strip())
    unknown_roles = sorted(set(roles) - set(LINEAR_PATHS))
    if not roles or unknown_roles:
        raise ValueError(f"unknown or empty --roles selection: {unknown_roles}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model / "processor", local_files_only=True, padding_side="left"
    )
    load_started = time.perf_counter()
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model / "text_encoder",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    load_seconds = time.perf_counter() - load_started

    reference_started = time.perf_counter()
    reference = run_prompts(encoder, tokenizer, prompts, device)
    reference_seconds = time.perf_counter() - reference_started
    candidates = []
    previous_end = 0
    cumulative_quantization_seconds = 0.0
    cumulative_parameter_count = 0
    cumulative_packed_bytes = 0
    cumulative_matrix_count = 0
    total_linear_parameters = 6_945_767_424
    for end_layer in scan:
        quantization = replace_linears(
            encoder.model.language_model, device, previous_end, end_layer, roles
        )
        cumulative_quantization_seconds += quantization["elapsed_seconds"]
        cumulative_parameter_count += quantization["parameter_count"]
        cumulative_packed_bytes += quantization["packed_bytes"]
        cumulative_matrix_count += quantization["matrix_count"]
        quantized_started = time.perf_counter()
        actual = run_prompts(encoder, tokenizer, prompts, device)
        quantized_seconds = time.perf_counter() - quantized_started
        cases = []
        for prompt, candidate, expected in zip(prompts, actual, reference):
            metrics = error_metrics(candidate, expected)
            cases.append(
                {
                    "prompt": prompt,
                    "tokens": candidate.shape[0],
                    **metrics,
                    "passes_text_gate": metrics["normalized_rms"] <= 0.04,
                    "passes_one_percent_calibration_target": metrics["normalized_rms"]
                    <= 0.01,
                }
            )
            print(
                f"text Q8 first_layers={end_layer} tokens={candidate.shape[0]} "
                f"nRMSE={metrics['normalized_rms']:.6%}",
                flush=True,
            )
        retained_parameters = total_linear_parameters - cumulative_parameter_count
        projected_linear_bytes = cumulative_packed_bytes + retained_parameters * 2
        candidates.append(
            {
                "first_quantized_layer_count": end_layer,
                "quantized_matrix_count": cumulative_matrix_count,
                "quantized_parameter_count": cumulative_parameter_count,
                "retained_bf16_parameter_count": retained_parameters,
                "projected_linear_bytes": projected_linear_bytes,
                "linear_storage_reduction_percent": 100.0
                * (1.0 - projected_linear_bytes / (total_linear_parameters * 2)),
                "cumulative_quantization_seconds": cumulative_quantization_seconds,
                "inference_seconds": quantized_seconds,
                "cases": cases,
                "all_pass_text_gate": all(case["passes_text_gate"] for case in cases),
                "all_pass_one_percent_calibration_target": all(
                    case["passes_one_percent_calibration_target"] for case in cases
                ),
            }
        )
        previous_end = end_layer

    report = {
        "schema_version": 1,
        "date": "2026-09-21",
        "model": "Qwen/Qwen-Image-2.1",
        "snapshot": MODEL_SNAPSHOT,
        "machine": "Apple M1 Max, 32 GB unified memory",
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "device": str(device),
        },
        "policy": {
            "scheme": "affine_int8",
            "group_axis": "input/K",
            "group_size": GROUP_SIZE,
            "scale": "FP16 per output-row group",
            "zero": "U8 per output-row group",
            "quantized": "cumulative early decoder layers selected by --scan",
            "roles": list(roles),
            "retained_bf16": "token embeddings and all norm vectors",
            "execution_emulation": "packed values dequantized by output-row chunks with FP32 inputs and accumulation, cast back to the surrounding BF16 dtype",
        },
        "timing_seconds": {
            "load": load_seconds,
            "reference": reference_seconds,
            "quantized": quantized_seconds,
        },
        "candidates": candidates,
        "limitations": [
            "PyTorch MPS FP32 matmul is not bit-exact with the future native packed Metal kernel.",
            "The downstream 256px latent and decoded-image gates must pass before this becomes a runtime policy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.export_first_embedding is not None:
        value = actual[0].contiguous().numpy().astype("<f4", copy=False)
        args.export_first_embedding.parent.mkdir(parents=True, exist_ok=True)
        args.export_first_embedding.write_bytes(value.tobytes())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
