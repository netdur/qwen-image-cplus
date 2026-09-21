#!/usr/bin/env python3
"""Capture Qwen3-VL text-only decoder boundaries for native validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers import __version__ as transformers_version

from calibrate_transformer_quantization import MODEL_SNAPSHOT


PROMPT = "A red fox asleep beside a mossy stone in soft morning light."
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/text_encoder_fp32"),
    )
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def write_tensor(directory: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    value = np.asarray(value, dtype="<f4")
    raw = value.tobytes()
    path = directory / f"{name}.f32"
    path.write_bytes(raw)
    return {
        "file": path.name,
        "dtype": "F32",
        "shape": list(value.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    if args.model.name != MODEL_SNAPSHOT:
        raise ValueError(f"expected pinned snapshot {MODEL_SNAPSHOT}")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model / "processor", local_files_only=True, padding_side="left"
    )
    template = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    system_message = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    system_tokens = tokenizer.apply_chat_template(
        system_message, tokenize=True, return_dict=False
    )
    if system_tokens and isinstance(system_tokens[0], list):
        system_tokens = system_tokens[0]
    drop_index = len(system_tokens)
    inputs = tokenizer(template.format(PROMPT), return_tensors="pt").to(device)

    started = time.perf_counter()
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model / "text_encoder",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    text_model = encoder.model.language_model
    captures: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            captures[name] = value[:, drop_index:].detach().cpu()

        return hook

    handles = [
        text_model.embed_tokens.register_forward_hook(capture("embedding")),
        text_model.layers[0].register_forward_hook(capture("layer_00")),
        text_model.layers[17].register_forward_hook(capture("layer_17")),
        text_model.layers[23].register_forward_hook(capture("layer_23")),
        text_model.layers[31].register_forward_hook(capture("layer_31")),
        text_model.layers[32].register_forward_hook(capture("layer_32")),
        text_model.layers[33].register_forward_hook(capture("layer_33")),
        text_model.layers[34].register_forward_hook(capture("layer_34")),
        text_model.layers[35].register_forward_hook(capture("layer_35")),
        text_model.norm.register_forward_hook(lambda _module, hook_args, _output: hook_args[0]),
    ]
    try:
        with torch.inference_mode():
            output = encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    captures["output"] = output.hidden_states[-1][:, drop_index:].detach().cpu()

    args.output.mkdir(parents=True, exist_ok=True)
    records = {
        name: write_tensor(args.output, name, value[0])
        for name, value in captures.items()
    }
    metadata = {
        "schema_version": 1,
        "model": "Qwen/Qwen-Image-2.1",
        "snapshot": MODEL_SNAPSHOT,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "device": str(device),
        },
        "prompt": PROMPT,
        "full_tokens": int(inputs.input_ids.shape[1]),
        "system_prefix_tokens_to_drop": drop_index,
        "output_tokens": int(inputs.input_ids.shape[1]) - drop_index,
        "capture_semantics": "decoder-layer outputs and final output before final RMSNorm",
        "elapsed_seconds": time.perf_counter() - started,
        "tensors": records,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

    del output, encoder, text_model, inputs
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
