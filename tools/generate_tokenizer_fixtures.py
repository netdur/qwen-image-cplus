#!/usr/bin/env python3
"""Generate exact text-only Qwen3-VL tokenizer fixtures.

This development-only tool deliberately exercises the tokenizer from the
pinned model snapshot.  The native runtime consumes only the resulting JSON
metadata and little-endian token-ID arrays; it never embeds Python or
Transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, __version__ as transformers_version

from calibrate_transformer_quantization import MODEL_SNAPSHOT


SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPTS = (
    ("empty", ""),
    ("canonical_fox", "A red fox asleep beside a mossy stone in soft morning light."),
    ("unicode_multiline", "A café's sign says 你好 — déjà vu.\nSecond line!"),
    ("whitespace", " leading  spaces\tand trailing "),
    ("special_tokens", "Show <|im_end|> and <think> literally."),
    ("nfc_decomposed", "Cafe\u0301"),
)
MULTI_IMAGE_PROMPT = "Place Picture 1 beside Picture 2."
MULTI_IMAGE_COUNTS = (3, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local Qwen-Image-2.1 snapshot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tokenizer"),
        help="fixture directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model.name != MODEL_SNAPSHOT:
        raise ValueError(f"expected pinned snapshot {MODEL_SNAPSHOT}, got {args.model.name}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model / "processor",
        local_files_only=True,
        padding_side="left",
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
        system_message,
        tokenize=True,
        return_dict=False,
    )
    if system_tokens and isinstance(system_tokens[0], list):
        system_tokens = system_tokens[0]
    drop_index = len(system_tokens)

    args.output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    full_sequences: list[list[int]] = []
    for name, prompt in PROMPTS:
        full_ids = tokenizer(template.format(prompt), add_special_tokens=True).input_ids
        output_ids = full_ids[drop_index:]
        values = np.asarray(output_ids, dtype="<i4")
        path = args.output / f"{name}.i32"
        path.write_bytes(values.tobytes())
        cases.append(
            {
                "name": name,
                "prompt": prompt,
                "full_token_count": len(full_ids),
                "output_token_count": len(output_ids),
                "file": path.name,
                "bytes": values.nbytes,
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "output_ids": output_ids,
            }
        )
        full_sequences.append(full_ids)

    maximum = max(map(len, full_sequences))
    pad_id = int(tokenizer.pad_token_id)
    left_padding = [
        [pad_id] * (maximum - len(sequence)) + sequence for sequence in full_sequences
    ]
    attention_mask = [
        [0] * (maximum - len(sequence)) + [1] * len(sequence) for sequence in full_sequences
    ]
    metadata = {
        "schema_version": 1,
        "model": "Qwen/Qwen-Image-2.1",
        "snapshot": MODEL_SNAPSHOT,
        "transformers": transformers_version,
        "tokenizer_class": tokenizer.__class__.__name__,
        "normalization": "NFC",
        "padding_side": tokenizer.padding_side,
        "pad_token_id": pad_id,
        "system_prompt": SYSTEM_PROMPT,
        "system_prefix_tokens_to_drop": drop_index,
        "system_tokens": system_tokens,
        "template": template,
        "cases": cases,
        "batch_left_padding": {
            "sequence_length": maximum,
            "input_ids": left_padding,
            "attention_mask": attention_mask,
        },
    }
    # Image-conditioned template, built the way QwenImage21Pipeline builds it
    # and with each `<|image_pad|>` expanded as the processor expands it. The
    # native test supplies the same per-image counts, so full IDs compare.
    image_labels = "".join(
        ("" if index == 0 else " ")
        + f"<image{index + 1}><|vision_start|>"
        + "<|image_pad|>" * count
        + "<|vision_end|>"
        for index, count in enumerate(MULTI_IMAGE_COUNTS)
    )
    multi_image_ids = tokenizer(
        template.format(image_labels + MULTI_IMAGE_PROMPT), add_special_tokens=True
    ).input_ids
    multi_image_values = np.asarray(multi_image_ids, dtype="<i4")
    (args.output / "multi_image_three.i32").write_bytes(multi_image_values.tobytes())
    metadata["multi_image"] = {
        "file": "multi_image_three.i32",
        "prompt": MULTI_IMAGE_PROMPT,
        "image_token_counts": list(MULTI_IMAGE_COUNTS),
        "full_token_count": len(multi_image_ids),
        "sha256": hashlib.sha256(multi_image_values.tobytes()).hexdigest(),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"wrote {len(cases)} tokenizer cases; drop={drop_index}; "
        + ", ".join(f"{case['name']}={case['output_token_count']}" for case in cases)
    )


if __name__ == "__main__":
    main()
