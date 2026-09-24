#!/usr/bin/env python3
"""Generate a small independent Qwen3-VL vision-tower oracle.

This deliberately uses only PyTorch and safetensors instead of importing
Transformers.  The equations follow the model-pinned Transformers 4.57.1
Qwen3-VL implementation, and the four-patch 32x32 input keeps the CPU oracle
cheap enough to regenerate during development.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


WIDTH = 1152
INTERMEDIATE_WIDTH = 4304
OUTPUT_WIDTH = 4096
HEADS = 16
HEAD_DIM = 72
PATCH = 16
TEMPORAL_PATCH = 2
MERGE = 2
LAYERS = 27


def save_f32(path: Path, value: torch.Tensor) -> None:
    value.detach().float().contiguous().numpy().tofile(path)


def deterministic_rgb() -> torch.Tensor:
    y = torch.arange(32, dtype=torch.float32)[:, None]
    x = torch.arange(32, dtype=torch.float32)[None, :]
    red = (x + 3 * y).remainder(256) / 255.0
    green = (5 * x + 7 * y + 19).remainder(256) / 255.0
    blue = (11 * x + 13 * y + 37).remainder(256) / 255.0
    return torch.stack((red.expand(32, 32), green, blue), dim=-1)


def patchify(rgb: torch.Tensor) -> torch.Tensor:
    # Official order after Qwen2VLImageProcessorFast's view/permute:
    # grid_t, grid_h/2, grid_w/2, merge_h, merge_w, C, T, patch_h, patch_w.
    pixels = (rgb.permute(2, 0, 1) * 2.0 - 1.0)[None, None]
    pixels = torch.cat((pixels, pixels), dim=1)
    patches = pixels.view(1, 1, TEMPORAL_PATCH, 3, 1, MERGE, PATCH, 1, MERGE, PATCH)
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return patches.reshape(4, 3 * TEMPORAL_PATCH * PATCH * PATCH)


def positions() -> torch.Tensor:
    return torch.tensor(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=torch.long)


def position_embedding(tensor, pos: torch.Tensor) -> torch.Tensor:
    # For a 2x2 grid, linspace(0, 47, 2) lands exactly at the four corners.
    indexes = pos[:, 0] * 47 * 48 + pos[:, 1] * 47
    return tensor.get_tensor("model.visual.pos_embed.weight")[indexes]


def rotary_embedding(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 36, 2, dtype=torch.float32) / 36.0))
    table = torch.outer(torch.arange(2, dtype=torch.float32), inv_freq)
    frequencies = table[pos].flatten(1)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos(), embedding.sin()


def linear(x: torch.Tensor, tensor, prefix: str) -> torch.Tensor:
    return F.linear(
        x,
        tensor.get_tensor(prefix + ".weight"),
        tensor.get_tensor(prefix + ".bias"),
    )


def norm(x: torch.Tensor, tensor, prefix: str) -> torch.Tensor:
    return F.layer_norm(
        x,
        (x.shape[-1],),
        tensor.get_tensor(prefix + ".weight"),
        tensor.get_tensor(prefix + ".bias"),
        1e-6,
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def merger(x: torch.Tensor, tensor, prefix: str, postshuffle: bool) -> torch.Tensor:
    if postshuffle:
        x = norm(x.view(-1, WIDTH * 4), tensor, prefix + ".norm")
    else:
        x = norm(x, tensor, prefix + ".norm").view(-1, WIDTH * 4)
    x = linear(x, tensor, prefix + ".linear_fc1")
    x = F.gelu(x, approximate="tanh")
    return linear(x, tensor, prefix + ".linear_fc2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("tests/fixtures/vision_encoder_fp32")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shard = args.model_root / "text_encoder" / "model-00001-of-00004.safetensors"

    torch.set_grad_enabled(False)
    rgb = deterministic_rgb()
    patch_values = patchify(rgb)
    pos = positions()
    cos, sin = rotary_embedding(pos)
    save_f32(args.output / "input_rgb.f32", rgb)

    with safe_open(shard, framework="pt", device="cpu") as tensor:
        patch_weight = tensor.get_tensor("model.visual.patch_embed.proj.weight").view(WIDTH, -1)
        hidden = F.linear(
            patch_values.to(patch_weight.dtype),
            patch_weight,
            tensor.get_tensor("model.visual.patch_embed.proj.bias"),
        )
        hidden = hidden + position_embedding(tensor, pos)
        save_f32(args.output / "patch_embed.f32", hidden)

        deepstack = []
        for layer in range(LAYERS):
            prefix = f"model.visual.blocks.{layer}"
            residual = hidden
            normalized = norm(hidden, tensor, prefix + ".norm1")
            qkv = linear(normalized, tensor, prefix + ".attn.qkv")
            query, key, value = qkv.reshape(4, 3, HEADS, HEAD_DIM).permute(1, 0, 2, 3)
            query_float = query.float()
            key_float = key.float()
            query = (query_float * cos[:, None] + rotate_half(query_float) * sin[:, None]).to(query.dtype)
            key = (key_float * cos[:, None] + rotate_half(key_float) * sin[:, None]).to(key.dtype)
            query = query.transpose(0, 1).unsqueeze(0)
            key = key.transpose(0, 1).unsqueeze(0)
            value = value.transpose(0, 1).unsqueeze(0)
            scores = torch.matmul(query, key.transpose(2, 3)) * (HEAD_DIM ** -0.5)
            probability = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attended = torch.matmul(probability, value).transpose(1, 2).reshape(4, WIDTH)
            hidden = residual + linear(attended, tensor, prefix + ".attn.proj")
            residual = hidden
            mlp = linear(norm(hidden, tensor, prefix + ".norm2"), tensor, prefix + ".mlp.linear_fc1")
            mlp = F.gelu(mlp, approximate="tanh")
            hidden = residual + linear(mlp, tensor, prefix + ".mlp.linear_fc2")
            save_f32(args.output / f"block_{layer:02d}.f32", hidden)

            if layer in (8, 16, 24):
                deep_index = (8, 16, 24).index(layer)
                merged = merger(
                    hidden,
                    tensor,
                    f"model.visual.deepstack_merger_list.{deep_index}",
                    True,
                )
                deepstack.append(merged)
                save_f32(args.output / f"deepstack_{deep_index}.f32", merged)

        output = merger(hidden, tensor, "model.visual.merger", False)
        save_f32(args.output / "output.f32", output)

    print(
        f"vision fixture: input=32x32 patches=4 tokens=1 "
        f"output_mean_abs={output.float().abs().mean().item():.9f} directory={args.output}"
    )


if __name__ == "__main__":
    main()
