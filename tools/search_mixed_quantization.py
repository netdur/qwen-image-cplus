#!/usr/bin/env python3
"""Search block-range and matrix-role mixed Q8 policies for Qwen-Image-2.1."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import calibrate_transformer_quantization as calibration


ALL_BLOCKS = tuple(range(32))
ALL_ROLES = tuple(calibration.LINEAR_ROLES)
ATTENTION_ROLES = ("q", "k", "v", "attention_output")
MLP_ROLES = ("mlp_gate", "mlp_projection", "mlp_output")
SCREEN_CASES = ("spatial-512-early", "text-256-late", "spatial-1024-late")


@dataclass(frozen=True)
class Policy:
    name: str
    blocks: tuple[int, ...]
    roles: tuple[str, ...]
    extra_segments: tuple[tuple[tuple[int, ...], tuple[str, ...]], ...] = ()

    def matrix_selection(self) -> frozenset[tuple[int, str]]:
        result = {(block, role) for block in self.blocks for role in self.roles}
        for blocks, roles in self.extra_segments:
            result.update((block, role) for block in blocks for role in roles)
        return frozenset(result)


POLICIES = (
    Policy("first_half_all_roles", tuple(range(0, 16)), ALL_ROLES),
    Policy("second_half_all_roles", tuple(range(16, 32)), ALL_ROLES),
    Policy("middle_half_all_roles", tuple(range(8, 24)), ALL_ROLES),
    Policy("outer_quarters_all_roles", (*range(0, 8), *range(24, 32)), ALL_ROLES),
    Policy("all_blocks_attention", ALL_BLOCKS, ATTENTION_ROLES),
    Policy("all_blocks_mlp", ALL_BLOCKS, MLP_ROLES),
    Policy("all_blocks_qkv", ALL_BLOCKS, ("q", "k", "v")),
    Policy(
        "all_blocks_except_attention_output",
        ALL_BLOCKS,
        tuple(role for role in ALL_ROLES if role != "attention_output"),
    ),
    Policy(
        "all_blocks_except_mlp_output",
        ALL_BLOCKS,
        tuple(role for role in ALL_ROLES if role != "mlp_output"),
    ),
    *(
        Policy(f"blocks_{start}_{start + 3}_all_roles", tuple(range(start, start + 4)), ALL_ROLES)
        for start in range(0, 32, 4)
    ),
    *(
        Policy(f"blocks_{start}_{start + 7}_all_roles", tuple(range(start, start + 8)), ALL_ROLES)
        for start in range(0, 32, 8)
    ),
    *(Policy(f"all_blocks_{role}", ALL_BLOCKS, (role,)) for role in ALL_ROLES),
    Policy("blocks_24_31_mlp", tuple(range(24, 32)), MLP_ROLES),
    Policy("blocks_20_31_mlp", tuple(range(20, 32)), MLP_ROLES),
    Policy("blocks_16_31_mlp", tuple(range(16, 32)), MLP_ROLES),
    Policy("blocks_8_31_mlp", tuple(range(8, 32)), MLP_ROLES),
    Policy("blocks_24_31_attention", tuple(range(24, 32)), ATTENTION_ROLES),
    Policy("blocks_16_31_attention", tuple(range(16, 32)), ATTENTION_ROLES),
    Policy(
        "blocks_24_31_attention_28_31_mlp",
        tuple(range(24, 32)),
        ATTENTION_ROLES,
        ((tuple(range(28, 32)), MLP_ROLES),),
    ),
    *(
        Policy(
            f"blocks_24_31_attention_28_31_{role}",
            tuple(range(24, 32)),
            ATTENTION_ROLES,
            ((tuple(range(28, 32)), (role,)),),
        )
        for role in MLP_ROLES
    ),
    *(
        Policy(
            f"blocks_24_31_attention_28_31_{name}",
            tuple(range(24, 32)),
            ATTENTION_ROLES,
            ((tuple(range(28, 32)), roles),),
        )
        for name, roles in (
            ("gate_projection", ("mlp_gate", "mlp_projection")),
            ("gate_output", ("mlp_gate", "mlp_output")),
            ("projection_output", ("mlp_projection", "mlp_output")),
        )
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--policy",
        action="append",
        help="evaluate only this named policy; repeat to select more than one",
    )
    return parser.parse_args()


def load_transformer(model_root: Path, device: torch.device):
    from diffusers import QwenImage21Transformer2DModel

    started = time.perf_counter()
    model = QwenImage21Transformer2DModel.from_pretrained(
        model_root / "transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    print(f"loaded transformer in {time.perf_counter() - started:.2f}s", flush=True)
    return model


def release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def compare_case(model, inputs: dict, reference: tuple) -> dict:
    actual, actual_blocks, elapsed = calibration.run_case(model, inputs)
    expected, expected_blocks, _ = reference
    final = calibration.error_metrics(actual, expected)
    blocks = {
        str(index): calibration.error_metrics(actual_blocks[index], expected_blocks[index])
        for index in calibration.SAMPLED_BLOCKS
    }
    return {
        "seconds": elapsed,
        "final_noise_prediction": final,
        "sampled_blocks": blocks,
    }


def main() -> int:
    args = parse_args()
    calibration.validate_model_root(args.model)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    from diffusers import __version__ as diffusers_version
    from transformers import __version__ as transformers_version

    started = time.perf_counter()
    policies = POLICIES
    if args.policy:
        requested = set(args.policy)
        known = {policy.name for policy in POLICIES}
        if requested - known:
            raise ValueError(f"unknown policies: {sorted(requested - known)}")
        policies = tuple(policy for policy in POLICIES if policy.name in requested)
    prompts = calibration.encode_prompts(args.model, device)
    timesteps = {
        case.name: calibration.schedule_timestep(
            args.model, (case.pixels // 16) ** 2, case.timestep_index, device
        )
        for case in calibration.CASES
    }
    inputs = {
        case.name: calibration.make_inputs(
            case, prompts[case.prompt_index], timesteps[case.name], device
        )
        for case in calibration.CASES
    }

    reference_model = load_transformer(args.model, device)
    references = {}
    for case in calibration.CASES:
        references[case.name] = calibration.run_case(reference_model, inputs[case.name])
        print(f"reference {case.name}: {references[case.name][2]:.2f}s", flush=True)
    del reference_model
    release_device_cache(device)

    policy_results = []
    for policy in policies:
        print(f"evaluating {policy.name}", flush=True)
        model = load_transformer(args.model, device)
        storage = calibration.quantize_transformer_blocks(
            model, matrix_selection=policy.matrix_selection()
        )
        cases = {
            name: compare_case(model, inputs[name], references[name])
            for name in SCREEN_CASES
        }
        screen_error = max(
            value["final_noise_prediction"]["normalized_rms"] for value in cases.values()
        )
        screened_in = screen_error <= calibration.FINAL_NRMSE_LIMIT
        if screened_in:
            for case in calibration.CASES:
                if case.name not in SCREEN_CASES:
                    cases[case.name] = compare_case(model, inputs[case.name], references[case.name])
        errors = [value["final_noise_prediction"]["normalized_rms"] for value in cases.values()]
        complete = len(cases) == len(calibration.CASES)
        passed = complete and max(errors) <= calibration.FINAL_NRMSE_LIMIT
        policy_results.append(
            {
                "name": policy.name,
                "quantized_blocks": list(policy.blocks),
                "quantized_roles": list(policy.roles),
                "extra_segments": [
                    {"blocks": list(blocks), "roles": list(roles)}
                    for blocks, roles in policy.extra_segments
                ],
                "storage": storage,
                "screened_in": screened_in,
                "complete_suite": complete,
                "passed": passed,
                "worst_final_normalized_rms": max(errors),
                "cases": cases,
            }
        )
        print(
            f"{policy.name}: screen={screen_error:.6%}, "
            f"worst={max(errors):.6%}, {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        del model
        release_device_cache(device)

    passing = [value for value in policy_results if value["passed"]]
    selected = min(passing, key=lambda value: value["storage"]["packed_bytes"]) if passing else None
    report = {
        "schema_version": 1,
        "model": "Qwen/Qwen-Image-2.1",
        "snapshot": calibration.MODEL_SNAPSHOT,
        "diffusers_commit": calibration.DIFFUSERS_COMMIT,
        "environment": {
            "torch": torch.__version__,
            "diffusers": diffusers_version,
            "transformers": transformers_version,
            "device": str(device),
        },
        "method": {
            "screen_cases": list(SCREEN_CASES),
            "screen_limit": calibration.FINAL_NRMSE_LIMIT,
            "passing_candidates_run_all_cases": True,
            "case_source": "tools/calibrate_transformer_quantization.py",
            "selection": "smallest mixed block-matrix storage with every final-noise nRMSE <= 0.01",
            "requested_policies": [policy.name for policy in policies],
        },
        "policies": policy_results,
        "selected_policy": None if selected is None else selected["name"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}; selected={report['selected_policy']}", flush=True)
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
