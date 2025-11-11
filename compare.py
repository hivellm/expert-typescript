"""
Qualitative Checkpoint Comparison - Expert TypeScript

This script runs the same prompts on all available checkpoints
and displays outputs for qualitative analysis by an external LLM.

Run with: F:/Node/hivellm/expert/cli/venv_windows/Scripts/python.exe compare.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CONFIGURATION - EXPERT-TYPESCRIPT
# ============================================================================

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINT_DIR = "weights/qwen3-06b"

GEN_CONFIG = {
    "max_new_tokens": 4000,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "do_sample": True,
}

# Representative prompts covering core TypeScript capabilities
test_cases = [
    {
        "id": "function_basic_001",
        "category": "basic_function",
        "system_prompt": "Task: typescript_function\nFocus: string utilities",
        "user_prompt": "Write a TypeScript function `isPalindrome` that returns true when a string reads the same forwards and backwards.",
        "expected_type": "typescript",
    },
    {
        "id": "interface_001",
        "category": "interface_design",
        "system_prompt": "Task: typescript_interface\nFocus: domain modeling",
        "user_prompt": "Create an interface for an ecommerce `Order` with id, items array, total, status union type, and createdAt date.",
        "expected_type": "typescript",
    },
    {
        "id": "generics_001",
        "category": "generics",
        "system_prompt": "Task: typescript_generics\nFocus: utility helpers",
        "user_prompt": "Create a generic utility `pluck<T, K extends keyof T>(items: T[], key: K): Array<T[K]>` with runtime implementation.",
        "expected_type": "typescript",
    },
    {
        "id": "type_guard_001",
        "category": "type_guard",
        "system_prompt": "Task: typescript_type_guard\nFocus: discriminated unions",
        "user_prompt": "Define a union `TextNode | ImageNode` and implement a type guard `isImageNode`.",
        "expected_type": "typescript",
    },
    {
        "id": "async_fetch_001",
        "category": "async",
        "system_prompt": "Task: typescript_async\nFocus: fetch wrappers",
        "user_prompt": "Implement an async function `fetchUser` that fetches JSON from `/api/users/{id}` with proper typing and error handling.",
        "expected_type": "typescript",
    },
    {
        "id": "class_001",
        "category": "class_design",
        "system_prompt": "Task: typescript_class\nFocus: stateful services",
        "user_prompt": "Create a class `EventBus` that allows registering listeners, emitting events, and removing listeners using generics for payload typing.",
        "expected_type": "typescript",
    },
    {
        "id": "mapped_types_001",
        "category": "advanced_types",
        "system_prompt": "Task: typescript_mapped_types\nFocus: type transformations",
        "user_prompt": "Define a mapped type `ReadonlyOptional<T>` that makes every property readonly and optional, and demonstrate it with an example.",
        "expected_type": "typescript",
    },
    {
        "id": "react_component_001",
        "category": "react",
        "system_prompt": "Task: typescript_react\nFocus: functional components",
        "user_prompt": "Write a typed React function component `UserCard` that accepts props for name, email, and optional avatar url.",
        "expected_type": "typescript",
    },
    {
        "id": "decorator_001",
        "category": "decorators",
        "system_prompt": "Task: typescript_decorators\nFocus: method decorators",
        "user_prompt": "Implement a simple method decorator `logCall` that logs method name and arguments, then apply it to a class method.",
        "expected_type": "typescript",
    },
    {
        "id": "transformer_001",
        "category": "type_manipulation",
        "system_prompt": "Task: typescript_type_utilities\nFocus: utility types",
        "user_prompt": "Create a conditional type `RequireAtLeastOne<T, Keys extends keyof T>` and show how it works on a configuration interface.",
        "expected_type": "typescript",
    },
    {
        "id": "enum_001",
        "category": "enums",
        "system_prompt": "Task: typescript_enum\nFocus: exhaustive checks",
        "user_prompt": "Define a string enum `JobStatus` and a function `assertNever` that ensures exhaustive switch handling.",
        "expected_type": "typescript",
    },
    {
        "id": "testing_001",
        "category": "testing",
        "system_prompt": "Task: typescript_testing\nFocus: unit tests",
        "user_prompt": "Using Vitest, write a test suite for a function `sum` that adds numbers and throws when arguments are not finite.",
        "expected_type": "typescript",
    },
]


# ============================================================================
# HELPER FUNCTIONS (from template - do not modify)
# ============================================================================

def detect_device() -> str:
    """Detect the preferred device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_checkpoints(checkpoint_dir: str):
    """Locate available checkpoints sorted by step."""
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            candidate = os.path.join(checkpoint_dir, item)
            if os.path.isdir(candidate) and item.startswith("checkpoint-"):
                try:
                    step = int(item.replace("checkpoint-", ""))
                except ValueError:
                    continue
                checkpoints.append((step, candidate))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_base_model(base_model_path: str, device: str):
    """Load base model and tokenizer."""
    print(f"\n[1/3] Loading Base Model from: {base_model_path}")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print(f"[OK] Base Model loaded (device: {device})")
    return model, tokenizer


def load_checkpoints(base_model_path: str, checkpoints, device: str):
    """Load all checkpoints with adapters."""
    checkpoint_models = {}
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    print(f"\n[2/3] Loading {len(checkpoints)} checkpoints...")
    for step, checkpoint_path in checkpoints:
        print(f"  Loading checkpoint-{step}...", end=" ", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
        )

        if device == "cpu":
            model = model.to(device)

        model = PeftModel.from_pretrained(model, checkpoint_path)
        checkpoint_models[step] = model
        print("[OK]")

    return checkpoint_models


def generate_output(model, tokenizer, system_prompt: str, user_prompt: str, gen_config: dict, device: str) -> str:
    """Generate a response with chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    gen_params = {
        **gen_config,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_params)
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    return generated_text


def print_separator(char: str = "=", width: int = 100) -> None:
    print(char * width)


def print_test_header(test_case: dict, test_num: int, total: int) -> None:
    print_separator()
    print(f"\nTEST {test_num}/{total}: {test_case.get('id', f'test_{test_num}')}")
    print(f"Category: {test_case.get('category', 'N/A')}")
    print(f"Expected type: {test_case.get('expected_type', 'N/A')}")
    print_separator("-")
    print("\n[SYSTEM PROMPT]")
    print(test_case["system_prompt"])
    print("\n[USER PROMPT]")
    print(test_case["user_prompt"])
    print_separator("-")


def print_output(label: str, output: str, max_length: int = 700) -> None:
    print(f"\n[{label}]")
    if len(output) > max_length:
        print(output[:max_length])
        print(f"\n... (truncated, total: {len(output)} characters)")
    else:
        print(output)


def main() -> None:
    device = detect_device()

    print_separator()
    print("QUALITATIVE CHECKPOINT COMPARISON - EXPERT TYPESCRIPT")
    print("This script generates outputs for external LLM analysis")
    print("Does not evaluate quality automatically")
    print_separator()

    if not test_cases:
        print("ERROR: No test cases defined!")
        sys.exit(1)

    checkpoints = find_checkpoints(CHECKPOINT_DIR)
    if not checkpoints:
        print(f"ERROR: No checkpoints found in: {CHECKPOINT_DIR}")
        print(f"Checkpoint directory: {os.path.abspath(CHECKPOINT_DIR)}")
        sys.exit(1)

    print(f"\nCheckpoints found: {[c[0] for c in checkpoints]}")
    print(f"Total tests: {len(test_cases)}")
    print(f"Device: {device}")

    base_model, tokenizer = load_base_model(BASE_MODEL_PATH, device)
    checkpoint_models = load_checkpoints(BASE_MODEL_PATH, checkpoints, device)

    print(f"\n[3/3] Running {len(test_cases)} tests...")
    print_separator()

    results = []

    for index, test_case in enumerate(test_cases, 1):
        print_test_header(test_case, index, len(test_cases))

        base_output = generate_output(
            base_model,
            tokenizer,
            test_case["system_prompt"],
            test_case["user_prompt"],
            GEN_CONFIG,
            device,
        )
        print_output("BASE MODEL", base_output)

        checkpoint_outputs: dict[int, str] = {}
        for step, model in checkpoint_models.items():
            ckp_output = generate_output(
                model,
                tokenizer,
                test_case["system_prompt"],
                test_case["user_prompt"],
                GEN_CONFIG,
                device,
            )
            checkpoint_outputs[step] = ckp_output
            print_output(f"CHECKPOINT-{step}", ckp_output)

        results.append(
            {
                "test_id": test_case.get("id", f"test_{index}"),
                "category": test_case.get("category", "N/A"),
                "expected_type": test_case.get("expected_type", "N/A"),
                "system_prompt": test_case["system_prompt"],
                "user_prompt": test_case["user_prompt"],
                "base_output": base_output,
                "checkpoint_outputs": checkpoint_outputs,
            }
        )

        print_separator()

    print_separator()
    print("\nEXECUTION SUMMARY")
    print_separator()
    print(f"Total tests executed: {len(test_cases)}")
    print(f"Checkpoints tested: {[c[0] for c in checkpoints]}")
    print(f"Base model: {BASE_MODEL_PATH}")
    print("\nAll outputs have been displayed above.")
    print("Analyze the results to determine:")
    print("  1. Which checkpoint delivers the best TypeScript quality")
    print("  2. Which checkpoint should be packaged")
    print("  3. Whether training is progressing appropriately")
    print_separator()

    output_file = "checkpoint_comparison_results.json"
    try:
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "expert": "expert-typescript",
                    "base_model": BASE_MODEL_PATH,
                    "checkpoints_tested": [c[0] for c in checkpoints],
                    "device": device,
                    "test_config": GEN_CONFIG,
                    "results": results,
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults saved to: {output_file}")
    except Exception as exc:  # pragma: no cover - informative logging
        print(f"\nWarning: Could not save results to JSON: {exc}")


if __name__ == "__main__":
    main()
