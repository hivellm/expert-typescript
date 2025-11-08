#!/usr/bin/env python3
"""
Analyze expert-typescript dataset coverage.

Outputs JSON with counts of variable declarations, functions, classes, imports, etc.
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path


def extract_assistant_text(chatml_text: str) -> str:
    """Return the assistant portion of a ChatML sample."""
    try:
        assistant_section = chatml_text.split("<|assistant|>", 1)[1]
    except IndexError:
        return ""
    assistant_section = assistant_section.split("<|end|>", 1)[0]
    return assistant_section.strip()


def quantile(values, q):
    if not values:
        return 0.0
    return statistics.quantiles(values, n=100)[int(q * 100) - 1]


def collect_stats(dataset_path: Path) -> dict:
    patterns = {
        "const_decl": re.compile(r"\bconst\s+[A-Za-z_$][\w$]*"),
        "let_decl": re.compile(r"\blet\s+[A-Za-z_$][\w$]*"),
        "var_decl": re.compile(r"\bvar\s+[A-Za-z_$][\w$]*"),
        "function_decl": re.compile(r"\bfunction\s+[A-Za-z_$][\w$]*"),
        "arrow_function": re.compile(r"=>"),
        "class_decl": re.compile(r"\bclass\s+[A-Za-z_$][\w$]*"),
        "interface_decl": re.compile(r"\binterface\s+[A-Za-z_$][\w$]*"),
        "type_alias": re.compile(r"\btype\s+[A-Za-z_$][\w$]*\s*="),
        "enum_decl": re.compile(r"\benum\s+[A-Za-z_$][\w$]*"),
        "import_stmt": re.compile(r"\bimport\s"),
        "export_stmt": re.compile(r"\bexport\s"),
        "implements": re.compile(r"\bimplements\b"),
        "extends": re.compile(r"\bextends\b"),
        "decorator": re.compile(r"@[A-Za-z_][\w$]*"),
        "namespace_decl": re.compile(r"\bnamespace\s+[A-Za-z_$][\w$]*"),
        "async_keyword": re.compile(r"\basync\b"),
        "await_keyword": re.compile(r"\bawait\b"),
        "promise_usage": re.compile(r"\bPromise<"),
    }

    import_from_pattern = re.compile(r"import\s+(?:[^;]+?)\s+from\s+['\"]([^'\"]+)['\"]")
    side_effect_import_pattern = re.compile(r"import\s+['\"]([^'\"]+)['\"]")
    require_pattern = re.compile(r"require\(['\"]([^'\"]+)['\"]\)")

    occurrence_counts = Counter()
    example_counts = Counter()
    import_targets = Counter()

    export_counts = Counter()

    line_lengths = []
    char_lengths = []
    word_counts = []

    total_examples = 0

    with dataset_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            assistant = extract_assistant_text(payload.get("text", ""))
            if not assistant:
                continue

            total_examples += 1
            line_lengths.append(assistant.count("\\n") + 1)
            char_lengths.append(len(assistant))
            word_counts.append(len(assistant.split()))

            for key, pattern in patterns.items():
                matches = pattern.findall(assistant)
                if matches:
                    occurrence_counts[key] += len(matches)
                    example_counts[key] += 1

            for module in import_from_pattern.findall(assistant):
                import_targets[module] += 1
            for module in side_effect_import_pattern.findall(assistant):
                import_targets[module] += 1
            for module in require_pattern.findall(assistant):
                import_targets[module] += 1

            if "export default" in assistant:
                export_counts["export_default"] += 1
            if "export {" in assistant or "export{" in assistant:
                export_counts["export_named"] += 1

    return {
        "total_examples": total_examples,
        "line_stats": {
            "min": min(line_lengths) if line_lengths else 0,
            "max": max(line_lengths) if line_lengths else 0,
            "mean": statistics.mean(line_lengths) if line_lengths else 0,
            "median": statistics.median(line_lengths) if line_lengths else 0,
            "p90": quantile(line_lengths, 0.90) if len(line_lengths) >= 10 else 0,
        },
        "char_stats": {
            "min": min(char_lengths) if char_lengths else 0,
            "max": max(char_lengths) if char_lengths else 0,
            "mean": statistics.mean(char_lengths) if char_lengths else 0,
            "median": statistics.median(char_lengths) if char_lengths else 0,
            "p90": quantile(char_lengths, 0.90) if len(char_lengths) >= 10 else 0,
        },
        "word_stats": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "mean": statistics.mean(word_counts) if word_counts else 0,
            "median": statistics.median(word_counts) if word_counts else 0,
            "p90": quantile(word_counts, 0.90) if len(word_counts) >= 10 else 0,
        },
        "occurrence_counts": occurrence_counts,
        "example_counts": example_counts,
        "top_imports": import_targets.most_common(20),
        "export_counts": export_counts,
    }


def main() -> None:
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "train.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    summary = collect_stats(dataset_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


