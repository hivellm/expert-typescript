#!/usr/bin/env python3
"""
Generate Markdown coverage report for expert-typescript dataset.
"""

from pathlib import Path

from analyze_dataset import collect_stats


def format_report(stats: dict) -> str:
    total = stats["total_examples"]
    char_stats = stats["char_stats"]
    word_stats = stats["word_stats"]
    occurrence_counts = stats["occurrence_counts"]
    example_counts = stats["example_counts"]

    ordered_features = [
        ("export_stmt", "export statements"),
        ("interface_decl", "interface declarations"),
        ("class_decl", "class declarations"),
        ("function_decl", "function declarations"),
        ("const_decl", "const declarations"),
        ("let_decl", "let declarations"),
        ("arrow_function", "arrow functions (=>)"),
        ("extends", "extends clauses"),
        ("implements", "implements clauses"),
        ("decorator", "decorators (@)"),
        ("promise_usage", "Promise<T> usage"),
        ("async_keyword", "async keyword"),
        ("await_keyword", "await keyword"),
        ("import_stmt", "import statements"),
        ("type_alias", "type aliases"),
        ("var_decl", "var declarations"),
        ("enum_decl", "enum declarations"),
        ("namespace_decl", "namespace declarations"),
    ]

    feature_rows = []
    for code, label in ordered_features:
        occurrences = occurrence_counts.get(code, 0)
        examples = example_counts.get(code, 0)
        coverage = (examples / total * 100) if total else 0.0
        feature_rows.append((label, occurrences, examples, coverage))

    feature_table_lines = [
        "| Feature | Total occurrences | Examples | Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, occ, examples, pct in feature_rows:
        feature_table_lines.append(
            f"| {label} | {occ:,} | {examples:,} | {pct:5.2f}% |"
        )

    imports_table = ""
    top_imports = stats["top_imports"][:10]
    if top_imports:
        lines = ["## Top import targets (sample)"]
        lines.append("| Module | Count |")
        lines.append("| --- | ---: |")
        for module, count in top_imports:
            lines.append(f"| `{module}` | {count:,} |")
        imports_table = "\n".join(lines)

    exports_table_lines = [
        "## Export styles",
        "| Pattern | Examples | Share |",
        "| --- | ---: | ---: |",
    ]
    export_variants = [
        ("Any export keyword", example_counts.get("export_stmt", 0)),
        ("Named object exports", stats["export_counts"].get("export_named", 0)),
        ("Default exports", stats["export_counts"].get("export_default", 0)),
    ]
    for label, count in export_variants:
        share = (count / total * 100) if total else 0.0
        exports_table_lines.append(f"| {label} | {count:,} | {share:5.2f}% |")

    lines = [
        "# Dataset Coverage Snapshot (TypeScript)",
        "",
        f"- **Total examples analysed:** {total:,}",
        f"- **Median assistant response length:** {int(char_stats['median'])} characters "
        f"({int(word_stats['median'])} words)",
        f"- **90th percentile response length:** {int(char_stats['p90'])} characters",
        f"- **Longest response:** {int(char_stats['max']):,} characters "
        f"({int(word_stats['max']):,} words)",
        "",
        "## Feature Coverage",
        *feature_table_lines,
    ]

    if imports_table:
        lines.extend(["", imports_table])

    lines.extend(["", *exports_table_lines])

    return "\n".join(lines) + "\n"


def main() -> None:
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "train.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    stats = collect_stats(dataset_path)
    report = format_report(stats)

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    report_path = docs_dir / "DATASET_STATUS.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OK] Report written to {report_path}")


if __name__ == "__main__":
    main()


