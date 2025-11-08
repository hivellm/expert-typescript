#!/usr/bin/env python3
"""
Generate distribution charts for the expert-typescript dataset.

Breaks down the dataset by the first line of each user prompt (which maps to the
lesson/topic) and emits bar/pie charts plus a summary table saved under docs/.
"""

import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def extract_user_title(chatml_text: str) -> str:
    """Return the first line of the <|user|> section."""
    try:
        user_section = chatml_text.split("<|user|>\n", 1)[1]
        user_content = user_section.split("\n<|end|>", 1)[0].strip()
    except IndexError:
        return "Unknown"

    if not user_content:
        return "Unknown"

    first_line = user_content.splitlines()[0].strip()
    if not first_line:
        return "Unknown"

    return first_line


def prepare_distribution(
    dataset_path: Path, max_categories: int = 15
) -> Tuple[List[str], List[int], int]:
    counts = Counter()
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = extract_user_title(payload.get("text", ""))
            counts[title] += 1

    total = sum(counts.values())
    top = counts.most_common(max_categories)

    if total and len(counts) > max_categories:
        other_count = total - sum(count for _, count in top)
        if other_count > 0:
            top.append(("Other", other_count))

    labels = [label for label, _ in top]
    values = [count for _, count in top]

    return labels, values, total


def shorten(label: str, limit: int = 48) -> str:
    return label if len(label) <= limit else label[: limit - 3] + "..."


def render_chart(labels: List[str], counts: List[int], total: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    percentages = [c / total * 100 if total else 0 for c in counts]
    short_labels = [shorten(label) for label in labels]
    color_primary = "#3178C6"

    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(18, max(6, 0.4 * len(short_labels))), gridspec_kw={"width_ratios": [3, 2]}
    )

    y_positions = range(len(short_labels))
    bars = ax_bar.barh(y_positions, counts, color=color_primary)
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels(short_labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Number of examples", fontweight="bold")
    ax_bar.set_title("expert-typescript dataset: top sections", fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.3)

    x_offset = max(counts) * 0.01 if counts else 1
    for bar, count, pct in zip(bars, counts, percentages):
        ax_bar.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax_pie.pie(
        counts,
        labels=[shorten(label, 32) for label in labels],
        autopct="%1.1f%%",
        startangle=120,
        colors=plt.cm.Blues(
            [0.25 + 0.6 * i / max(1, len(counts) - 1) for i in range(len(counts))]
        ),
        textprops={"fontsize": 9},
    )
    ax_pie.set_title("Share by section", fontweight="bold")

    fig.suptitle(
        f"expert-typescript dataset distribution (total = {total:,} examples)", fontsize=16
    )
    plt.tight_layout()

    png_path = output_dir / "dataset_distribution.png"
    pdf_path = output_dir / "dataset_distribution.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Charts saved to {png_path} and {pdf_path}")


def print_summary(labels: List[str], counts: List[int], total: int) -> None:
    print("=" * 80)
    print("DATASET SECTION BREAKDOWN")
    print("=" * 80)
    print(f"Total examples: {total:,}")
    print()
    print(f"{'Section':48} | {'Count':>10} | {'Share':>7}")
    print("-" * 80)
    for label, count in zip(labels, counts):
        pct = count / total * 100 if total else 0
        print(f"{shorten(label, 48):48} | {count:10,} | {pct:6.2f}%")
    print("=" * 80)


def main() -> None:
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "train.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    labels, counts, total = prepare_distribution(dataset_path)
    if total == 0:
        raise RuntimeError("No examples found in dataset.")

    output_dir = Path(__file__).resolve().parent.parent / "docs"
    render_chart(labels, counts, total, output_dir)
    print_summary(labels, counts, total)


if __name__ == "__main__":
    main()


