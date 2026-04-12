from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAMS = ["ma_len", "atr_len", "atr_mult", "stop_lookback"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot parameter distributions for the top slice of a param_search_results.jsonl file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to param_search_results.jsonl",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=0.2,
        help="Fraction of rows taken from the head of the file. Default: 0.2",
    )
    parser.add_argument(
        "--chart-type",
        choices=["line", "bar"],
        default="line",
        help="Chart style for each parameter distribution. Default: line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <input_dir>/top20_param_distributions.png",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional summary json output. Default: <input_dir>/top20_param_distributions_summary.json",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_distributions(rows: list[dict], top_pct: float) -> tuple[list[dict], dict]:
    if not 0 < top_pct <= 1:
        raise ValueError("--top-pct must be between 0 and 1")
    if not rows:
        raise ValueError("Input file has no rows")

    top_n = math.ceil(len(rows) * top_pct)
    top_rows = rows[:top_n]
    counts = {
        param: collections.Counter(row[param] for row in top_rows)
        for param in PARAMS
    }
    summary = {
        "total_rows": len(rows),
        "top_n": top_n,
        "top_pct": top_pct,
        "distributions": {},
    }

    for param in PARAMS:
        items = sorted(counts[param].items(), key=lambda item: item[0])
        total = sum(count for _, count in items)
        summary["distributions"][param] = [
            {
                "value": value,
                "count": count,
                "ratio": count / total,
            }
            for value, count in items
        ]
    return top_rows, summary


def plot_distributions(summary: dict, output_path: Path, chart_type: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, param in zip(axes, PARAMS):
        items = summary["distributions"][param]
        xs = [item["value"] for item in items]
        ys = [item["count"] for item in items]

        if chart_type == "bar":
            ax.bar([str(x) for x in xs], ys, color="#4C78A8")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.plot(xs, ys, marker="o", linewidth=2, color="#4C78A8")

        ax.set_title(f"{param} distribution in top {summary['top_pct']:.0%}")
        ax.set_xlabel(param)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Top {summary['top_pct']:.0%} parameter distributions "
        f"({summary['top_n']} / {summary['total_rows']} rows)"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output or input_path.with_name("top20_param_distributions.png")
    summary_output = args.summary_output or input_path.with_name("top20_param_distributions_summary.json")

    rows = load_rows(input_path)
    _, summary = build_distributions(rows, args.top_pct)
    plot_distributions(summary, output_path, args.chart_type)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved plot: {output_path}")
    print(f"saved summary: {summary_output}")


if __name__ == "__main__":
    main()
