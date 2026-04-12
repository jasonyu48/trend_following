from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PARAMS = ["ma_len", "atr_len", "atr_mult", "stop_lookback"]


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
        "--neighbor-radius",
        type=int,
        default=2,
        help="1D neighborhood radius, in sorted grid steps, used to choose the highlighted value. Default: 2",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=DEFAULT_PARAMS,
        help="Parameter columns to include. Default: ma_len atr_len atr_mult stop_lookback",
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


def _coerce_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def build_distributions(
    rows: list[dict],
    top_pct: float,
    params: Sequence[str] | None = None,
) -> tuple[list[dict], dict]:
    if not 0 < top_pct <= 1:
        raise ValueError("--top-pct must be between 0 and 1")
    if not rows:
        raise ValueError("Input file has no rows")

    resolved_params = list(params or DEFAULT_PARAMS)
    missing = [param for param in resolved_params if param not in rows[0]]
    if missing:
        raise ValueError(f"Input rows are missing parameter columns: {missing}")

    top_n = math.ceil(len(rows) * top_pct)
    top_rows = rows[:top_n]
    counts = {
        param: collections.Counter(_coerce_jsonable(row[param]) for row in top_rows)
        for param in resolved_params
    }
    summary = {
        "total_rows": len(rows),
        "top_n": top_n,
        "top_pct": top_pct,
        "params": resolved_params,
        "distributions": {},
    }

    for param in resolved_params:
        items = sorted(counts[param].items(), key=lambda item: item[0])
        total = sum(count for _, count in items)
        summary["distributions"][param] = [
            {
                "value": _coerce_jsonable(value),
                "count": int(count),
                "ratio": count / total if total else 0.0,
            }
            for value, count in items
        ]
    return top_rows, summary


def select_param_values_by_1d_neighbor_median(
    summary: dict,
    neighbor_radius: int = 2,
) -> dict[str, Any]:
    if neighbor_radius < 0:
        raise ValueError("--neighbor-radius must be >= 0")

    selection = {
        "method": "top_slice_1d_neighbor_median_count",
        "neighbor_radius": int(neighbor_radius),
        "param_values": {},
        "details": {},
    }
    for param in summary.get("params", list(summary.get("distributions", {}).keys())):
        items = list(summary["distributions"].get(param, []))
        if not items:
            raise ValueError(f"No distribution items found for parameter {param!r}")
        counts = [int(item["count"]) for item in items]
        scored_items: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            left = max(0, idx - neighbor_radius)
            right = min(len(items), idx + neighbor_radius + 1)
            neighborhood = items[left:right]
            neighborhood_counts = [int(neigh["count"]) for neigh in neighborhood]
            scored_items.append(
                {
                    "value": item["value"],
                    "count": int(item["count"]),
                    "ratio": float(item["ratio"]),
                    "neighbor_median_count": float(statistics.median(neighborhood_counts)),
                    "neighbor_values": [neigh["value"] for neigh in neighborhood],
                    "neighbor_counts": neighborhood_counts,
                    "grid_index": idx,
                }
            )
        best_item = max(
            scored_items,
            key=lambda item: (
                float(item["neighbor_median_count"]),
                int(item["count"]),
                -int(item["grid_index"]),
            ),
        )
        best_item = {key: value for key, value in best_item.items() if key != "grid_index"}
        selection["param_values"][param] = best_item["value"]
        selection["details"][param] = best_item

    summary["selection"] = selection
    return selection


def plot_distributions(
    summary: dict,
    output_path: Path,
    chart_type: str,
    selected_values: dict[str, Any] | None = None,
) -> None:
    params = list(summary.get("params", list(summary["distributions"].keys())))
    n_params = max(1, len(params))
    ncols = min(2, n_params)
    nrows = math.ceil(n_params / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    selection_map = dict(summary.get("selection", {}).get("param_values", {}))
    if selected_values:
        selection_map.update(selected_values)

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(params):
            ax.axis("off")
            continue
        param = params[ax_idx]
        items = summary["distributions"][param]
        xs = [item["value"] for item in items]
        ys = [item["count"] for item in items]
        selected_value = selection_map.get(param)
        selected_idx = next((idx for idx, value in enumerate(xs) if value == selected_value), None)

        if chart_type == "bar":
            colors = ["#4C78A8"] * len(xs)
            if selected_idx is not None:
                colors[selected_idx] = "#E45756"
            ax.bar([str(x) for x in xs], ys, color=colors)
            ax.tick_params(axis="x", rotation=45)
            if selected_idx is not None:
                ax.text(
                    selected_idx,
                    ys[selected_idx],
                    f" selected={xs[selected_idx]}",
                    color="#E45756",
                    ha="center",
                    va="bottom",
                )
        else:
            ax.plot(xs, ys, marker="o", linewidth=2, color="#4C78A8")
            if selected_idx is not None:
                ax.scatter([xs[selected_idx]], [ys[selected_idx]], color="#E45756", s=70, zorder=3)
                ax.axvline(xs[selected_idx], linestyle="--", linewidth=1.2, color="#E45756", alpha=0.8)
                ax.annotate(
                    f"selected={xs[selected_idx]}",
                    xy=(xs[selected_idx], ys[selected_idx]),
                    xytext=(8, 10),
                    textcoords="offset points",
                    color="#E45756",
                )

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
    _, summary = build_distributions(rows, args.top_pct, params=args.params)
    select_param_values_by_1d_neighbor_median(summary, neighbor_radius=args.neighbor_radius)
    plot_distributions(summary, output_path, args.chart_type)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved plot: {output_path}")
    print(f"saved summary: {summary_output}")


if __name__ == "__main__":
    main()
