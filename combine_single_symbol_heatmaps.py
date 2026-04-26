from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine single-symbol heatmap result folders into one multi-panel figure with a shared color scale."
    )
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--strategy", type=str, default="supertrend")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--metric", type=str, choices=["max_recovery_time", "total_return", "calmar"], required=True)
    p.add_argument("--out-path", type=Path, default=None)
    return p.parse_args()


def _timeframe_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    text = str(value)
    try:
        offset = pd.tseries.frequencies.to_offset(text.strip().lower())
        delta = pd.Timedelta(offset)
        return (1, float(delta / pd.Timedelta(minutes=1)))
    except (ValueError, TypeError):
        return (2, text)


def _load_records_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def discover_heatmap_inputs(
    results_root: Path,
    glob_pattern: str,
    timeframe_from_path: Callable[[Path], str],
) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for directory in sorted(results_root.glob(glob_pattern)):
        if not directory.is_dir():
            continue
        results_path = directory / "heatmap_results.jsonl"
        if results_path.exists():
            pairs.append((str(timeframe_from_path(directory)), results_path))
    return sorted(pairs, key=lambda item: _timeframe_sort_key(item[0]))


def _discover_heatmap_inputs(results_root: Path, strategy: str, symbol: str) -> list[tuple[str, Path]]:
    pattern = f"single_symbol_param_heatmap_{strategy}_{str(symbol).upper()}_*"
    return discover_heatmap_inputs(
        results_root,
        pattern,
        timeframe_from_path=lambda directory: directory.name.rsplit("_", 1)[-1],
    )


def _build_metric_pivot(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    required_cols = {"atr_len", "atr_mult", metric}
    if results.empty or not required_cols.issubset(results.columns):
        return pd.DataFrame()
    grid = results.loc[:, ["atr_len", "atr_mult", metric]].copy()
    grid["atr_len"] = pd.to_numeric(grid["atr_len"], errors="coerce")
    grid["atr_mult"] = pd.to_numeric(grid["atr_mult"], errors="coerce")
    grid[metric] = pd.to_numeric(grid[metric], errors="coerce")
    grid = grid.dropna(subset=["atr_len", "atr_mult", metric])
    if grid.empty:
        return pd.DataFrame()
    return grid.pivot(index="atr_mult", columns="atr_len", values=metric).sort_index().sort_index(axis=1)


def _collect_pivots(inputs: list[tuple[str, Path]], metric: str) -> list[tuple[str, pd.DataFrame]]:
    pivots: list[tuple[str, pd.DataFrame]] = []
    for timeframe, path in inputs:
        pivot = _build_metric_pivot(_load_records_jsonl(path), metric)
        if not pivot.empty:
            pivots.append((timeframe, pivot))
    return pivots


def _format_cell(metric: str, value: float) -> str:
    if metric == "max_recovery_time":
        return f"{float(value):.0f}"
    return f"{float(value):.2f}"


def _default_out_path(results_root: Path, strategy: str, symbol: str, metric: str) -> Path:
    return results_root / f"single_symbol_param_heatmap_{strategy}_{str(symbol).upper()}_{metric}_combined.png"


def _plot_combined_heatmaps(
    pivots: list[tuple[str, pd.DataFrame]],
    metric: str,
    out_path: Path,
    *,
    symbol: str,
    strategy: str,
) -> None:
    if not pivots:
        raise ValueError(f"No non-empty pivots available for metric {metric!r}")

    all_values = np.concatenate([pivot.to_numpy().ravel() for _, pivot in pivots])
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        raise ValueError(f"No finite values available for metric {metric!r}")
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    n = len(pivots)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 4.4 * nrows),
        squeeze=False,
    )
    cmap = "viridis" if metric == "max_recovery_time" else "RdYlGn"

    mesh = None
    for ax, (timeframe, pivot) in zip(axes.ravel(), pivots):
        mesh = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(str(timeframe))
        ax.set_xlabel("atr_len")
        ax.set_ylabel("atr_mult")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(v) for v in pivot.index])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for row_idx, atr_mult in enumerate(pivot.index):
            for col_idx, atr_len in enumerate(pivot.columns):
                value = pivot.loc[atr_mult, atr_len]
                if pd.isna(value):
                    continue
                color = "black" if norm(float(value)) > 0.58 else "white"
                ax.text(
                    col_idx,
                    row_idx,
                    _format_cell(metric, float(value)),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )

    for ax in axes.ravel()[len(pivots) :]:
        ax.axis("off")

    if mesh is None:
        raise RuntimeError("Expected at least one mesh to be created")
    fig.subplots_adjust(right=0.90, wspace=0.25, hspace=0.32, top=0.90)
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(metric)
    fig.suptitle(f"{str(symbol).upper()} {strategy} {metric} heatmaps", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_combined_heatmap(
    inputs: list[tuple[str, Path]],
    metric: str,
    out_path: Path,
    *,
    symbol: str,
    strategy: str,
) -> int:
    pivots = _collect_pivots(inputs, metric)
    _plot_combined_heatmaps(
        pivots,
        metric,
        out_path,
        symbol=symbol,
        strategy=strategy,
    )
    return len(pivots)


def main() -> int:
    args = _parse_args()
    inputs = _discover_heatmap_inputs(args.results_root, args.strategy, args.symbol)
    if not inputs:
        raise ValueError(
            f"No heatmap result folders found for symbol {args.symbol!r} and strategy {args.strategy!r} under {args.results_root}"
        )
    out_path = args.out_path or _default_out_path(args.results_root, args.strategy, args.symbol, args.metric)
    n_panels = write_combined_heatmap(
        inputs,
        args.metric,
        out_path,
        symbol=args.symbol,
        strategy=args.strategy,
    )
    print(json.dumps({"metric": args.metric, "out_path": str(out_path), "n_panels": n_panels}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
