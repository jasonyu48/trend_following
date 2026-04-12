from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from run_backtest import build_engine_config, build_strategy_params, _load_fx_daily, _load_symbol_specs
from run_cta_workflow import _build_markets_from_cache, _load_m1_cache, _slice_m1_window
from search_params import run_portfolio_backtest
from strategies import STRATEGIES, get_strategy

_ANALYZE_MARKET_CACHE: dict[tuple[str, str], dict[str, Any]] | None = None
_ANALYZE_CONFIG_CACHE: dict[str, Any] | None = None
_ANALYZE_PORTFOLIO_KWARGS: dict[str, Any] | None = None
_ANALYZE_SYMBOL: str | None = None
_ANALYZE_STRATEGY_NAME: str | None = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate whether top-N workflow params generalize from validation to test.")
    p.add_argument("--workflow-dir", type=Path, default=Path("results/cta_workflow4"))
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--symbol-specs", type=Path, default=Path("symbol_specs.json"))
    p.add_argument("--fx-daily", type=Path, default=Path("data/fx_daily/fx_daily_2012_2025.csv"))
    p.add_argument("--val-weight", type=float, default=0.3)
    p.add_argument("--gap-weight", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", type=Path, default="results/cta_workflow4/generalization_check")
    return p.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _resolve_param_results_path(workflow_dir: Path) -> Path:
    for name in ["param_search_results.jsonl", "param_search_results.csv"]:
        path = workflow_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No param search results found in {workflow_dir}")


def _write_records_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in df.to_dict(orient="records"):
            fh.write(json.dumps(record, default=str))
            fh.write("\n")


def _normalize_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(0.0, index=series.index, dtype="float64")
    min_value = float(finite.min())
    max_value = float(finite.max())
    if np.isclose(min_value, max_value):
        return pd.Series(0.0, index=series.index, dtype="float64")
    return ((values - min_value) / (max_value - min_value)).fillna(0.0).astype("float64")


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").rank(method="average")
    y = pd.to_numeric(b, errors="coerce").rank(method="average")
    value = x.corr(y)
    return float(value) if pd.notna(value) else float("nan")


def _split_oos(validation_start: str, portfolio_end: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    val_start = pd.Timestamp(validation_start, tz="UTC")
    test_end = pd.Timestamp(portfolio_end, tz="UTC")
    midpoint = val_start + (test_end - val_start) / 2
    midpoint = midpoint.floor("D")
    return val_start, midpoint, test_end


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _evaluate_split(
    *,
    markets: dict[str, Any],
    symbol: str,
    params: object,
    strategy: Any,
    config: Any,
    portfolio_kwargs: dict[str, Any],
) -> dict[str, Any]:
    result = run_portfolio_backtest(
        market_by_symbol=markets,
        params=params,
        strategy=strategy,
        config=config,
        show_progress=False,
        collect_events=False,
        **portfolio_kwargs,
    )
    return dict(result["symbol_results"][symbol]["stats"])


def _prepare_market_cache(
    *,
    m1: pd.DataFrame,
    symbol: str,
    timeframes: list[str],
    phase_windows: dict[str, tuple[str | pd.Timestamp, str | pd.Timestamp]],
) -> dict[tuple[str, str], dict[str, Any]]:
    bars_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    market_cache: dict[tuple[str, str], dict[str, Any]] = {}
    total = len(phase_windows) * len(timeframes)
    for phase, (start, end) in tqdm(phase_windows.items(), total=len(phase_windows), desc="Preparing phase windows", leave=False):
        start_ts = _utc_timestamp(start)
        end_ts = _utc_timestamp(end)
        sliced = _slice_m1_window(
            {symbol: m1},
            str(start_ts.strftime("%Y%m%d")),
            str(end_ts.strftime("%Y%m%d")),
        )
        for timeframe in tqdm(timeframes, total=len(timeframes), desc=f"Resampling {phase}", leave=False):
            market_cache[(phase, timeframe)] = _build_markets_from_cache(
                sliced,
                {symbol: timeframe},
                bars_cache=bars_cache,
                phase=phase,
            )
    return market_cache


def _init_analyze_worker(
    market_cache: dict[tuple[str, str], dict[str, Any]],
    config_cache: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
    symbol: str,
    strategy_name: str,
) -> None:
    global _ANALYZE_MARKET_CACHE, _ANALYZE_CONFIG_CACHE, _ANALYZE_PORTFOLIO_KWARGS
    global _ANALYZE_SYMBOL, _ANALYZE_STRATEGY_NAME
    _ANALYZE_MARKET_CACHE = market_cache
    _ANALYZE_CONFIG_CACHE = config_cache
    _ANALYZE_PORTFOLIO_KWARGS = portfolio_kwargs
    _ANALYZE_SYMBOL = symbol
    _ANALYZE_STRATEGY_NAME = strategy_name


def _evaluate_candidate(task: dict[str, Any]) -> dict[str, Any]:
    if (
        _ANALYZE_MARKET_CACHE is None
        or _ANALYZE_CONFIG_CACHE is None
        or _ANALYZE_PORTFOLIO_KWARGS is None
        or _ANALYZE_SYMBOL is None
        or _ANALYZE_STRATEGY_NAME is None
    ):
        raise RuntimeError("Analyze worker not initialized")
    rank = int(task["rank"])
    timeframe = str(task["timeframe"])
    symbol = _ANALYZE_SYMBOL
    strategy_name = _ANALYZE_STRATEGY_NAME
    strategy = get_strategy(strategy_name)

    param_values = {
        name: _coerce_param_value(task[name])
        for name in strategy.param_names
    }
    params = build_strategy_params(strategy_name, param_values)
    train_stats = _evaluate_split(
        markets=_ANALYZE_MARKET_CACHE[("train", timeframe)],
        symbol=symbol,
        params=params,
        strategy=strategy,
        config=_ANALYZE_CONFIG_CACHE[timeframe],
        portfolio_kwargs=_ANALYZE_PORTFOLIO_KWARGS,
    )
    val_stats = _evaluate_split(
        markets=_ANALYZE_MARKET_CACHE[("val", timeframe)],
        symbol=symbol,
        params=params,
        strategy=strategy,
        config=_ANALYZE_CONFIG_CACHE[timeframe],
        portfolio_kwargs=_ANALYZE_PORTFOLIO_KWARGS,
    )
    test_stats = _evaluate_split(
        markets=_ANALYZE_MARKET_CACHE[("test", timeframe)],
        symbol=symbol,
        params=params,
        strategy=strategy,
        config=_ANALYZE_CONFIG_CACHE[timeframe],
        portfolio_kwargs=_ANALYZE_PORTFOLIO_KWARGS,
    )
    return {
        "candidate_rank_in_workflow": rank,
        "symbol": symbol,
        "timeframe": timeframe,
        **param_values,
        "train_calmar": task["train_calmar"],
        "train_annualized_return": task["train_annualized_return"],
        "train_max_drawdown": task["train_max_drawdown"],
        "val_calmar": val_stats.get("calmar"),
        "val_annualized_return": val_stats.get("annualized_return"),
        "val_max_drawdown": val_stats.get("max_drawdown"),
        "test_calmar": test_stats.get("calmar"),
        "test_annualized_return": test_stats.get("annualized_return"),
        "test_max_drawdown": test_stats.get("max_drawdown"),
        "train_val_calmar_gap_abs": abs(float(task["train_calmar"]) - float(val_stats.get("calmar", np.nan))),
        "workflow_weighted_score": task.get("weighted_score", np.nan),
    }


def main() -> int:
    args = _parse_args()
    workflow_dir = args.workflow_dir.resolve()
    out_dir = (args.out_dir or (workflow_dir / "generalization_check")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow_summary = _read_json(workflow_dir / "workflow_summary.json")
    param_results_path = _resolve_param_results_path(workflow_dir)
    param_results = _read_table(param_results_path)
    if param_results.empty:
        raise ValueError(f"No parameter rows found in {param_results_path}")

    top_n = max(1, min(int(args.top_n), len(param_results)))
    top_params = param_results.head(top_n).copy()

    strategy_name = str(workflow_summary["strategy"])
    opt_symbol = str(workflow_summary["opt_symbol"]).upper()
    resolved_opt_timeframe = str(workflow_summary.get("resolved_opt_timeframe") or workflow_summary.get("opt_timeframe") or "4H")
    train_start = str(workflow_summary["selection_start"])
    train_end = str(workflow_summary["selection_end"])
    val_start, test_start, test_end = _split_oos(train_end, str(workflow_summary["portfolio_end"]))

    symbol_specs = _load_symbol_specs(args.symbol_specs)
    fx_daily = _load_fx_daily(args.fx_daily)
    strategy = get_strategy(strategy_name)

    config_kwargs = {
        "commission_bps": float(workflow_summary.get("commission_bps", 2.0)),
        "slippage_bps": float(workflow_summary.get("slippage_bps", 0.0)),
        "overnight_long_rate": float(workflow_summary.get("overnight_long_rate", 0.0)),
        "overnight_short_rate": float(workflow_summary.get("overnight_short_rate", 0.0)),
        "overnight_day_count": int(workflow_summary.get("overnight_day_count", 360)),
        "initial_equity": float(workflow_summary["portfolio_stats"]["initial_equity"]),
        "initial_margin_ratio": float(workflow_summary["portfolio_stats"]["initial_margin_ratio"]),
        "maintenance_margin_ratio": float(workflow_summary["portfolio_stats"]["maintenance_margin_ratio"]),
    }
    portfolio_kwargs = {
        "portfolio_mode": str(workflow_summary["portfolio_mode"]),
        "cash_per_trade": float(workflow_summary["portfolio_stats"].get("cash_per_trade", 1000.0)),
        "risk_per_trade": float(workflow_summary["portfolio_stats"].get("risk_per_trade", 100.0)),
        "risk_per_trade_pct": float(workflow_summary["portfolio_stats"].get("risk_per_trade_pct", 0.01)),
    }

    load_args = argparse.Namespace(
        data_dir=args.data_dir,
        symbols=[opt_symbol],
        selection_start=train_start,
        selection_end=str(val_start.strftime("%Y%m%d")),
        start=train_start,
        end=str(test_end.strftime("%Y%m%d")),
    )
    m1 = _load_m1_cache(load_args)[opt_symbol]
    unique_timeframes = sorted({str(getattr(row, "timeframe", resolved_opt_timeframe)) for row in top_params.itertuples(index=False)})
    phase_windows = {
        "train": (train_start, train_end),
        "val": (val_start, test_start),
        "test": (test_start, test_end),
    }
    market_cache = _prepare_market_cache(
        m1=m1,
        symbol=opt_symbol,
        timeframes=unique_timeframes,
        phase_windows=phase_windows,
    )
    config_cache = {
        timeframe: build_engine_config(
            default_timeframe=timeframe,
            timeframe_by_symbol={opt_symbol: timeframe},
            symbol_specs=symbol_specs,
            fx_daily=fx_daily,
            **config_kwargs,
        )
        for timeframe in unique_timeframes
    }

    tasks = [
        {
            "rank": rank,
            "timeframe": str(getattr(row, "timeframe", resolved_opt_timeframe)),
            **row._asdict(),
            "train_calmar": float(getattr(row, "calmar", np.nan)),
            "train_annualized_return": float(getattr(row, "annualized_return", np.nan)),
            "train_max_drawdown": float(getattr(row, "max_drawdown", np.nan)),
        }
        for rank, row in enumerate(top_params.itertuples(index=False), start=1)
    ]
    workers = max(1, min(int(args.workers), len(tasks)))
    rows: list[dict[str, Any]] = []
    if workers == 1:
        _init_analyze_worker(market_cache, config_cache, portfolio_kwargs, opt_symbol, strategy_name)
        rows = [
            _evaluate_candidate(task)
            for task in tqdm(tasks, total=len(tasks), desc="Evaluating candidates")
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_analyze_worker,
            initargs=(market_cache, config_cache, portfolio_kwargs, opt_symbol, strategy_name),
        ) as executor:
            futures = [executor.submit(_evaluate_candidate, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating candidates"):
                rows.append(future.result())

    results = pd.DataFrame(rows).sort_values("candidate_rank_in_workflow").reset_index(drop=True)
    results["norm_val_calmar"] = _normalize_minmax(results["val_calmar"])
    results["norm_stability"] = 1.0 - _normalize_minmax(results["train_val_calmar_gap_abs"])
    results["generalization_score"] = (
        float(args.val_weight) * results["norm_val_calmar"]
        + float(args.gap_weight) * results["norm_stability"]
    )
    results = results.sort_values(
        ["generalization_score", "val_calmar", "test_calmar"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    _write_records_jsonl(results, out_dir / "candidate_generalization.jsonl")

    summary = {
        "workflow_dir": str(workflow_dir),
        "param_results_path": str(param_results_path),
        "top_n": int(top_n),
        "symbol": opt_symbol,
        "train_start": train_start,
        "train_end": train_end,
        "val_start": str(val_start.date()),
        "val_end": str(test_start.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "val_weight": float(args.val_weight),
        "gap_weight": float(args.gap_weight),
        "workers": int(workers),
        "train_metrics_source": str(param_results_path),
        "config_source": "workflow_summary.json portfolio_stats + optional top-level fields",
        "spearman_val_vs_test_calmar": _rank_corr(results["val_calmar"], results["test_calmar"]),
        "spearman_generalization_score_vs_test_calmar": _rank_corr(results["generalization_score"], results["test_calmar"]),
        "spearman_workflow_score_vs_test_calmar": _rank_corr(results["workflow_weighted_score"], results["test_calmar"]),
        "top_by_generalization_score": results.head(10).to_dict(orient="records"),
        "top_by_test_calmar": results.sort_values("test_calmar", ascending=False).head(10).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
