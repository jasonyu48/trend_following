from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

import search_params as sp
from combine_single_symbol_heatmaps import write_combined_heatmap
from data_paths import DEFAULT_DATA_DIR
from run_backtest import _load_fx_daily, _load_symbol_specs, normalize_symbols
from run_cta_workflow import (
    ATR50_HL_SUM_RATIO_MODE,
    _build_markets_from_cache,
    _build_shared_m1_payload,
    _cleanup_shared_blocks,
    _load_m1_cache,
    _make_config,
    _resolve_timeframes_for_opt_timeframe,
    _slice_m1_window,
    _timeframe_sort_key,
    _write_json,
    _write_records_jsonl,
)
from search_params import run_portfolio_backtest
from strategies import get_strategy


DEFAULT_SYMBOLS = ["XAUUSD", "GBPJPY"]
DEFAULT_TIMEFRAME_CANDIDATES = ["1H", "2H", "3H", "4H", "6H", "8H", "12H"]
HEATMAP_METRICS = [
    ("max_recovery_time", True),
    ("total_return", False),
    ("calmar", False),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parallel supertrend portfolio development workflow: scan opt timeframes, match other symbols via "
            "median ATR(50)/(high+low), run the full supertrend grid, and build combined heatmaps."
        )
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--symbol-specs", type=Path, default=Path("symbol_specs.json"))
    p.add_argument("--fx-daily", type=Path, default=Path("data/fx_daily/fx_daily_2012_2025.csv"))
    p.add_argument("--start", type=str, default="20120101")
    p.add_argument("--end", type=str, default="20200101")
    p.add_argument("--selection-start", type=str, default="20120101", help="used for timeframe selection")
    p.add_argument("--selection-end", type=str, default="20200101", help="used for timeframe selection")
    p.add_argument("--opt-symbol", type=str, default="XAUUSD")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframe-candidates", nargs="+", default=DEFAULT_TIMEFRAME_CANDIDATES)
    p.add_argument("--commission-bps", type=float, default=0.35)
    p.add_argument("--slippage-bps", type=float, default=0.3)
    p.add_argument("--overnight-long-rate", type=float, default=0.0)
    p.add_argument("--overnight-short-rate", type=float, default=0.0)
    p.add_argument("--overnight-day-count", type=int, default=360)
    p.add_argument("--initial-margin-ratio", type=float, default=0.01)
    p.add_argument("--maintenance-margin-ratio", type=float, default=0.005)
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--initial-equity", type=float, default=1000000.0)
    p.add_argument(
        "--portfolio-mode",
        type=str,
        default="fixed_risk",
        choices=["fixed_cash", "fixed_risk", "fixed_risk_pct"],
    )
    p.add_argument("--cash-per-trade", type=float, default=100000.0)
    p.add_argument("--risk-per-trade", type=float, default=10000.0)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    p.add_argument(
        "--opposite-signal-action",
        type=str,
        default="close_and_reverse",
        choices=["close_only", "close_and_reverse"],
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _default_out_dir(args: argparse.Namespace) -> Path:
    symbol_slug = "_".join(args.symbols)
    return Path("results") / f"supertrend_portfolio_dev_{symbol_slug}"


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.strategy = "supertrend"
    args.timeframe_selection_mode = ATR50_HL_SUM_RATIO_MODE
    args.selection_metric = "max_recovery_time"
    args.opt_symbol = str(args.opt_symbol).upper()
    normalized = normalize_symbols(list(args.symbols))
    ordered: list[str] = []
    for symbol in [args.opt_symbol, *normalized]:
        if symbol not in ordered:
            ordered.append(symbol)
    args.symbols = ordered
    if args.out_dir is None:
        args.out_dir = _default_out_dir(args)
    return args


def _rebuild_worker_market() -> dict[str, object]:
    if sp._GRID_WORKER_MARKET_BY_SYMBOL is None or sp._GRID_WORKER_CONFIG is None:  # type: ignore[attr-defined]
        raise RuntimeError("Shared market context is not initialized")
    if sp._GRID_WORKER_MARKET_SHARED is None:  # type: ignore[attr-defined]
        return sp._GRID_WORKER_MARKET_BY_SYMBOL  # type: ignore[attr-defined]

    market_by_symbol: dict[str, object] = {}
    for symbol, bars in sp._GRID_WORKER_MARKET_BY_SYMBOL.items():  # type: ignore[attr-defined]
        shared = sp._GRID_WORKER_MARKET_SHARED[symbol]  # type: ignore[attr-defined]
        m1 = pd.DataFrame(
            shared["data"],
            columns=["bid_open", "bid_low", "bid_close", "ask_open", "ask_high", "ask_close"],
            index=pd.to_datetime(shared["ts_ns"], unit="ns", utc=True),
        )
        market_by_symbol[symbol] = type("SharedMarketSlice", (), {"m1": m1, "bars": bars})()
    return market_by_symbol


def _run_portfolio_grid_task(task: dict[str, Any]) -> dict[str, Any]:
    strategy = get_strategy(sp._GRID_WORKER_STRATEGY_NAME)  # type: ignore[attr-defined]
    params = strategy.make_params(atr_len=int(task["atr_len"]), atr_mult=float(task["atr_mult"]))
    markets = _rebuild_worker_market()
    result = run_portfolio_backtest(
        market_by_symbol=markets,
        params=params,
        strategy=strategy,
        config=sp._GRID_WORKER_CONFIG,  # type: ignore[attr-defined]
        show_progress=False,
        collect_events=False,
        portfolio_mode=sp._GRID_WORKER_PORTFOLIO_MODE,  # type: ignore[attr-defined]
        cash_per_trade=sp._GRID_WORKER_CASH_PER_TRADE,  # type: ignore[attr-defined]
        risk_per_trade=sp._GRID_WORKER_RISK_PER_TRADE,  # type: ignore[attr-defined]
        risk_per_trade_pct=sp._GRID_WORKER_RISK_PER_TRADE_PCT,  # type: ignore[attr-defined]
    )
    return {
        "strategy": strategy.name,
        "opt_symbol": task["opt_symbol"],
        "opt_timeframe": task["opt_timeframe"],
        "symbols": task["symbols"],
        "symbol_timeframes": task["symbol_timeframes"],
        "atr_len": int(task["atr_len"]),
        "atr_mult": float(task["atr_mult"]),
        **result["portfolio_stats"],
    }


def _build_grid_tasks(
    *,
    opt_symbol: str,
    opt_timeframe: str,
    symbols: list[str],
    symbol_timeframes: dict[str, str],
) -> list[dict[str, Any]]:
    strategy = get_strategy("supertrend")
    grid = strategy.default_grid()
    atr_lens = list(grid.get("atr_len", []))
    atr_mults = list(grid.get("atr_mult", []))
    if not atr_lens or not atr_mults:
        raise ValueError("Supertrend default_grid must include atr_len and atr_mult")
    return [
        {
            "opt_symbol": opt_symbol,
            "opt_timeframe": str(opt_timeframe),
            "symbols": list(symbols),
            "symbol_timeframes": dict(symbol_timeframes),
            "atr_len": int(atr_len),
            "atr_mult": float(atr_mult),
        }
        for atr_len in atr_lens
        for atr_mult in atr_mults
    ]


def _run_portfolio_grid(
    *,
    tasks: list[dict[str, Any]],
    markets: dict[str, object],
    args: argparse.Namespace,
    config: Any,
    progress_desc: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared_blocks = []
    progress = tqdm(total=len(tasks), desc=progress_desc)
    try:
        if args.max_workers == 1:
            sp._init_grid_worker(
                markets,
                config,
                "supertrend",
                args.portfolio_mode,
                args.cash_per_trade,
                args.risk_per_trade,
                args.risk_per_trade_pct,
            )
            for task in tasks:
                rows.append(_run_portfolio_grid_task(task))
                progress.update(1)
        else:
            market_shared, bars_by_symbol, shared_blocks = sp._build_shared_market_payload(markets)
            with ProcessPoolExecutor(
                max_workers=min(args.max_workers, len(tasks)),
                initializer=sp._init_grid_worker_shared,
                initargs=(
                    market_shared,
                    bars_by_symbol,
                    config,
                    "supertrend",
                    args.portfolio_mode,
                    args.cash_per_trade,
                    args.risk_per_trade,
                    args.risk_per_trade_pct,
                ),
            ) as executor:
                futures = [executor.submit(_run_portfolio_grid_task, task) for task in tasks]
                for future in as_completed(futures):
                    rows.append(future.result())
                    progress.update(1)
    finally:
        progress.close()
        _cleanup_shared_blocks(shared_blocks)
    return rows


def _best_record(results: pd.DataFrame, metric: str, *, ascending: bool) -> dict[str, Any] | None:
    if results.empty or metric not in results.columns:
        return None
    ranked = results.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked = ranked.dropna(subset=[metric])
    if ranked.empty:
        return None
    return ranked.sort_values(metric, ascending=ascending).iloc[0].to_dict()


def _timeframe_summary_payload(
    *,
    opt_timeframe: str,
    symbol_timeframes: dict[str, str],
    results: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "opt_timeframe": str(opt_timeframe),
        "symbol_timeframes": dict(symbol_timeframes),
        "n_runs": int(len(results)),
        "best_by_total_return": _best_record(results, "total_return", ascending=False),
        "best_by_calmar": _best_record(results, "calmar", ascending=False),
        "best_by_min_recovery_time": _best_record(results, "max_recovery_time", ascending=True),
    }


def _combined_inputs(out_dir: Path, timeframes: list[str]) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    for timeframe in sorted(timeframes, key=_timeframe_sort_key):
        results_path = out_dir / str(timeframe) / "heatmap_results.jsonl"
        if results_path.exists():
            inputs.append((str(timeframe), results_path))
    return inputs


def main() -> int:
    args = _normalize_args(_parse_args())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        args.out_dir / "run_args.json",
        {
            "strategy": args.strategy,
            "opt_symbol": args.opt_symbol,
            "symbols": args.symbols,
            "start": args.start,
            "end": args.end,
            "selection_start": args.selection_start,
            "selection_end": args.selection_end,
            "timeframe_candidates": args.timeframe_candidates,
            "timeframe_selection_mode": args.timeframe_selection_mode,
            "portfolio_mode": args.portfolio_mode,
            "cash_per_trade": args.cash_per_trade,
            "risk_per_trade": args.risk_per_trade,
            "risk_per_trade_pct": args.risk_per_trade_pct,
            "max_workers": args.max_workers,
            "opposite_signal_action": args.opposite_signal_action,
        },
    )

    symbol_specs = _load_symbol_specs(args.symbol_specs)
    fx_daily = _load_fx_daily(args.fx_daily)
    m1_by_symbol = _load_m1_cache(args)
    full_m1_window = _slice_m1_window(m1_by_symbol, args.start, args.end, copy=False)
    bars_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    shared_m1_by_symbol: dict[str, dict[str, Any]] | None = None
    shared_m1_blocks = []
    if args.max_workers > 1:
        shared_m1_by_symbol, shared_m1_blocks = _build_shared_m1_payload(m1_by_symbol)

    all_rows: list[dict[str, Any]] = []
    timeframe_summaries: list[dict[str, Any]] = []
    ordered_timeframes = sorted([str(value) for value in args.timeframe_candidates], key=_timeframe_sort_key)
    try:
        for opt_timeframe in tqdm(ordered_timeframes, desc="Scanning opt timeframes"):
            timeframe_dir = args.out_dir / str(opt_timeframe)
            timeframe_dir.mkdir(parents=True, exist_ok=True)
            selected_timeframes, timeframe_results = _resolve_timeframes_for_opt_timeframe(
                args=args,
                selection_mode=str(opt_timeframe),
                resolved_opt_timeframe=str(opt_timeframe),
                fixed_param_values={},
                m1_by_symbol=m1_by_symbol,
                shared_m1_by_symbol=shared_m1_by_symbol,
                symbol_specs=symbol_specs,
                fx_daily=fx_daily,
                bars_cache=bars_cache,
                config_kwargs={
                    "commission_bps": args.commission_bps,
                    "slippage_bps": args.slippage_bps,
                    "overnight_long_rate": args.overnight_long_rate,
                    "overnight_short_rate": args.overnight_short_rate,
                    "overnight_day_count": args.overnight_day_count,
                    "initial_equity": args.initial_equity,
                    "initial_margin_ratio": args.initial_margin_ratio,
                    "maintenance_margin_ratio": args.maintenance_margin_ratio,
                    "opposite_signal_action": args.opposite_signal_action,
                },
                portfolio_kwargs={
                    "portfolio_mode": args.portfolio_mode,
                    "cash_per_trade": args.cash_per_trade,
                    "risk_per_trade": args.risk_per_trade,
                    "risk_per_trade_pct": args.risk_per_trade_pct,
                },
                timeframe_chart_root=timeframe_dir / "timeframe_selection_charts",
            )
            final_timeframes = {symbol: str(opt_timeframe) for symbol in args.symbols}
            final_timeframes.update(selected_timeframes)
            _write_json(timeframe_dir / "symbol_timeframes.json", final_timeframes)
            if not timeframe_results.empty:
                timeframe_results = timeframe_results.copy()
                timeframe_results["opt_timeframe"] = str(opt_timeframe)
            _write_records_jsonl(timeframe_results, timeframe_dir / "timeframe_search_results.jsonl")

            markets = _build_markets_from_cache(
                full_m1_window,
                final_timeframes,
                bars_cache=bars_cache,
                phase=f"portfolio_dev_{opt_timeframe}",
            )
            config = _make_config(
                args,
                symbol_specs,
                fx_daily,
                final_timeframes,
                default_timeframe=str(opt_timeframe),
            )
            tasks = _build_grid_tasks(
                opt_symbol=args.opt_symbol,
                opt_timeframe=str(opt_timeframe),
                symbols=args.symbols,
                symbol_timeframes=final_timeframes,
            )
            rows = _run_portfolio_grid(
                tasks=tasks,
                markets=markets,
                args=args,
                config=config,
                progress_desc=f"Running portfolio grid [{opt_timeframe}]",
            )
            results = pd.DataFrame(rows).sort_values(["atr_len", "atr_mult"]).reset_index(drop=True)
            _write_records_jsonl(results, timeframe_dir / "heatmap_results.jsonl")
            summary = _timeframe_summary_payload(
                opt_timeframe=str(opt_timeframe),
                symbol_timeframes=final_timeframes,
                results=results,
            )
            _write_json(timeframe_dir / "summary.json", summary)
            timeframe_summaries.append(summary)
            all_rows.extend(rows)

        all_results = pd.DataFrame(all_rows)
        if not all_results.empty:
            sort_key_map = {timeframe: idx for idx, timeframe in enumerate(ordered_timeframes)}
            all_results["_opt_timeframe_order"] = all_results["opt_timeframe"].map(sort_key_map)
            all_results = all_results.sort_values(["_opt_timeframe_order", "atr_len", "atr_mult"]).drop(
                columns=["_opt_timeframe_order"]
            )
        _write_records_jsonl(all_results, args.out_dir / "all_results.jsonl")

        inputs = _combined_inputs(args.out_dir, ordered_timeframes)
        combined_outputs: dict[str, str] = {}
        title_symbol = ",".join(args.symbols)
        for metric, _ascending in HEATMAP_METRICS:
            out_path = args.out_dir / f"{metric}_combined.png"
            write_combined_heatmap(
                inputs,
                metric,
                out_path,
                symbol=f"{title_symbol} portfolio",
                strategy="supertrend",
            )
            combined_outputs[metric] = str(out_path)

        top_level_summary = {
            "strategy": args.strategy,
            "opt_symbol": args.opt_symbol,
            "symbols": args.symbols,
            "n_timeframes": int(len(ordered_timeframes)),
            "n_runs": int(len(all_results)),
            "timeframe_summaries": timeframe_summaries,
            "best_by_total_return": _best_record(all_results, "total_return", ascending=False),
            "best_by_calmar": _best_record(all_results, "calmar", ascending=False),
            "best_by_min_recovery_time": _best_record(all_results, "max_recovery_time", ascending=True),
            "combined_outputs": combined_outputs,
        }
        _write_json(args.out_dir / "summary.json", top_level_summary)
    finally:
        _cleanup_shared_blocks(shared_m1_blocks)

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "n_timeframes": len(ordered_timeframes),
                "n_runs": len(all_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
