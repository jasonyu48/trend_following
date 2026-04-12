from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from market_data import MarketDataSlice, load_symbol_m1_bid_ask, resample_mid_bars
from plot_top_param_distributions import (
    build_distributions,
    plot_distributions,
    select_param_values_by_1d_neighbor_median,
)
from run_backtest import (
    DEFAULT_SYMBOLS,
    build_engine_config,
    build_strategy_params,
    normalize_symbols,
    save_backtest_outputs,
    _load_fx_daily,
    _load_symbol_specs,
)
from search_params import add_neighbor_medians, run_grid_search, run_portfolio_backtest, score_grid_search_results
from strategies import STRATEGIES, get_strategy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTA workflow: optimize params on one symbol, pick timeframe per symbol, then run portfolio backtest.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--symbol-specs", type=Path, default=Path("symbol_specs.json"))
    p.add_argument("--fx-daily", type=Path, default=Path("data/fx_daily/fx_daily_2012_2025.csv"))
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--selection-start", type=str, default="20120101", help="start date used for parameter and timeframe selection")
    p.add_argument("--selection-end", type=str, default="20200101", help="end date used for parameter and timeframe selection")
    p.add_argument("--start", type=str, default="20120101", help="start date used for final portfolio backtest")
    p.add_argument("--end", type=str, default="20250101", help="end date used for final portfolio backtest")
    p.add_argument("--strategy", type=str, default="ma_atr_breakout", choices=sorted(STRATEGIES))
    p.add_argument("--opt-symbol", type=str, default="GBPJPY", help="symbol used for parameter search")
    p.add_argument("--opt-timeframe", type=str, default="4H", help="timeframe used during parameter search; use 'none' to include timeframe in parameter search")
    p.add_argument(
        "--skip-param-search",
        action="store_true",
        help="skip parameter search and use fixed ma_atr_breakout params before per-symbol timeframe selection",
    )
    p.add_argument("--fixed-ma-len", type=int, default=12, help="fixed ma_len used when --skip-param-search is enabled")
    p.add_argument("--fixed-atr-len", type=int, default=45, help="fixed atr_len used when --skip-param-search is enabled")
    p.add_argument("--fixed-atr-mult", type=float, default=2.2, help="fixed atr_mult used when --skip-param-search is enabled")
    p.add_argument(
        "--fixed-stop-lookback",
        type=int,
        default=30,
        help="fixed stop_lookback used when --skip-param-search is enabled",
    )
    p.add_argument(
        "--skip-timeframe-selection",
        action="store_true",
        help="skip per-symbol timeframe optimization and use the resolved opt timeframe for all symbols",
    )
    p.add_argument("--timeframe-candidates", nargs="+", default=["1H", "2H", "3H", "4H"])
    p.add_argument(
        "--neighbor-radius",
        type=int,
        default=2,
        help="taxi-cab distance used when computing neighborhood median scores during parameter search",
    )
    p.add_argument(
        "--selection-metric",
        type=str,
        default="calmar",
        choices=["annualized_return", "calmar", "sharpe", "total_return"],
        help="metric used to pick the best timeframe for each symbol",
    )
    p.add_argument(
        "--param-top-pct",
        type=float,
        default=0.2,
        help="top fraction of param_search_results used when plotting distributions and building the generalized candidate",
    )
    p.add_argument(
        "--param-distribution-chart-type",
        type=str,
        default="line",
        choices=["line", "bar"],
        help="chart style used for the top-parameter distribution plot",
    )
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
        default="fixed_risk_pct",
        choices=["fixed_cash", "fixed_risk", "fixed_risk_pct"],
    )
    p.add_argument("--cash-per-trade", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=100.0)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    p.add_argument("--out-dir", type=Path, default=Path("results/cta_workflow12_45_22_30"))
    return p.parse_args()


def _default_grid(strategy_name: str) -> dict[str, list]:
    return get_strategy(strategy_name).default_grid()


def _normalize_optional_timeframe(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_records_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in df.to_dict(orient="records"):
            fh.write(json.dumps(record, default=str))
            fh.write("\n")


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_m1_cache(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    load_start = min(str(args.selection_start), str(args.start))
    load_end = max(str(args.selection_end), str(args.end))
    for symbol in tqdm(args.symbols, desc="Loading M1 data"):
        cache[symbol] = load_symbol_m1_bid_ask(
            data_dir=args.data_dir,
            symbol=symbol,
            start=load_start,
            end=load_end,
        )
    return cache


def _slice_m1_window(
    m1_by_symbol: dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    for symbol, m1 in m1_by_symbol.items():
        out[symbol] = m1.loc[(m1.index >= start_ts) & (m1.index < end_ts)].copy()
    return out


def _build_markets_from_cache(
    m1_by_symbol: dict[str, pd.DataFrame],
    timeframe_by_symbol: dict[str, str],
    bars_cache: dict[tuple[str, str, str], pd.DataFrame] | None = None,
    phase: str = "default",
) -> dict[str, MarketDataSlice]:
    markets: dict[str, MarketDataSlice] = {}
    for symbol, m1 in m1_by_symbol.items():
        timeframe = timeframe_by_symbol[symbol]
        cache_key = (str(phase), str(symbol).upper(), str(timeframe))
        bars = None
        if bars_cache is not None:
            bars = bars_cache.get(cache_key)
        if bars is None:
            bars = resample_mid_bars(m1, timeframe=timeframe)
            if bars_cache is not None:
                bars_cache[cache_key] = bars
        markets[symbol] = MarketDataSlice(
            symbol=symbol,
            m1=m1,
            bars=bars,
        )
    return markets


def _make_config(
    args: argparse.Namespace,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    timeframe_by_symbol: dict[str, str],
    default_timeframe: str,
):
    return build_engine_config(
        default_timeframe=default_timeframe,
        timeframe_by_symbol=timeframe_by_symbol,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        overnight_long_rate=args.overnight_long_rate,
        overnight_short_rate=args.overnight_short_rate,
        overnight_day_count=args.overnight_day_count,
        initial_equity=args.initial_equity,
        initial_margin_ratio=args.initial_margin_ratio,
        maintenance_margin_ratio=args.maintenance_margin_ratio,
        symbol_specs=symbol_specs,
        fx_daily=fx_daily,
    )


def _extract_best_params(result: pd.DataFrame, strategy_name: str) -> tuple[object, dict[str, Any]]:
    if result.empty:
        raise ValueError("Parameter search returned no rows")
    strategy = get_strategy(strategy_name)
    best_row = result.iloc[0]
    param_values = {name: _jsonable_value(best_row[name]) for name in strategy.param_names}
    return build_strategy_params(strategy_name, param_values), param_values


def _param_values_from_selection(
    strategy_name: str,
    selected_param_values: dict[str, Any],
) -> dict[str, Any]:
    normalized = {name: _jsonable_value(value) for name, value in selected_param_values.items()}
    params = build_strategy_params(strategy_name, normalized)
    if is_dataclass(params):
        return {name: _jsonable_value(value) for name, value in asdict(params).items()}
    return normalized


def _mark_selected_param_rows(
    param_search: pd.DataFrame,
    primary_param_values: dict[str, Any],
    generalized_param_values: dict[str, Any] | None,
) -> pd.DataFrame:
    out = param_search.copy()
    out["is_primary_candidate"] = False
    if not out.empty:
        out.loc[out.index[0], "is_primary_candidate"] = True
    out["matches_generalized_candidate"] = False
    if generalized_param_values:
        mask = pd.Series(True, index=out.index, dtype=bool)
        for name, value in generalized_param_values.items():
            if name in out.columns:
                mask &= out[name].eq(value)
        out["matches_generalized_candidate"] = mask
    out["matches_primary_param_values"] = False
    if primary_param_values:
        mask = pd.Series(True, index=out.index, dtype=bool)
        for name, value in primary_param_values.items():
            if name in out.columns:
                mask &= out[name].eq(value)
        out["matches_primary_param_values"] = mask
    return out


def _generate_param_distribution_artifacts(
    *,
    args: argparse.Namespace,
    param_search: pd.DataFrame,
    strategy_name: str,
) -> dict[str, Any]:
    param_columns = list(_default_grid(strategy_name).keys())
    distribution_rows = param_search.to_dict(orient="records")
    _, summary = build_distributions(
        distribution_rows,
        top_pct=args.param_top_pct,
        params=param_columns,
    )
    selection = select_param_values_by_1d_neighbor_median(
        summary,
        neighbor_radius=args.neighbor_radius,
    )
    plot_distributions(
        summary,
        args.out_dir / "top20_param_distributions.png",
        args.param_distribution_chart_type,
        selected_values=selection["param_values"],
    )
    _write_json(args.out_dir / "top20_param_distributions_summary.json", summary)
    return summary


def _metric_value(stats: dict[str, Any], metric: str) -> float:
    value = float(stats.get(metric, np.nan))
    return value if np.isfinite(value) else float("-inf")


def _default_timeframe_for_workflow(args: argparse.Namespace) -> str:
    if args.opt_timeframe is not None:
        return str(args.opt_timeframe)
    if not args.timeframe_candidates:
        raise ValueError("At least one timeframe candidate is required")
    return str(args.timeframe_candidates[0])


def _fixed_param_values(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.strategy) != "ma_atr_breakout":
        raise ValueError("--skip-param-search currently supports only --strategy ma_atr_breakout")
    return {
        "ma_len": int(args.fixed_ma_len),
        "atr_len": int(args.fixed_atr_len),
        "atr_mult": float(args.fixed_atr_mult),
        "stop_lookback": int(args.fixed_stop_lookback),
        "ma_kind": "ema",
    }


def _run_param_search(
    args: argparse.Namespace,
    strategy: Any,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    selection_m1_by_symbol: dict[str, pd.DataFrame],
    bars_cache: dict[tuple[str, str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    if args.opt_timeframe is not None:
        opt_timeframes = {args.opt_symbol: args.opt_timeframe}
        opt_markets = _build_markets_from_cache(
            {args.opt_symbol: selection_m1_by_symbol[args.opt_symbol]},
            opt_timeframes,
            bars_cache=bars_cache,
            phase="selection",
        )
        opt_config = _make_config(args, symbol_specs, fx_daily, opt_timeframes, default_timeframe=args.opt_timeframe)
        result = run_grid_search(
            market_by_symbol=opt_markets,
            strategy=strategy,
            param_grid=_default_grid(args.strategy),
            config=opt_config,
            neighbor_radius=args.neighbor_radius,
            max_workers=args.max_workers,
            portfolio_mode=args.portfolio_mode,
            cash_per_trade=args.cash_per_trade,
            risk_per_trade=args.risk_per_trade,
            risk_per_trade_pct=args.risk_per_trade_pct,
        )
        result["timeframe"] = str(args.opt_timeframe)
        return score_grid_search_results(result, opt_symbol=args.opt_symbol), str(args.opt_timeframe)

    frames: list[pd.DataFrame] = []
    for timeframe in tqdm(args.timeframe_candidates, desc="Searching params across timeframes"):
        opt_timeframes = {args.opt_symbol: timeframe}
        opt_markets = _build_markets_from_cache(
            {args.opt_symbol: selection_m1_by_symbol[args.opt_symbol]},
            opt_timeframes,
            bars_cache=bars_cache,
            phase="selection",
        )
        opt_config = _make_config(args, symbol_specs, fx_daily, opt_timeframes, default_timeframe=timeframe)
        result = run_grid_search(
            market_by_symbol=opt_markets,
            strategy=strategy,
            param_grid=_default_grid(args.strategy),
            config=opt_config,
            neighbor_radius=args.neighbor_radius,
            max_workers=args.max_workers,
            portfolio_mode=args.portfolio_mode,
            cash_per_trade=args.cash_per_trade,
            risk_per_trade=args.risk_per_trade,
            risk_per_trade_pct=args.risk_per_trade_pct,
        )
        result["timeframe"] = str(timeframe)
        frames.append(result)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        raise ValueError("Parameter search returned no rows")
    combined = add_neighbor_medians(
        combined,
        param_columns=[*list(_default_grid(args.strategy).keys()), "timeframe"],
        radius=args.neighbor_radius,
    )
    combined = score_grid_search_results(combined, opt_symbol=args.opt_symbol)
    selected_timeframe = str(combined.iloc[0]["timeframe"])
    return combined, selected_timeframe


def _evaluate_symbol_timeframes(task: dict[str, Any]) -> dict[str, Any]:
    symbol = str(task["symbol"]).upper()
    timeframes = [str(tf) for tf in task["timeframes"]]
    m1 = task["m1"]
    symbol_specs = dict(task["symbol_specs"])
    fx_daily = task["fx_daily"]
    strategy_name = str(task["strategy_name"])
    fixed_param_values = dict(task["fixed_param_values"])
    selection_metric = str(task["selection_metric"])
    config_kwargs = dict(task["config_kwargs"])
    portfolio_kwargs = dict(task["portfolio_kwargs"])

    strategy = get_strategy(strategy_name)
    params = build_strategy_params(strategy_name, fixed_param_values)
    rows: list[dict[str, Any]] = []
    best_timeframe = None
    best_score = float("-inf")
    bars_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    for timeframe in timeframes:
        timeframe_by_symbol = {symbol: timeframe}
        single_markets = _build_markets_from_cache(
            {symbol: m1},
            timeframe_by_symbol,
            bars_cache=bars_cache,
            phase="selection",
        )
        single_config = build_engine_config(
            default_timeframe=timeframe,
            timeframe_by_symbol=timeframe_by_symbol,
            symbol_specs=symbol_specs,
            fx_daily=fx_daily,
            **config_kwargs,
        )
        result = run_portfolio_backtest(
            market_by_symbol=single_markets,
            params=params,
            strategy=strategy,
            config=single_config,
            show_progress=False,
            collect_events=False,
            **portfolio_kwargs,
        )
        stats = result["symbol_results"][symbol]["stats"]
        score = _metric_value(stats, selection_metric)
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "selection_metric": selection_metric,
                "selection_score": score,
                **stats,
            }
        )
        if score > best_score:
            best_score = score
            best_timeframe = timeframe

    if best_timeframe is None:
        raise ValueError(f"No valid timeframe found for symbol {symbol}")
    return {
        "symbol": symbol,
        "best_timeframe": best_timeframe,
        "best_score": best_score,
        "rows": rows,
    }


def _run_candidate_workflow(
    *,
    args: argparse.Namespace,
    selection_mode: str,
    fixed_params: object,
    fixed_param_values: dict[str, Any],
    resolved_opt_timeframe: str,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    selection_m1_by_symbol: dict[str, pd.DataFrame],
    final_m1_by_symbol: dict[str, pd.DataFrame],
    bars_cache: dict[tuple[str, str, str], pd.DataFrame],
    strategy: Any,
    config_kwargs: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timeframe_rows: list[dict[str, Any]] = []
    selected_timeframes: dict[str, str] = {}
    if args.skip_timeframe_selection:
        selected_timeframes = {symbol: str(resolved_opt_timeframe) for symbol in args.symbols}
        timeframe_results = pd.DataFrame()
    else:
        timeframe_tasks = [
            {
                "symbol": symbol,
                "timeframes": args.timeframe_candidates,
                "m1": selection_m1_by_symbol[symbol],
                "symbol_specs": symbol_specs,
                "fx_daily": fx_daily,
                "strategy_name": args.strategy,
                "fixed_param_values": fixed_param_values,
                "selection_metric": args.selection_metric,
                "config_kwargs": config_kwargs,
                "portfolio_kwargs": portfolio_kwargs,
            }
            for symbol in args.symbols
        ]
        if args.max_workers == 1:
            timeframe_results_raw = [
                _evaluate_symbol_timeframes(task)
                for task in tqdm(timeframe_tasks, desc=f"Selecting timeframes [{selection_mode}]")
            ]
        else:
            timeframe_results_raw = []
            with ProcessPoolExecutor(max_workers=min(args.max_workers, len(timeframe_tasks))) as executor:
                futures = [executor.submit(_evaluate_symbol_timeframes, task) for task in timeframe_tasks]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Selecting timeframes [{selection_mode}]",
                ):
                    timeframe_results_raw.append(future.result())
        for item in sorted(timeframe_results_raw, key=lambda x: args.symbols.index(x["symbol"])):
            selected_timeframes[item["symbol"]] = item["best_timeframe"]
            timeframe_rows.extend(item["rows"])

        timeframe_results = pd.DataFrame(timeframe_rows).sort_values(
            ["symbol", "selection_score", "annualized_return"],
            ascending=[True, False, False],
        )
    if not timeframe_results.empty:
        timeframe_results["selection_mode"] = selection_mode
    _write_records_jsonl(timeframe_results, out_dir / "timeframe_search_results.jsonl")

    final_timeframes = {symbol: str(resolved_opt_timeframe) for symbol in args.symbols}
    final_timeframes.update(selected_timeframes)
    final_markets = _build_markets_from_cache(
        final_m1_by_symbol,
        final_timeframes,
        bars_cache=bars_cache,
        phase=f"portfolio_{selection_mode}",
    )
    final_config = _make_config(args, symbol_specs, fx_daily, final_timeframes, default_timeframe=resolved_opt_timeframe)
    final_result = run_portfolio_backtest(
        market_by_symbol=final_markets,
        params=fixed_params,
        strategy=strategy,
        config=final_config,
        show_progress=True,
        progress_desc=f"Portfolio backtest [{selection_mode}]",
        portfolio_mode=args.portfolio_mode,
        cash_per_trade=args.cash_per_trade,
        risk_per_trade=args.risk_per_trade,
        risk_per_trade_pct=args.risk_per_trade_pct,
    )
    save_backtest_outputs(
        result=final_result,
        markets=final_markets,
        out_dir=out_dir,
        initial_equity=args.initial_equity,
    )
    return {
        "selection_mode": selection_mode,
        "output_dir": str(out_dir),
        "resolved_opt_timeframe": resolved_opt_timeframe,
        "fixed_params": asdict(fixed_params) if is_dataclass(fixed_params) else fixed_param_values,
        "selected_timeframes": final_timeframes,
        "portfolio_stats": final_result["portfolio_stats"],
    }


def main() -> int:
    args = _parse_args()
    args.symbols = normalize_symbols(args.symbols)
    args.opt_symbol = str(args.opt_symbol or args.symbols[0]).upper()
    args.opt_timeframe = _normalize_optional_timeframe(args.opt_timeframe)
    args.timeframe_candidates = [str(tf) for tf in args.timeframe_candidates]
    if args.opt_symbol not in args.symbols:
        raise ValueError(f"opt_symbol {args.opt_symbol!r} must be included in --symbols")
    if args.skip_timeframe_selection and args.opt_timeframe is None:
        raise ValueError("--skip-timeframe-selection requires --opt-timeframe to be a concrete timeframe")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    symbol_specs = _load_symbol_specs(args.symbol_specs)
    fx_daily = _load_fx_daily(args.fx_daily)
    m1_by_symbol = _load_m1_cache(args)
    selection_m1_by_symbol = _slice_m1_window(m1_by_symbol, args.selection_start, args.selection_end)
    final_m1_by_symbol = _slice_m1_window(m1_by_symbol, args.start, args.end)
    strategy = get_strategy(args.strategy)
    bars_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    if args.skip_param_search:
        fixed_param_values = _fixed_param_values(args)
        fixed_params = build_strategy_params(args.strategy, fixed_param_values)
        param_search = pd.DataFrame(
            [
                {
                    "selection_mode": "fixed_params",
                    "timeframe": np.nan,
                    **fixed_param_values,
                }
            ]
        )
        resolved_opt_timeframe = _default_timeframe_for_workflow(args)
        args.skip_timeframe_selection = False
        param_distribution_summary = None
        generalized_param_values = None
    else:
        param_search, resolved_opt_timeframe = _run_param_search(
            args=args,
            strategy=strategy,
            symbol_specs=symbol_specs,
            fx_daily=fx_daily,
            selection_m1_by_symbol=selection_m1_by_symbol,
            bars_cache=bars_cache,
        )
        fixed_params, fixed_param_values = _extract_best_params(param_search, args.strategy)
        param_distribution_summary = _generate_param_distribution_artifacts(
            args=args,
            param_search=param_search,
            strategy_name=args.strategy,
        )
        generalized_param_values = _param_values_from_selection(
            args.strategy,
            param_distribution_summary["selection"]["param_values"],
        )
    param_search = _mark_selected_param_rows(param_search, fixed_param_values, generalized_param_values)
    _write_records_jsonl(param_search, args.out_dir / "param_search_results.jsonl")

    config_kwargs = {
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "overnight_long_rate": args.overnight_long_rate,
        "overnight_short_rate": args.overnight_short_rate,
        "overnight_day_count": args.overnight_day_count,
        "initial_equity": args.initial_equity,
        "initial_margin_ratio": args.initial_margin_ratio,
        "maintenance_margin_ratio": args.maintenance_margin_ratio,
    }
    portfolio_kwargs = {
        "portfolio_mode": args.portfolio_mode,
        "cash_per_trade": args.cash_per_trade,
        "risk_per_trade": args.risk_per_trade,
        "risk_per_trade_pct": args.risk_per_trade_pct,
    }
    candidate_runs = [
        _run_candidate_workflow(
            args=args,
            selection_mode="best_ranked",
            fixed_params=fixed_params,
            fixed_param_values=fixed_param_values,
            resolved_opt_timeframe=resolved_opt_timeframe,
            symbol_specs=symbol_specs,
            fx_daily=fx_daily,
            selection_m1_by_symbol=selection_m1_by_symbol,
            final_m1_by_symbol=final_m1_by_symbol,
            bars_cache=bars_cache,
            strategy=strategy,
            config_kwargs=config_kwargs,
            portfolio_kwargs=portfolio_kwargs,
            out_dir=args.out_dir,
        )
    ]
    if generalized_param_values is not None:
        if generalized_param_values == fixed_param_values:
            candidate_runs.append(
                {
                    "selection_mode": "generalized_1d_neighbor",
                    "output_dir": str(args.out_dir),
                    "resolved_opt_timeframe": resolved_opt_timeframe,
                    "fixed_params": generalized_param_values,
                    "selected_timeframes": candidate_runs[0]["selected_timeframes"],
                    "portfolio_stats": candidate_runs[0]["portfolio_stats"],
                    "skipped_duplicate_run": True,
                    "same_as": "best_ranked",
                }
            )
        else:
            generalized_params = build_strategy_params(args.strategy, generalized_param_values)
            candidate_runs.append(
                _run_candidate_workflow(
                    args=args,
                    selection_mode="generalized_1d_neighbor",
                    fixed_params=generalized_params,
                    fixed_param_values=generalized_param_values,
                    resolved_opt_timeframe=resolved_opt_timeframe,
                    symbol_specs=symbol_specs,
                    fx_daily=fx_daily,
                    selection_m1_by_symbol=selection_m1_by_symbol,
                    final_m1_by_symbol=final_m1_by_symbol,
                    bars_cache=bars_cache,
                    strategy=strategy,
                    config_kwargs=config_kwargs,
                    portfolio_kwargs=portfolio_kwargs,
                    out_dir=args.out_dir / "generalized_1d_neighbor",
                )
            )

    primary_run = candidate_runs[0]

    summary = {
        "symbols": args.symbols,
        "strategy": args.strategy,
        "opt_symbol": args.opt_symbol,
        "opt_timeframe": args.opt_timeframe,
        "resolved_opt_timeframe": resolved_opt_timeframe,
        "param_search_includes_timeframe": bool((args.opt_timeframe is None) and not args.skip_param_search),
        "skip_param_search": bool(args.skip_param_search),
        "skip_timeframe_selection": bool(args.skip_timeframe_selection),
        "selection_start": args.selection_start,
        "selection_end": args.selection_end,
        "portfolio_start": args.start,
        "portfolio_end": args.end,
        "timeframe_candidates": args.timeframe_candidates,
        "neighbor_radius": args.neighbor_radius,
        "selection_metric": args.selection_metric,
        "param_top_pct": args.param_top_pct,
        "param_distribution_chart_type": args.param_distribution_chart_type,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "overnight_long_rate": args.overnight_long_rate,
        "overnight_short_rate": args.overnight_short_rate,
        "overnight_day_count": args.overnight_day_count,
        "initial_equity": args.initial_equity,
        "initial_margin_ratio": args.initial_margin_ratio,
        "maintenance_margin_ratio": args.maintenance_margin_ratio,
        "fixed_params": primary_run["fixed_params"],
        "selected_timeframes": primary_run["selected_timeframes"],
        "portfolio_mode": args.portfolio_mode,
        "cash_per_trade": args.cash_per_trade,
        "risk_per_trade": args.risk_per_trade,
        "risk_per_trade_pct": args.risk_per_trade_pct,
        "portfolio_stats": primary_run["portfolio_stats"],
        "primary_selection_mode": primary_run["selection_mode"],
        "candidate_runs": candidate_runs,
        "generalized_candidate_params": generalized_param_values,
        "param_distribution_plot": (
            str(args.out_dir / "top20_param_distributions.png")
            if param_distribution_summary is not None
            else None
        ),
        "param_distribution_summary": (
            str(args.out_dir / "top20_param_distributions_summary.json")
            if param_distribution_summary is not None
            else None
        ),
        "param_distribution_selection": (
            param_distribution_summary.get("selection")
            if param_distribution_summary is not None
            else None
        ),
    }
    _write_json(args.out_dir / "workflow_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
