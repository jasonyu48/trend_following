from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
import json
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data_paths import DEFAULT_DATA_DIR
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from execution_engine import _bars_per_year
from indicators import average_true_range, bar_performance_stats, resampled_bar_performance_stats
from market_data import MarketDataSlice, load_symbol_m1_bid_ask, resample_mid_bars
from plot_top_param_distributions import (
    build_distributions,
    plot_distributions,
    select_param_values_by_top_count,
)
from run_backtest import (
    DEFAULT_SYMBOLS,
    _save_portfolio_pnl_plot,
    build_engine_config,
    build_strategy_params,
    load_symbol_timeframes,
    normalize_symbols,
    save_backtest_outputs,
    _load_fx_daily,
    _load_symbol_specs,
)
from search_params import add_neighbor_means, run_grid_search, run_portfolio_backtest, score_grid_search_results
from strategies import STRATEGIES, get_strategy

_WORKFLOW_LOCAL_M1_BY_SYMBOL: dict[str, pd.DataFrame] | None = None
_WORKFLOW_SHARED_M1_BY_SYMBOL: dict[str, dict[str, Any]] | None = None
_WORKFLOW_SHARED_HANDLES: list[SharedMemory] = []
_WORKFLOW_SYMBOL_SPECS: dict[str, dict[str, object]] | None = None
_WORKFLOW_FX_DAILY: pd.DataFrame | None = None
_WORKFLOW_CONFIG_KWARGS: dict[str, Any] | None = None
_WORKFLOW_PORTFOLIO_KWARGS: dict[str, Any] | None = None
ATR50_HL_SUM_RATIO_MODE = "atr50_hl_sum_ratio_median_match"
MATCHING_TIMEFRAME_SELECTION_MODES = {
    "ptr20_median_match",
    ATR50_HL_SUM_RATIO_MODE,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CTA workflow: choose strategy params from the opt symbol, assign a timeframe per symbol, "
            "then run the portfolio backtest."
        )
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--symbol-specs", type=Path, default=Path("symbol_specs.json"))
    p.add_argument("--fx-daily", type=Path, default=Path("data/fx_daily/fx_daily_2012_2025.csv"))
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--selection-start", type=str, default="20120101", help="start date used for parameter and timeframe selection")
    p.add_argument("--selection-end", type=str, default="20200101", help="end date used for parameter and timeframe selection")
    p.add_argument("--start", type=str, default="20120101", help="start date used for final portfolio backtest")
    p.add_argument("--end", type=str, default="20250101", help="end date used for final portfolio backtest")
    p.add_argument("--strategy", type=str, default="supertrend", choices=sorted(STRATEGIES))
    p.add_argument("--opt-symbol", type=str, default="XAUUSD", help="reference symbol used for parameter search and timeframe matching")
    p.add_argument(
        "--opt-timeframe",
        type=str,
        default="4H",
        help="reference timeframe for the opt symbol; use 'none' to include timeframe in parameter search",
    )
    p.add_argument(
        "--symbol-timeframes",
        type=Path,
        default=None,
        help="optional JSON mapping of symbol -> timeframe to override the selected per-symbol timeframes",
    )
    p.add_argument(
        "--skip-param-search",
        action="store_true",
        help="skip parameter search and use the fixed strategy params before per-symbol timeframe selection",
    )
    p.add_argument("--fixed-ma-len", type=int, default=10, help="fixed ma_len used when --skip-param-search is enabled")
    p.add_argument("--fixed-atr-len", type=int, default=30, help="fixed atr_len used when --skip-param-search is enabled")
    p.add_argument("--fixed-atr-mult", type=float, default=2.5, help="fixed atr_mult used when --skip-param-search is enabled")
    p.add_argument(
        "--fixed-stop-lookback",
        type=int,
        default=30,
        help="fixed stop_lookback used when --skip-param-search is enabled",
    )
    p.add_argument(
        "--skip-timeframe-selection",
        action="store_true",
        help="skip per-symbol timeframe selection and use the resolved opt timeframe for all symbols",
    )
    p.add_argument(
        "--timeframe-selection-mode",
        type=str,
        default="atr50_hl_sum_ratio_median_match",
        choices=["backtest_metric", "ptr20_median_match", ATR50_HL_SUM_RATIO_MODE],
        help=(
            "how to choose per-symbol timeframes: backtest_metric optimizes a metric, "
            "ptr20_median_match matches median PTR(20) vs the opt symbol, "
            f"{ATR50_HL_SUM_RATIO_MODE} matches median ATR(50)/(high+low) vs the opt symbol"
        ),
    )
    p.add_argument("--timeframe-candidates", nargs="+", default=["1H", "2H", "3H", "4H", "6H", "8H", "12H"])
    p.add_argument(
        "--neighbor-radius",
        type=int,
        default=1,
        help="taxi-cab distance used when computing neighborhood mean scores during parameter search",
    )
    p.add_argument(
        "--selection-metric",
        type=str,
        default="max_recovery_time",
        choices=["annualized_return", "calmar", "sharpe", "total_return", "max_recovery_time"],
        help="metric used to rank candidate timeframes when --timeframe-selection-mode=backtest_metric",
    )
    p.add_argument(
        "--param-top-pct",
        type=float,
        default=0.1,
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
    p.add_argument("--cash-per-trade", type=float, default=100000.0)
    p.add_argument("--risk-per-trade", type=float, default=10000.0)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.03)
    p.add_argument(
        "--opposite-signal-action",
        type=str,
        default="close_and_reverse",
        choices=["close_only", "close_and_reverse"],
        help="when an opposite signal appears while already in a position, either just close or close and reverse",
    )
    p.add_argument("--out-dir", type=Path, default=Path("results/cta_workflow43_supertrend_4h_30_25_3pct"))
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


def _load_symbol_timeframes_payload(symbol_timeframes_path: Path | None) -> dict[str, str] | None:
    if symbol_timeframes_path is None:
        return None
    payload = json.loads(symbol_timeframes_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in symbol timeframe file: {symbol_timeframes_path}")
    return {str(symbol).upper(): str(timeframe) for symbol, timeframe in payload.items()}


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


def _slice_m1_frame(
    m1: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    copy: bool,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    window = m1.loc[(m1.index >= start_ts) & (m1.index < end_ts)]
    return window.copy() if copy else window


def _slice_m1_window(
    m1_by_symbol: dict[str, pd.DataFrame],
    start: str,
    end: str,
    *,
    copy: bool = True,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, m1 in m1_by_symbol.items():
        out[symbol] = _slice_m1_frame(m1, start, end, copy=copy)
    return out


def _create_shared_block(array: np.ndarray) -> tuple[SharedMemory, dict[str, Any]]:
    contiguous = np.ascontiguousarray(array)
    shm = SharedMemory(create=True, size=contiguous.nbytes)
    view = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
    view[:] = contiguous
    return shm, {"name": shm.name, "shape": contiguous.shape, "dtype": str(contiguous.dtype)}


def _build_shared_m1_payload(
    m1_by_symbol: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, Any]], list[SharedMemory]]:
    payload: dict[str, dict[str, Any]] = {}
    blocks: list[SharedMemory] = []
    for symbol, m1 in m1_by_symbol.items():
        idx_shm, idx_meta = _create_shared_block(m1.index.asi8.astype("int64", copy=False))
        data_matrix = m1.to_numpy(dtype="float64", copy=False)
        data_shm, data_meta = _create_shared_block(data_matrix)
        blocks.extend([idx_shm, data_shm])
        payload[str(symbol).upper()] = {
            "index": idx_meta,
            "data": data_meta,
            "cols": list(m1.columns),
        }
    return payload, blocks


def _init_workflow_context(
    *,
    local_m1_by_symbol: dict[str, pd.DataFrame] | None,
    shared_m1_by_symbol: dict[str, dict[str, Any]] | None,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    config_kwargs: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
) -> None:
    global _WORKFLOW_LOCAL_M1_BY_SYMBOL, _WORKFLOW_SHARED_M1_BY_SYMBOL, _WORKFLOW_SHARED_HANDLES
    global _WORKFLOW_SYMBOL_SPECS, _WORKFLOW_FX_DAILY, _WORKFLOW_CONFIG_KWARGS, _WORKFLOW_PORTFOLIO_KWARGS
    _WORKFLOW_LOCAL_M1_BY_SYMBOL = local_m1_by_symbol
    _WORKFLOW_SHARED_M1_BY_SYMBOL = shared_m1_by_symbol
    _WORKFLOW_SYMBOL_SPECS = symbol_specs
    _WORKFLOW_FX_DAILY = fx_daily
    _WORKFLOW_CONFIG_KWARGS = dict(config_kwargs)
    _WORKFLOW_PORTFOLIO_KWARGS = dict(portfolio_kwargs)
    _WORKFLOW_SHARED_HANDLES = []


def _init_workflow_worker_local(
    m1_by_symbol: dict[str, pd.DataFrame],
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    config_kwargs: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
) -> None:
    _init_workflow_context(
        local_m1_by_symbol=m1_by_symbol,
        shared_m1_by_symbol=None,
        symbol_specs=symbol_specs,
        fx_daily=fx_daily,
        config_kwargs=config_kwargs,
        portfolio_kwargs=portfolio_kwargs,
    )


def _init_workflow_worker_shared(
    shared_m1_by_symbol: dict[str, dict[str, Any]],
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    config_kwargs: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
) -> None:
    _init_workflow_context(
        local_m1_by_symbol=None,
        shared_m1_by_symbol={},
        symbol_specs=symbol_specs,
        fx_daily=fx_daily,
        config_kwargs=config_kwargs,
        portfolio_kwargs=portfolio_kwargs,
    )
    if _WORKFLOW_SHARED_M1_BY_SYMBOL is None:
        raise RuntimeError("Workflow shared context did not initialize")
    for symbol, meta in shared_m1_by_symbol.items():
        idx_shm = SharedMemory(name=meta["index"]["name"])
        data_shm = SharedMemory(name=meta["data"]["name"])
        _WORKFLOW_SHARED_HANDLES.extend([idx_shm, data_shm])
        _WORKFLOW_SHARED_M1_BY_SYMBOL[symbol] = {
            "ts_ns": np.ndarray(meta["index"]["shape"], dtype=np.dtype(meta["index"]["dtype"]), buffer=idx_shm.buf),
            "data": np.ndarray(meta["data"]["shape"], dtype=np.dtype(meta["data"]["dtype"]), buffer=data_shm.buf),
            "cols": list(meta["cols"]),
        }


def _workflow_m1_window(symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    symbol_key = str(symbol).upper()
    if _WORKFLOW_LOCAL_M1_BY_SYMBOL is not None:
        return _slice_m1_frame(_WORKFLOW_LOCAL_M1_BY_SYMBOL[symbol_key], start, end, copy=False)
    if _WORKFLOW_SHARED_M1_BY_SYMBOL is None:
        raise RuntimeError("Workflow M1 context is not initialized")
    shared = _WORKFLOW_SHARED_M1_BY_SYMBOL[symbol_key]
    ts_ns = shared["ts_ns"]
    data = shared["data"]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    start_ns = start_ts.value
    end_ns = end_ts.value
    left = int(np.searchsorted(ts_ns, start_ns, side="left"))
    right = int(np.searchsorted(ts_ns, end_ns, side="left"))
    return pd.DataFrame(
        data[left:right],
        columns=shared["cols"],
        index=pd.to_datetime(ts_ns[left:right], unit="ns", utc=True),
    )


def _cleanup_shared_blocks(blocks: list[SharedMemory]) -> None:
    for shm in blocks:
        try:
            shm.close()
        except FileNotFoundError:
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


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
        opposite_signal_action=args.opposite_signal_action,
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
    if args.opt_timeframe is None and "timeframe" in param_search.columns:
        param_columns.append("timeframe")
    distribution_rows = param_search.to_dict(orient="records")
    _, summary = build_distributions(
        distribution_rows,
        top_pct=args.param_top_pct,
        params=param_columns,
    )
    selection = select_param_values_by_top_count(summary)
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
    return value if np.isfinite(value) else np.nan


def _metric_lower_is_better(metric: str) -> bool:
    return str(metric) in {"max_recovery_time"}


def _metric_score_from_value(value: float, metric: str) -> float:
    if not np.isfinite(value):
        return float("-inf")
    return float(-value) if _metric_lower_is_better(metric) else float(value)


def _metric_score(stats: dict[str, Any], metric: str) -> float:
    return _metric_score_from_value(_metric_value(stats, metric), metric)


def _metric_comparison_outcome(selected_value: float, baseline_value: float, metric: str) -> str:
    selected_finite = np.isfinite(selected_value)
    baseline_finite = np.isfinite(baseline_value)
    if not selected_finite and not baseline_finite:
        return "unknown"
    if selected_finite and not baseline_finite:
        return "selected_better"
    if baseline_finite and not selected_finite:
        return "opt_timeframe_better"
    selected_score = _metric_score_from_value(float(selected_value), metric)
    baseline_score = _metric_score_from_value(float(baseline_value), metric)
    if np.isclose(selected_score, baseline_score):
        return "tie"
    return "selected_better" if selected_score > baseline_score else "opt_timeframe_better"


def _utc_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _window_mask(
    timestamps: pd.Series | pd.Index,
    *,
    period_name: str,
    portfolio_start: pd.Timestamp,
    portfolio_end: pd.Timestamp,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
) -> np.ndarray:
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    full_mask = (ts >= portfolio_start) & (ts < portfolio_end)
    selection_mask = (ts >= selection_start) & (ts < selection_end)
    if period_name == "in_sample":
        return np.asarray(full_mask & selection_mask, dtype=bool)
    if period_name == "out_of_sample":
        return np.asarray(full_mask & ~selection_mask, dtype=bool)
    if period_name == "full_sample":
        return np.asarray(full_mask, dtype=bool)
    raise ValueError(f"Unsupported period_name: {period_name}")


def _slice_rows_for_period(
    frame: pd.DataFrame,
    time_col: str,
    *,
    period_name: str,
    portfolio_start: pd.Timestamp,
    portfolio_end: pd.Timestamp,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty or time_col not in frame.columns:
        return frame.iloc[0:0].copy()
    mask = _window_mask(
        frame[time_col],
        period_name=period_name,
        portfolio_start=portfolio_start,
        portfolio_end=portfolio_end,
        selection_start=selection_start,
        selection_end=selection_end,
    )
    return frame.loc[mask].copy().reset_index(drop=True)


def _elapsed_years_from_timestamps(timestamps: pd.Series | pd.Index) -> float:
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).sort_values()
    if len(ts) == 0:
        return np.nan
    if len(ts) == 1:
        return 1.0 / 365.25
    elapsed_days = float((ts[-1] - ts[0]) / pd.Timedelta(days=1))
    if not np.isfinite(elapsed_days) or elapsed_days <= 0.0:
        return 1.0 / 365.25
    return max(elapsed_days / 365.25, 1.0 / 365.25)


def _requested_stat_block(stats: dict[str, Any], timestamps: pd.Series | pd.Index) -> dict[str, float]:
    total_return = float(stats.get("total_return", np.nan))
    annualized_return = float(stats.get("annualized_return", np.nan))
    sharpe = float(stats.get("sharpe", np.nan))
    max_drawdown = float(stats.get("max_drawdown", np.nan))
    calmar = float(stats.get("calmar", np.nan))
    max_recovery_time = float(stats.get("max_recovery_time", np.nan))
    elapsed_years = _elapsed_years_from_timestamps(timestamps)
    simple_annualized_return = (
        float(total_return / elapsed_years)
        if np.isfinite(total_return) and np.isfinite(elapsed_years) and elapsed_years > 0.0
        else np.nan
    )
    simple_annualized_return_over_max_drawdown = (
        float(simple_annualized_return / abs(max_drawdown))
        if np.isfinite(simple_annualized_return) and np.isfinite(max_drawdown) and max_drawdown < 0.0
        else np.nan
    )
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "max_recovery_time": max_recovery_time,
        "simple_annualized_return": simple_annualized_return,
        "simple_annualized_return_over_max_drawdown": simple_annualized_return_over_max_drawdown,
        "elapsed_years": float(elapsed_years) if np.isfinite(elapsed_years) else np.nan,
    }


def _period_elapsed_years(
    *,
    period_name: str,
    portfolio_start: pd.Timestamp,
    portfolio_end: pd.Timestamp,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
) -> float:
    if period_name == "in_sample":
        start = max(portfolio_start, selection_start)
        end = min(portfolio_end, selection_end)
        elapsed_days = float((end - start) / pd.Timedelta(days=1))
    elif period_name == "out_of_sample":
        full_days = float((portfolio_end - portfolio_start) / pd.Timedelta(days=1))
        overlap_start = max(portfolio_start, selection_start)
        overlap_end = min(portfolio_end, selection_end)
        overlap_days = float((overlap_end - overlap_start) / pd.Timedelta(days=1)) if overlap_end > overlap_start else 0.0
        elapsed_days = full_days - max(overlap_days, 0.0)
    elif period_name == "full_sample":
        elapsed_days = float((portfolio_end - portfolio_start) / pd.Timedelta(days=1))
    else:
        raise ValueError(f"Unsupported period_name: {period_name}")
    if not np.isfinite(elapsed_days) or elapsed_days <= 0.0:
        return 1.0 / 365.25
    return max(elapsed_days / 365.25, 1.0 / 365.25)


def _trade_timing_metrics_for_period(
    trades: pd.DataFrame,
    *,
    period_name: str,
    portfolio_start: pd.Timestamp,
    portfolio_end: pd.Timestamp,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
) -> dict[str, float]:
    period_trades = _slice_rows_for_period(
        trades,
        "entry_time",
        period_name=period_name,
        portfolio_start=portfolio_start,
        portfolio_end=portfolio_end,
        selection_start=selection_start,
        selection_end=selection_end,
    )
    elapsed_years = _period_elapsed_years(
        period_name=period_name,
        portfolio_start=portfolio_start,
        portfolio_end=portfolio_end,
        selection_start=selection_start,
        selection_end=selection_end,
    )
    n_entries = int(len(period_trades))
    holding_hours = pd.Series(dtype="float64")
    if not period_trades.empty and {"entry_time", "exit_time"}.issubset(period_trades.columns):
        holding_hours = (
            (pd.to_datetime(period_trades["exit_time"], utc=True) - pd.to_datetime(period_trades["entry_time"], utc=True))
            / pd.Timedelta(hours=1)
        )
        holding_hours = pd.to_numeric(holding_hours, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    breakout_frequency_per_year = float(n_entries / elapsed_years) if elapsed_years > 0.0 else np.nan
    return {
        "n_entries": float(n_entries),
        "elapsed_years": float(elapsed_years),
        "breakout_frequency_per_year": breakout_frequency_per_year,
        "median_holding_hours": float(holding_hours.median()) if not holding_hours.empty else np.nan,
        "mean_holding_hours": float(holding_hours.mean()) if not holding_hours.empty else np.nan,
    }


def _trade_timing_alignment(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    breakout_frequency = float(candidate.get("breakout_frequency_per_year", np.nan))
    reference_breakout_frequency = float(reference.get("breakout_frequency_per_year", np.nan))
    median_holding_hours = float(candidate.get("median_holding_hours", np.nan))
    reference_median_holding_hours = float(reference.get("median_holding_hours", np.nan))
    mean_holding_hours = float(candidate.get("mean_holding_hours", np.nan))
    reference_mean_holding_hours = float(reference.get("mean_holding_hours", np.nan))
    return {
        "breakout_frequency_per_year_abs_diff": (
            float(abs(breakout_frequency - reference_breakout_frequency))
            if np.isfinite(breakout_frequency) and np.isfinite(reference_breakout_frequency)
            else np.nan
        ),
        "median_holding_hours_abs_diff": (
            float(abs(median_holding_hours - reference_median_holding_hours))
            if np.isfinite(median_holding_hours) and np.isfinite(reference_median_holding_hours)
            else np.nan
        ),
        "mean_holding_hours_abs_diff": (
            float(abs(mean_holding_hours - reference_mean_holding_hours))
            if np.isfinite(mean_holding_hours) and np.isfinite(reference_mean_holding_hours)
            else np.nan
        ),
    }


def _compute_period_stats(
    *,
    args: argparse.Namespace,
    result: dict[str, Any],
    selected_timeframes: dict[str, str],
) -> dict[str, Any]:
    portfolio_start = _utc_timestamp(args.start)
    portfolio_end = _utc_timestamp(args.end)
    selection_start = _utc_timestamp(args.selection_start)
    selection_end = _utc_timestamp(args.selection_end)
    period_rules = {
        "in_sample": "bars in [selection_start, selection_end)",
        "out_of_sample": "bars in [portfolio_start, portfolio_end) excluding [selection_start, selection_end)",
        "full_sample": "bars in [portfolio_start, portfolio_end)",
    }
    payload: dict[str, Any] = {}

    for period_name, rule in period_rules.items():
        portfolio_bars = _slice_rows_for_period(
            result["portfolio_bars"],
            "bar_end",
            period_name=period_name,
            portfolio_start=portfolio_start,
            portfolio_end=portfolio_end,
            selection_start=selection_start,
            selection_end=selection_end,
        )
        portfolio_stats = resampled_bar_performance_stats(
            portfolio_bars["equity"] if not portfolio_bars.empty else pd.Series(dtype="float64"),
            portfolio_bars["bar_end"] if not portfolio_bars.empty else pd.DatetimeIndex([]),
            resample_freq="1D",
            bars_per_year=252,
        )
        symbols_payload: dict[str, Any] = {}
        for symbol, symbol_result in result["symbol_results"].items():
            symbol_bars = _slice_rows_for_period(
                symbol_result["bars"],
                "bar_end",
                period_name=period_name,
                portfolio_start=portfolio_start,
                portfolio_end=portfolio_end,
                selection_start=selection_start,
                selection_end=selection_end,
            )
            symbol_stats = bar_performance_stats(
                symbol_bars["equity"] if not symbol_bars.empty else pd.Series(dtype="float64"),
                timestamps=symbol_bars["bar_end"] if not symbol_bars.empty else pd.DatetimeIndex([]),
                bars_per_year=_bars_per_year(selected_timeframes[symbol]),
            )
            symbols_payload[symbol] = {
                "timeframe": selected_timeframes[symbol],
                "n_bars": int(len(symbol_bars)),
                "stats": _requested_stat_block(
                    symbol_stats,
                    symbol_bars["bar_end"] if not symbol_bars.empty else pd.DatetimeIndex([]),
                ),
            }

        payload[period_name] = {
            "rule": rule,
            "portfolio_start": str(portfolio_start),
            "portfolio_end": str(portfolio_end),
            "selection_start": str(selection_start),
            "selection_end": str(selection_end),
            "portfolio_n_bars": int(len(portfolio_bars)),
            "portfolio_stats": _requested_stat_block(
                portfolio_stats,
                portfolio_bars["bar_end"] if not portfolio_bars.empty else pd.DatetimeIndex([]),
            ),
            "symbol_stats": symbols_payload,
        }
    return payload


def _run_single_symbol_result(
    *,
    symbol: str,
    timeframe: str,
    m1: pd.DataFrame,
    strategy: Any,
    fixed_params: object,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    config_kwargs: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
    bars_cache: dict[tuple[str, str, str], pd.DataFrame],
    phase: str,
) -> dict[str, Any]:
    timeframe_by_symbol = {symbol: timeframe}
    single_markets = _build_markets_from_cache(
        {symbol: m1},
        timeframe_by_symbol,
        bars_cache=bars_cache,
        phase=phase,
    )
    single_config = build_engine_config(
        default_timeframe=timeframe,
        timeframe_by_symbol=timeframe_by_symbol,
        symbol_specs=symbol_specs,
        fx_daily=fx_daily,
        **config_kwargs,
    )
    return run_portfolio_backtest(
        market_by_symbol=single_markets,
        params=fixed_params,
        strategy=strategy,
        config=single_config,
        show_progress=False,
        collect_events=False,
        **portfolio_kwargs,
    )


def _save_validation_single_symbol_pnl(
    result: dict[str, Any],
    output_dir: Path,
    initial_equity: float,
) -> str | None:
    portfolio_bars = result.get("portfolio_bars")
    if not isinstance(portfolio_bars, pd.DataFrame) or portfolio_bars.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_portfolio_pnl_plot(portfolio_bars, output_dir, initial_equity=float(initial_equity))
    return str(output_dir / "portfolio_pnl.png")


def _save_timeframe_selection_single_symbol_pnl(
    result: dict[str, Any],
    output_dir: Path,
    initial_equity: float,
) -> tuple[str | None, str | None]:
    portfolio_bars = result.get("portfolio_bars")
    if not isinstance(portfolio_bars, pd.DataFrame) or portfolio_bars.empty:
        return None, None
    output_dir.mkdir(parents=True, exist_ok=True)
    bars_path = output_dir / "portfolio_bars.jsonl"
    _write_records_jsonl(portfolio_bars, bars_path)
    _save_portfolio_pnl_plot(portfolio_bars, output_dir, initial_equity=float(initial_equity))
    return str(output_dir / "portfolio_pnl.png"), str(bars_path)


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


def _timeframe_selection_progress_desc(args: argparse.Namespace, selection_mode: str) -> str:
    verb = "Matching" if args.timeframe_selection_mode in MATCHING_TIMEFRAME_SELECTION_MODES else "Selecting"
    return f"{verb} timeframes [{selection_mode}]"


def _compute_ptr20_series(bars: pd.DataFrame) -> pd.Series:
    if bars.empty:
        return pd.Series(index=bars.index, dtype="float64")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    ptr = tr / close.replace(0.0, np.nan)
    ptr = ptr.replace([np.inf, -np.inf], np.nan)
    return ptr.rolling(20, min_periods=20).mean()


def _median_ptr20_from_bars(bars: pd.DataFrame) -> tuple[float | None, int]:
    ptr20 = _compute_ptr20_series(bars).dropna()
    if ptr20.empty:
        return None, 0
    return float(ptr20.median()), int(ptr20.shape[0])


def _compute_atr50_range_ratio_series(bars: pd.DataFrame) -> pd.Series:
    if bars.empty:
        return pd.Series(index=bars.index, dtype="float64")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    bar_sum = (high + low).replace(0.0, np.nan)
    atr50 = average_true_range(bars, length=50)
    ratio = atr50 / bar_sum
    return ratio.replace([np.inf, -np.inf], np.nan)


def _median_atr50_range_ratio_from_bars(bars: pd.DataFrame) -> tuple[float | None, int]:
    ratio = _compute_atr50_range_ratio_series(bars).dropna()
    if ratio.empty:
        return None, 0
    return float(ratio.median()), int(ratio.shape[0])


def _build_ptr20_target(
    *,
    opt_symbol: str,
    resolved_opt_timeframe: str,
    selection_start: str,
    selection_end: str,
    m1_by_symbol: dict[str, pd.DataFrame],
    bars_cache: dict[tuple[str, str, str], pd.DataFrame],
) -> dict[str, Any]:
    opt_symbol = str(opt_symbol).upper()
    selection_m1 = {opt_symbol: _slice_m1_frame(m1_by_symbol[opt_symbol], selection_start, selection_end, copy=False)}
    target_market = _build_markets_from_cache(
        selection_m1,
        {opt_symbol: str(resolved_opt_timeframe)},
        bars_cache=bars_cache,
        phase=f"ptr20_target_{opt_symbol}",
    )
    bars = target_market[opt_symbol].bars
    ptr20_median, ptr20_count = _median_ptr20_from_bars(bars)
    if ptr20_median is None:
        raise ValueError(
            f"Unable to compute PTR(20) target for {opt_symbol} at {resolved_opt_timeframe} "
            f"within [{selection_start}, {selection_end})"
        )
    return {
        "symbol": opt_symbol,
        "timeframe": str(resolved_opt_timeframe),
        "target_ptr20_median": float(ptr20_median),
        "ptr20_count": int(ptr20_count),
        "n_bars": int(len(bars)),
    }


def _build_atr50_range_ratio_target(
    *,
    opt_symbol: str,
    resolved_opt_timeframe: str,
    selection_start: str,
    selection_end: str,
    m1_by_symbol: dict[str, pd.DataFrame],
    bars_cache: dict[tuple[str, str, str], pd.DataFrame],
) -> dict[str, Any]:
    opt_symbol = str(opt_symbol).upper()
    selection_m1 = {opt_symbol: _slice_m1_frame(m1_by_symbol[opt_symbol], selection_start, selection_end, copy=False)}
    target_market = _build_markets_from_cache(
        selection_m1,
        {opt_symbol: str(resolved_opt_timeframe)},
        bars_cache=bars_cache,
        phase=f"atr50_range_ratio_target_{opt_symbol}",
    )
    bars = target_market[opt_symbol].bars
    ratio_median, ratio_count = _median_atr50_range_ratio_from_bars(bars)
    if ratio_median is None:
        raise ValueError(
            f"Unable to compute ATR(50)/(high+low) target for {opt_symbol} at {resolved_opt_timeframe} "
            f"within [{selection_start}, {selection_end})"
        )
    return {
        "symbol": opt_symbol,
        "timeframe": str(resolved_opt_timeframe),
        "target_atr50_range_ratio_median": float(ratio_median),
        "atr50_range_ratio_count": int(ratio_count),
        "n_bars": int(len(bars)),
    }


def _build_timeframe_selection_pdf(
    timeframe_results: pd.DataFrame,
    output_path: Path,
    initial_equity: float,
) -> str | None:
    required_cols = {"symbol", "timeframe", "selection_chart_data_path"}
    if timeframe_results.empty or not required_cols.issubset(timeframe_results.columns):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        cover = plt.figure(figsize=(11, 8.5))
        cover.text(0.05, 0.92, "Timeframe Selection PnL Comparison", fontsize=18, weight="bold")
        cover.text(
            0.05,
            0.86,
            "Each page overlays per-timeframe portfolio PnL curves for one symbol over the selection window.",
            fontsize=10,
        )
        y = 0.76
        for symbol in timeframe_results["symbol"].drop_duplicates().tolist():
            timeframes = timeframe_results.loc[timeframe_results["symbol"] == symbol, "timeframe"].tolist()
            ordered = ", ".join(str(tf) for tf in sorted(timeframes, key=_timeframe_sort_key))
            cover.text(0.05, y, f"{symbol}: {ordered}", fontsize=9)
            y -= 0.04
            if y < 0.08:
                break
        cover.tight_layout()
        pdf.savefig(cover)
        plt.close(cover)

        for symbol, group in timeframe_results.groupby("symbol", sort=False):
            fig, ax = plt.subplots(figsize=(12, 6))
            plotted = False
            for row in group.sort_values("timeframe", key=lambda s: s.map(_timeframe_sort_key)).itertuples(index=False):
                bars_path = getattr(row, "selection_chart_data_path", None)
                if not bars_path:
                    continue
                bars_file = Path(str(bars_path))
                if not bars_file.exists():
                    continue
                rows = [json.loads(line) for line in bars_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                bars = pd.DataFrame(rows)
                if bars.empty or "bar_end" not in bars or "equity" not in bars:
                    continue
                bars["bar_end"] = pd.to_datetime(bars["bar_end"], utc=True)
                bars["pnl"] = pd.to_numeric(bars["equity"], errors="coerce") - float(initial_equity)
                ax.plot(bars["bar_end"], bars["pnl"], linewidth=1.1, label=str(row.timeframe), alpha=0.9)
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.axhline(0.0, color="gray", linewidth=0.9, linestyle="--")
            ax.set_title(f"{symbol} timeframe selection")
            ax.set_xlabel("Time")
            ax.set_ylabel("Portfolio PnL")
            ax.grid(True, alpha=0.3)
            ax.legend(title="Timeframe", fontsize=8, title_fontsize=9, loc="best", ncol=2)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return str(output_path)


def _validate_symbol_timeframe_out_of_sample(task: dict[str, Any]) -> dict[str, Any]:
    symbol = str(task["symbol"]).upper()
    selected_timeframe = str(task["selected_timeframe"])
    baseline_timeframe = str(task["baseline_timeframe"])
    strategy_name = str(task["strategy_name"])
    fixed_param_values = dict(task["fixed_param_values"])
    if _WORKFLOW_SYMBOL_SPECS is None or _WORKFLOW_FX_DAILY is None:
        raise RuntimeError("Workflow context is not initialized for validation")
    if _WORKFLOW_CONFIG_KWARGS is None or _WORKFLOW_PORTFOLIO_KWARGS is None:
        raise RuntimeError("Workflow config context is not initialized for validation")
    symbol_specs = dict(_WORKFLOW_SYMBOL_SPECS)
    fx_daily = _WORKFLOW_FX_DAILY
    config_kwargs = dict(_WORKFLOW_CONFIG_KWARGS)
    portfolio_kwargs = dict(_WORKFLOW_PORTFOLIO_KWARGS)
    selection_metric = str(task["selection_metric"])
    validation_output_dir = Path(task["validation_output_dir"])
    m1 = _workflow_m1_window(symbol, str(task["start"]), str(task["end"]))
    args = argparse.Namespace(
        start=str(task["start"]),
        end=str(task["end"]),
        selection_start=str(task["selection_start"]),
        selection_end=str(task["selection_end"]),
    )
    portfolio_start = _utc_timestamp(args.start)
    portfolio_end = _utc_timestamp(args.end)
    selection_start = _utc_timestamp(args.selection_start)
    selection_end = _utc_timestamp(args.selection_end)
    reference_trade_timing = dict(task["reference_trade_timing"])

    strategy = get_strategy(strategy_name)
    fixed_params = build_strategy_params(strategy_name, fixed_param_values)
    local_bars_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    selected_result = _run_single_symbol_result(
        symbol=symbol,
        timeframe=selected_timeframe,
        m1=m1,
        strategy=strategy,
        fixed_params=fixed_params,
        symbol_specs=symbol_specs,
        fx_daily=fx_daily,
        config_kwargs=config_kwargs,
        portfolio_kwargs=portfolio_kwargs,
        bars_cache=local_bars_cache,
        phase=f"validation_selected_{symbol}_{selected_timeframe}",
    )
    selected_periods = _compute_period_stats(
        args=args,
        result=selected_result,
        selected_timeframes={symbol: selected_timeframe},
    )
    selected_oos_stats = selected_periods["out_of_sample"]["symbol_stats"][symbol]["stats"]
    selected_trade_timing = _trade_timing_metrics_for_period(
        selected_result["symbol_results"][symbol]["trades"],
        period_name="out_of_sample",
        portfolio_start=portfolio_start,
        portfolio_end=portfolio_end,
        selection_start=selection_start,
        selection_end=selection_end,
    )
    selected_pnl_chart = _save_validation_single_symbol_pnl(
        selected_result,
        validation_output_dir / symbol / f"selected_{selected_timeframe}",
        initial_equity=float(config_kwargs["initial_equity"]),
    )

    if selected_timeframe == baseline_timeframe:
        baseline_oos_stats = dict(selected_oos_stats)
        baseline_pnl_chart = selected_pnl_chart
        baseline_trade_timing = dict(selected_trade_timing)
    else:
        baseline_result = _run_single_symbol_result(
            symbol=symbol,
            timeframe=baseline_timeframe,
            m1=m1,
            strategy=strategy,
            fixed_params=fixed_params,
            symbol_specs=symbol_specs,
            fx_daily=fx_daily,
            config_kwargs=config_kwargs,
            portfolio_kwargs=portfolio_kwargs,
            bars_cache=local_bars_cache,
            phase=f"validation_opt_{symbol}_{baseline_timeframe}",
        )
        baseline_periods = _compute_period_stats(
            args=args,
            result=baseline_result,
            selected_timeframes={symbol: baseline_timeframe},
        )
        baseline_oos_stats = baseline_periods["out_of_sample"]["symbol_stats"][symbol]["stats"]
        baseline_trade_timing = _trade_timing_metrics_for_period(
            baseline_result["symbol_results"][symbol]["trades"],
            period_name="out_of_sample",
            portfolio_start=portfolio_start,
            portfolio_end=portfolio_end,
            selection_start=selection_start,
            selection_end=selection_end,
        )
        baseline_pnl_chart = _save_validation_single_symbol_pnl(
            baseline_result,
            validation_output_dir / symbol / f"opt_{baseline_timeframe}",
            initial_equity=float(config_kwargs["initial_equity"]),
        )

    selected_metric_value = float(selected_oos_stats.get(selection_metric, np.nan))
    baseline_metric_value = float(baseline_oos_stats.get(selection_metric, np.nan))
    outcome = _metric_comparison_outcome(selected_metric_value, baseline_metric_value, selection_metric)
    return {
        "symbol": symbol,
        "selection_metric": selection_metric,
        "selection_metric_lower_is_better": bool(_metric_lower_is_better(selection_metric)),
        "selected_timeframe": selected_timeframe,
        "opt_timeframe": baseline_timeframe,
        "selected_oos_metric_value": selected_metric_value,
        "opt_timeframe_oos_metric_value": baseline_metric_value,
        "comparison_outcome": outcome,
        "selected_beats_opt_timeframe": outcome == "selected_better",
        "selected_oos_stats": selected_oos_stats,
        "opt_timeframe_oos_stats": baseline_oos_stats,
        "opt_symbol_opt_timeframe_trade_timing_reference": reference_trade_timing,
        "selected_trade_timing": selected_trade_timing,
        "opt_timeframe_trade_timing": baseline_trade_timing,
        "selected_vs_opt_symbol_trade_timing_alignment": _trade_timing_alignment(
            selected_trade_timing,
            reference_trade_timing,
        ),
        "opt_timeframe_vs_opt_symbol_trade_timing_alignment": _trade_timing_alignment(
            baseline_trade_timing,
            reference_trade_timing,
        ),
        "selected_pnl_chart": selected_pnl_chart,
        "opt_timeframe_pnl_chart": baseline_pnl_chart,
    }


def _validate_timeframe_selection_out_of_sample(
    *,
    args: argparse.Namespace,
    fixed_param_values: dict[str, Any],
    selected_timeframes: dict[str, str],
    m1_by_symbol: dict[str, pd.DataFrame],
    shared_m1_by_symbol: dict[str, dict[str, Any]] | None,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    strategy_name: str,
    config_kwargs: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
    validation_output_dir: Path,
) -> dict[str, Any] | None:
    if args.opt_timeframe is None:
        return None

    baseline_timeframe = str(args.opt_timeframe)
    opt_m1 = _slice_m1_frame(m1_by_symbol[args.opt_symbol], args.start, args.end, copy=False)
    opt_strategy = get_strategy(strategy_name)
    opt_fixed_params = build_strategy_params(strategy_name, fixed_param_values)
    opt_reference_result = _run_single_symbol_result(
        symbol=args.opt_symbol,
        timeframe=baseline_timeframe,
        m1=opt_m1,
        strategy=opt_strategy,
        fixed_params=opt_fixed_params,
        symbol_specs=symbol_specs,
        fx_daily=fx_daily,
        config_kwargs=config_kwargs,
        portfolio_kwargs=portfolio_kwargs,
        bars_cache={},
        phase=f"validation_reference_{args.opt_symbol}_{baseline_timeframe}",
    )
    portfolio_start = _utc_timestamp(args.start)
    portfolio_end = _utc_timestamp(args.end)
    selection_start = _utc_timestamp(args.selection_start)
    selection_end = _utc_timestamp(args.selection_end)
    reference_trade_timing = _trade_timing_metrics_for_period(
        opt_reference_result["symbol_results"][args.opt_symbol]["trades"],
        period_name="out_of_sample",
        portfolio_start=portfolio_start,
        portfolio_end=portfolio_end,
        selection_start=selection_start,
        selection_end=selection_end,
    )
    summary_counts = {
        "selected_better": 0,
        "opt_timeframe_better": 0,
        "tie": 0,
        "unknown": 0,
    }
    validation_tasks = [
        {
            "symbol": symbol,
            "selected_timeframe": selected_timeframes[symbol],
            "baseline_timeframe": baseline_timeframe,
            "strategy_name": strategy_name,
            "fixed_param_values": fixed_param_values,
            "validation_output_dir": str(validation_output_dir),
            "selection_metric": args.selection_metric,
            "start": args.start,
            "end": args.end,
            "selection_start": args.selection_start,
            "selection_end": args.selection_end,
            "reference_trade_timing": reference_trade_timing,
        }
        for symbol in args.symbols
    ]
    if args.max_workers == 1:
        _init_workflow_worker_local(m1_by_symbol, symbol_specs, fx_daily, config_kwargs, portfolio_kwargs)
        rows = [
            _validate_symbol_timeframe_out_of_sample(task)
            for task in tqdm(
                validation_tasks,
                desc="Validating timeframes [out_of_sample]",
                unit="symbol",
            )
        ]
    else:
        if shared_m1_by_symbol is None:
            raise RuntimeError("Shared M1 payload is required for parallel timeframe validation")
        rows = []
        with ProcessPoolExecutor(
            max_workers=min(args.max_workers, len(validation_tasks)),
            initializer=_init_workflow_worker_shared,
            initargs=(shared_m1_by_symbol, symbol_specs, fx_daily, config_kwargs, portfolio_kwargs),
        ) as executor:
            futures = [executor.submit(_validate_symbol_timeframe_out_of_sample, task) for task in validation_tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Validating timeframes [out_of_sample]",
                unit="symbol",
            ):
                rows.append(future.result())
    rows = sorted(rows, key=lambda row: args.symbols.index(row["symbol"]))
    for row in rows:
        summary_counts[str(row["comparison_outcome"])] += 1

    return {
        "selection_metric": args.selection_metric,
        "selection_metric_lower_is_better": bool(_metric_lower_is_better(args.selection_metric)),
        "opt_timeframe": baseline_timeframe,
        "opt_symbol_opt_timeframe_trade_timing_reference": {
            "symbol": args.opt_symbol,
            "timeframe": baseline_timeframe,
            **reference_trade_timing,
        },
        "period": "out_of_sample",
        "charts_root": str(validation_output_dir),
        "rule": "bars in [portfolio_start, portfolio_end) excluding [selection_start, selection_end)",
        "summary": {
            **summary_counts,
            "n_symbols": len(args.symbols),
            "selected_better_ratio": (
                float(summary_counts["selected_better"]) / float(len(args.symbols))
                if args.symbols
                else np.nan
            ),
        },
        "rows": rows,
    }


def _default_timeframe_for_workflow(args: argparse.Namespace) -> str:
    if args.opt_timeframe is not None:
        return str(args.opt_timeframe)
    if not args.timeframe_candidates:
        raise ValueError("At least one timeframe candidate is required")
    return str(args.timeframe_candidates[0])


def _fixed_param_values(args: argparse.Namespace) -> dict[str, Any]:
    strategy_name = str(args.strategy)
    if strategy_name == "ma_atr_breakout":
        return {
            "ma_len": int(args.fixed_ma_len),
            "atr_len": int(args.fixed_atr_len),
            "atr_mult": float(args.fixed_atr_mult),
            "stop_lookback": int(args.fixed_stop_lookback),
            "ma_kind": "ema",
        }
    if strategy_name == "supertrend":
        return {
            "atr_len": int(args.fixed_atr_len),
            "atr_mult": float(args.fixed_atr_mult),
        }
    raise ValueError("--skip-param-search currently supports only --strategy ma_atr_breakout or --strategy supertrend")


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
    combined = add_neighbor_means(
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
    timeframe_selection_mode = str(task.get("timeframe_selection_mode", "backtest_metric"))
    strategy_name = str(task["strategy_name"])
    fixed_param_values = dict(task["fixed_param_values"])
    selection_metric = str(task["selection_metric"])
    timeframe_chart_root = Path(str(task["timeframe_chart_root"]))
    initial_equity = float(task["initial_equity"])
    if _WORKFLOW_SYMBOL_SPECS is None or _WORKFLOW_FX_DAILY is None:
        raise RuntimeError("Workflow context is not initialized for timeframe selection")
    if _WORKFLOW_CONFIG_KWARGS is None or _WORKFLOW_PORTFOLIO_KWARGS is None:
        raise RuntimeError("Workflow config context is not initialized for timeframe selection")
    symbol_specs = dict(_WORKFLOW_SYMBOL_SPECS)
    fx_daily = _WORKFLOW_FX_DAILY
    config_kwargs = dict(_WORKFLOW_CONFIG_KWARGS)
    portfolio_kwargs = dict(_WORKFLOW_PORTFOLIO_KWARGS)
    m1 = _workflow_m1_window(symbol, str(task["selection_start"]), str(task["selection_end"]))

    strategy = get_strategy(strategy_name)
    params = None
    if timeframe_selection_mode == "backtest_metric":
        params = build_strategy_params(strategy_name, fixed_param_values)
    rows: list[dict[str, Any]] = []
    best_timeframe = None
    best_score = float("-inf")
    bars_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    if timeframe_selection_mode == "ptr20_median_match":
        target_ptr20_median = float(task["target_ptr20_median"])
        for timeframe in timeframes:
            timeframe_by_symbol = {symbol: timeframe}
            single_markets = _build_markets_from_cache(
                {symbol: m1},
                timeframe_by_symbol,
                bars_cache=bars_cache,
                phase="selection_ptr20",
            )
            bars = single_markets[symbol].bars
            ptr20_median, ptr20_count = _median_ptr20_from_bars(bars)
            if ptr20_median is None:
                continue
            distance = abs(float(ptr20_median) - target_ptr20_median)
            score = -distance
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "selection_metric": "ptr20_abs_distance",
                    "selection_metric_value": distance,
                    "selection_score": score,
                    "selection_chart": None,
                    "selection_chart_data_path": None,
                    "target_ptr20_median": target_ptr20_median,
                    "ptr20_median": float(ptr20_median),
                    "ptr20_abs_distance": distance,
                    "ptr20_count": int(ptr20_count),
                    "n_bars": int(len(bars)),
                    "annualized_return": np.nan,
                }
            )
            if (
                best_timeframe is None
                or score > best_score
                or (
                    np.isclose(score, best_score)
                    and _timeframe_sort_key(timeframe) < _timeframe_sort_key(best_timeframe)
                )
            ):
                best_score = score
                best_timeframe = timeframe
    elif timeframe_selection_mode == ATR50_HL_SUM_RATIO_MODE:
        target_ratio_median = float(task["target_atr50_range_ratio_median"])
        for timeframe in timeframes:
            timeframe_by_symbol = {symbol: timeframe}
            single_markets = _build_markets_from_cache(
                {symbol: m1},
                timeframe_by_symbol,
                bars_cache=bars_cache,
                phase="selection_atr50_range_ratio",
            )
            bars = single_markets[symbol].bars
            ratio_median, ratio_count = _median_atr50_range_ratio_from_bars(bars)
            if ratio_median is None:
                continue
            distance = abs(float(ratio_median) - target_ratio_median)
            score = -distance
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "selection_metric": "atr50_range_ratio_abs_distance",
                    "selection_metric_value": distance,
                    "selection_score": score,
                    "selection_chart": None,
                    "selection_chart_data_path": None,
                    "target_atr50_range_ratio_median": target_ratio_median,
                    "atr50_range_ratio_median": float(ratio_median),
                    "atr50_range_ratio_abs_distance": distance,
                    "atr50_range_ratio_count": int(ratio_count),
                    "n_bars": int(len(bars)),
                    "annualized_return": np.nan,
                }
            )
            if (
                best_timeframe is None
                or score > best_score
                or (
                    np.isclose(score, best_score)
                    and _timeframe_sort_key(timeframe) < _timeframe_sort_key(best_timeframe)
                )
            ):
                best_score = score
                best_timeframe = timeframe
    elif timeframe_selection_mode == "backtest_metric":
        if params is None:
            raise RuntimeError("Strategy params must be initialized for backtest_metric timeframe selection")
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
            metric_value = _metric_value(stats, selection_metric)
            score = _metric_score(stats, selection_metric)
            chart_output_dir = timeframe_chart_root / symbol / str(timeframe)
            chart_path, chart_data_path = _save_timeframe_selection_single_symbol_pnl(
                result,
                chart_output_dir,
                initial_equity=initial_equity,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "selection_metric": selection_metric,
                    "selection_metric_value": metric_value,
                    "selection_score": score,
                    "selection_chart": chart_path,
                    "selection_chart_data_path": chart_data_path,
                    **stats,
                }
            )
            if best_timeframe is None or score > best_score:
                best_score = score
                best_timeframe = timeframe
    else:
        raise ValueError(f"Unsupported timeframe selection mode: {timeframe_selection_mode}")

    scored = pd.DataFrame(rows)
    if scored.empty or best_timeframe is None:
        raise ValueError(f"No valid timeframe found for symbol {symbol}")
    scored = scored.sort_values(
        ["selection_score", "annualized_return"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return {
        "symbol": symbol,
        "best_timeframe": str(best_timeframe),
        "best_score": float(best_score),
        "rows": scored.to_dict(orient="records"),
    }


def _resolve_timeframes_for_opt_timeframe(
    *,
    args: argparse.Namespace,
    selection_mode: str,
    resolved_opt_timeframe: str,
    fixed_param_values: dict[str, Any],
    m1_by_symbol: dict[str, pd.DataFrame],
    shared_m1_by_symbol: dict[str, dict[str, Any]] | None,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    bars_cache: dict[tuple[str, str, str], pd.DataFrame],
    config_kwargs: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
    timeframe_chart_root: Path,
) -> tuple[dict[str, str], pd.DataFrame]:
    ptr20_target = None
    atr50_range_ratio_target = None
    selection_symbols = list(args.symbols)
    timeframe_rows: list[dict[str, Any]] = []
    selected_timeframes: dict[str, str] = {}
    if args.timeframe_selection_mode == "ptr20_median_match":
        ptr20_target = _build_ptr20_target(
            opt_symbol=args.opt_symbol,
            resolved_opt_timeframe=resolved_opt_timeframe,
            selection_start=args.selection_start,
            selection_end=args.selection_end,
            m1_by_symbol=m1_by_symbol,
            bars_cache=bars_cache,
        )
        selected_timeframes[args.opt_symbol] = str(resolved_opt_timeframe)
        timeframe_rows.append(
            {
                "symbol": args.opt_symbol,
                "timeframe": str(resolved_opt_timeframe),
                "selection_metric": "ptr20_abs_distance",
                "selection_metric_value": 0.0,
                "selection_score": -0.0,
                "selection_chart": None,
                "selection_chart_data_path": None,
                "target_ptr20_median": float(ptr20_target["target_ptr20_median"]),
                "ptr20_median": float(ptr20_target["target_ptr20_median"]),
                "ptr20_abs_distance": 0.0,
                "ptr20_count": int(ptr20_target["ptr20_count"]),
                "n_bars": int(ptr20_target["n_bars"]),
                "annualized_return": np.nan,
            }
        )
        selection_symbols = [symbol for symbol in args.symbols if symbol != args.opt_symbol]
    elif args.timeframe_selection_mode == ATR50_HL_SUM_RATIO_MODE:
        atr50_range_ratio_target = _build_atr50_range_ratio_target(
            opt_symbol=args.opt_symbol,
            resolved_opt_timeframe=resolved_opt_timeframe,
            selection_start=args.selection_start,
            selection_end=args.selection_end,
            m1_by_symbol=m1_by_symbol,
            bars_cache=bars_cache,
        )
        selected_timeframes[args.opt_symbol] = str(resolved_opt_timeframe)
        timeframe_rows.append(
            {
                "symbol": args.opt_symbol,
                "timeframe": str(resolved_opt_timeframe),
                "selection_metric": "atr50_range_ratio_abs_distance",
                "selection_metric_value": 0.0,
                "selection_score": -0.0,
                "selection_chart": None,
                "selection_chart_data_path": None,
                "target_atr50_range_ratio_median": float(
                    atr50_range_ratio_target["target_atr50_range_ratio_median"]
                ),
                "atr50_range_ratio_median": float(
                    atr50_range_ratio_target["target_atr50_range_ratio_median"]
                ),
                "atr50_range_ratio_abs_distance": 0.0,
                "atr50_range_ratio_count": int(atr50_range_ratio_target["atr50_range_ratio_count"]),
                "n_bars": int(atr50_range_ratio_target["n_bars"]),
                "annualized_return": np.nan,
            }
        )
        selection_symbols = [symbol for symbol in args.symbols if symbol != args.opt_symbol]
    timeframe_tasks = [
        {
            "symbol": symbol,
            "timeframes": args.timeframe_candidates,
            "timeframe_selection_mode": args.timeframe_selection_mode,
            "strategy_name": args.strategy,
            "fixed_param_values": fixed_param_values,
            "selection_metric": args.selection_metric,
            "selection_start": args.selection_start,
            "selection_end": args.selection_end,
            "timeframe_chart_root": str(timeframe_chart_root),
            "initial_equity": float(args.initial_equity),
            "target_ptr20_median": float(ptr20_target["target_ptr20_median"]) if ptr20_target is not None else None,
            "target_atr50_range_ratio_median": (
                float(atr50_range_ratio_target["target_atr50_range_ratio_median"])
                if atr50_range_ratio_target is not None
                else None
            ),
        }
        for symbol in selection_symbols
    ]
    if not timeframe_tasks:
        timeframe_results_raw = []
    elif args.max_workers == 1:
        _init_workflow_worker_local(m1_by_symbol, symbol_specs, fx_daily, config_kwargs, portfolio_kwargs)
        timeframe_results_raw = [
            _evaluate_symbol_timeframes(task)
            for task in tqdm(
                timeframe_tasks,
                desc=_timeframe_selection_progress_desc(args, selection_mode),
            )
        ]
    else:
        if shared_m1_by_symbol is None:
            raise RuntimeError("Shared M1 payload is required for parallel timeframe selection")
        timeframe_results_raw = []
        with ProcessPoolExecutor(
            max_workers=min(args.max_workers, len(timeframe_tasks)),
            initializer=_init_workflow_worker_shared,
            initargs=(shared_m1_by_symbol, symbol_specs, fx_daily, config_kwargs, portfolio_kwargs),
        ) as executor:
            futures = [executor.submit(_evaluate_symbol_timeframes, task) for task in timeframe_tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=_timeframe_selection_progress_desc(args, selection_mode),
            ):
                timeframe_results_raw.append(future.result())
    for item in sorted(timeframe_results_raw, key=lambda x: args.symbols.index(x["symbol"])):
        selected_timeframes[item["symbol"]] = item["best_timeframe"]
        timeframe_rows.extend(item["rows"])
    timeframe_results = pd.DataFrame(timeframe_rows)
    if not timeframe_results.empty:
        timeframe_results = timeframe_results.sort_values(
            ["symbol", "selection_score", "annualized_return"],
            ascending=[True, False, False],
        )
    return selected_timeframes, timeframe_results


def _run_candidate_workflow(
    *,
    args: argparse.Namespace,
    selection_mode: str,
    fixed_params: object,
    fixed_param_values: dict[str, Any],
    resolved_opt_timeframe: str,
    selected_timeframes_override: dict[str, str] | None,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
    m1_by_symbol: dict[str, pd.DataFrame],
    bars_cache: dict[tuple[str, str, str], pd.DataFrame],
    strategy: Any,
    config_kwargs: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shared_m1_by_symbol: dict[str, dict[str, Any]] | None = None
    shared_blocks: list[SharedMemory] = []
    if args.max_workers > 1:
        shared_m1_by_symbol, shared_blocks = _build_shared_m1_payload(m1_by_symbol)
    try:
        selected_timeframes: dict[str, str] = {}
        if selected_timeframes_override is not None:
            selected_timeframes = {symbol: str(selected_timeframes_override[symbol]) for symbol in args.symbols}
            timeframe_results = pd.DataFrame()
        elif args.skip_timeframe_selection:
            selected_timeframes = {symbol: str(resolved_opt_timeframe) for symbol in args.symbols}
            timeframe_results = pd.DataFrame()
        else:
            selected_timeframes, timeframe_results = _resolve_timeframes_for_opt_timeframe(
                args=args,
                selection_mode=selection_mode,
                resolved_opt_timeframe=resolved_opt_timeframe,
                fixed_param_values=fixed_param_values,
                m1_by_symbol=m1_by_symbol,
                shared_m1_by_symbol=shared_m1_by_symbol,
                symbol_specs=symbol_specs,
                fx_daily=fx_daily,
                bars_cache=bars_cache,
                config_kwargs=config_kwargs,
                portfolio_kwargs=portfolio_kwargs,
                timeframe_chart_root=out_dir / "timeframe_selection_charts",
            )
        if not timeframe_results.empty:
            timeframe_results["selection_mode"] = selection_mode
            timeframe_results["timeframe_selection_mode"] = args.timeframe_selection_mode
        _write_records_jsonl(timeframe_results, out_dir / "timeframe_search_results.jsonl")
        timeframe_selection_pdf_path = _build_timeframe_selection_pdf(
            timeframe_results,
            out_dir / "timeframe_selection_pnl_comparison.pdf",
            initial_equity=float(args.initial_equity),
        )

        final_timeframes = {symbol: str(resolved_opt_timeframe) for symbol in args.symbols}
        final_timeframes.update(selected_timeframes)
        final_m1_window = _slice_m1_window(m1_by_symbol, args.start, args.end, copy=False)
        final_markets = _build_markets_from_cache(
            final_m1_window,
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
        period_stats = _compute_period_stats(
            args=args,
            result=final_result,
            selected_timeframes=final_timeframes,
        )
        period_stats_path = out_dir / "period_stats.json"
        _write_json(period_stats_path, period_stats)
        timeframe_validation = _validate_timeframe_selection_out_of_sample(
            args=args,
            fixed_param_values=fixed_param_values,
            selected_timeframes=final_timeframes,
            m1_by_symbol=m1_by_symbol,
            shared_m1_by_symbol=shared_m1_by_symbol,
            symbol_specs=symbol_specs,
            fx_daily=fx_daily,
            strategy_name=args.strategy,
            config_kwargs=config_kwargs,
            portfolio_kwargs=portfolio_kwargs,
            validation_output_dir=out_dir / "timeframe_validation_out_of_sample_charts",
        )
        timeframe_validation_path = None
        if timeframe_validation is not None:
            timeframe_validation_path = out_dir / "timeframe_validation_out_of_sample.json"
            _write_json(timeframe_validation_path, timeframe_validation)
        return {
            "selection_mode": selection_mode,
            "output_dir": str(out_dir),
            "resolved_opt_timeframe": resolved_opt_timeframe,
            "fixed_params": asdict(fixed_params) if is_dataclass(fixed_params) else fixed_param_values,
            "selected_timeframes": final_timeframes,
            "portfolio_stats": final_result["portfolio_stats"],
            "period_stats_path": str(period_stats_path),
            "period_stats": period_stats,
            "timeframe_selection_pdf_path": timeframe_selection_pdf_path,
            "timeframe_validation_out_of_sample_path": (
                str(timeframe_validation_path) if timeframe_validation_path is not None else None
            ),
            "timeframe_validation_out_of_sample": timeframe_validation,
        }
    finally:
        _cleanup_shared_blocks(shared_blocks)


def main() -> int:
    args = _parse_args()
    args.symbols = normalize_symbols(args.symbols)
    args.opt_symbol = str(args.opt_symbol or args.symbols[0]).upper()
    args.opt_timeframe = _normalize_optional_timeframe(args.opt_timeframe)
    args.timeframe_candidates = [str(tf) for tf in args.timeframe_candidates]
    symbol_timeframes_payload = _load_symbol_timeframes_payload(args.symbol_timeframes)
    if args.opt_symbol not in args.symbols:
        raise ValueError(f"opt_symbol {args.opt_symbol!r} must be included in --symbols")
    if args.skip_timeframe_selection and args.opt_timeframe is None and args.symbol_timeframes is None:
        raise ValueError("--skip-timeframe-selection requires --opt-timeframe to be a concrete timeframe")
    if args.skip_param_search and args.opt_timeframe is None:
        if symbol_timeframes_payload is None or args.opt_symbol not in symbol_timeframes_payload:
            raise ValueError(
                "--skip-param-search with --opt-timeframe none requires --symbol-timeframes "
                f"to provide a timeframe for opt_symbol {args.opt_symbol!r}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    symbol_specs = _load_symbol_specs(args.symbol_specs)
    fx_daily = _load_fx_daily(args.fx_daily)
    m1_by_symbol = _load_m1_cache(args)
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
        resolved_opt_timeframe = (
            str(args.opt_timeframe)
            if args.opt_timeframe is not None
            else str(symbol_timeframes_payload[args.opt_symbol])
        )
        args.skip_timeframe_selection = False
        param_distribution_summary = None
        generalized_param_values = None
    else:
        selection_m1_by_symbol = _slice_m1_window(m1_by_symbol, args.selection_start, args.selection_end, copy=False)
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
        del selection_m1_by_symbol
    selected_timeframes_override = None
    if args.symbol_timeframes is not None:
        selected_timeframes_override = load_symbol_timeframes(
            symbols=args.symbols,
            default_timeframe=resolved_opt_timeframe,
            symbol_timeframes_path=args.symbol_timeframes,
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
        "opposite_signal_action": args.opposite_signal_action,
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
            selected_timeframes_override=selected_timeframes_override,
            symbol_specs=symbol_specs,
            fx_daily=fx_daily,
            m1_by_symbol=m1_by_symbol,
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
                    "selection_mode": "generalized_top_count",
                    "output_dir": str(args.out_dir),
                    "resolved_opt_timeframe": resolved_opt_timeframe,
                    "fixed_params": generalized_param_values,
                    "selected_timeframes": candidate_runs[0]["selected_timeframes"],
                    "portfolio_stats": candidate_runs[0]["portfolio_stats"],
                    "period_stats_path": candidate_runs[0]["period_stats_path"],
                    "period_stats": candidate_runs[0]["period_stats"],
                    "timeframe_selection_pdf_path": candidate_runs[0]["timeframe_selection_pdf_path"],
                    "timeframe_validation_out_of_sample_path": candidate_runs[0]["timeframe_validation_out_of_sample_path"],
                    "timeframe_validation_out_of_sample": candidate_runs[0]["timeframe_validation_out_of_sample"],
                    "skipped_duplicate_run": True,
                    "same_as": "best_ranked",
                }
            )
        else:
            generalized_params = build_strategy_params(args.strategy, generalized_param_values)
            candidate_runs.append(
                _run_candidate_workflow(
                    args=args,
                    selection_mode="generalized_top_count",
                    fixed_params=generalized_params,
                    fixed_param_values=generalized_param_values,
                    resolved_opt_timeframe=resolved_opt_timeframe,
                    selected_timeframes_override=selected_timeframes_override,
                    symbol_specs=symbol_specs,
                    fx_daily=fx_daily,
                    m1_by_symbol=m1_by_symbol,
                    bars_cache=bars_cache,
                    strategy=strategy,
                    config_kwargs=config_kwargs,
                    portfolio_kwargs=portfolio_kwargs,
                    out_dir=args.out_dir / "generalized_top_count",
                )
            )

    primary_run = candidate_runs[0]

    summary = {
        "symbols": args.symbols,
        "strategy": args.strategy,
        "opt_symbol": args.opt_symbol,
        "opt_timeframe": args.opt_timeframe,
        "symbol_timeframes_path": str(args.symbol_timeframes) if args.symbol_timeframes is not None else None,
        "selected_timeframes_source": "file" if selected_timeframes_override is not None else ("selection" if not args.skip_timeframe_selection else "resolved_opt_timeframe"),
        "resolved_opt_timeframe": resolved_opt_timeframe,
        "timeframe_selection_mode": args.timeframe_selection_mode,
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
        "selection_metric_lower_is_better": bool(_metric_lower_is_better(args.selection_metric)),
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
        "opposite_signal_action": args.opposite_signal_action,
        "portfolio_stats": primary_run["portfolio_stats"],
        "period_stats_path": primary_run["period_stats_path"],
        "period_stats": primary_run["period_stats"],
        "timeframe_selection_pdf_path": primary_run["timeframe_selection_pdf_path"],
        "timeframe_validation_out_of_sample_path": primary_run["timeframe_validation_out_of_sample_path"],
        "timeframe_validation_out_of_sample": primary_run["timeframe_validation_out_of_sample"],
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
