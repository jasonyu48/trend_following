from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from data_paths import DEFAULT_DATA_DIR
from indicators import average_true_range
from market_data import MarketDataSlice, load_symbol_market_data
from run_backtest import _load_fx_daily, _load_symbol_specs, build_engine_config
from search_params import run_portfolio_backtest
from symbol_universe import DEFAULT_SYMBOLS
from strategies import STRATEGIES, get_strategy
from strategies.base import StrategyDefinition

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INDICATORS = [
    "rsi_14",
    "rsi_20",
    "rsi_30",
    "vol_z_14",
    "vol_z_20",
    "vol_z_50",
    "atr_z_14",
    "atr_z_20",
    "atr_z_50",
    "atr_pct_z_14",
    "atr_pct_z_20",
    "atr_pct_z_50",
]


@dataclass(frozen=True)
class StudyConfig:
    data_dir: Path
    symbol_specs: Path
    fx_daily: Path
    symbols: list[str]
    start: str
    end: str
    timeframe: str
    commission_bps: float
    slippage_bps: float
    overnight_long_rate: float
    overnight_short_rate: float
    overnight_day_count: int
    initial_margin_ratio: float
    maintenance_margin_ratio: float
    strategy: str
    strategy_params: dict[str, Any]
    portfolio_mode: str
    cash_per_trade: float
    risk_per_trade: float
    risk_per_trade_pct: float
    initial_equity: float
    split_date: str | None
    indicators: list[str]
    out_dir: Path
    workers: int
    bootstrap_b: int
    bootstrap_seed: int


@dataclass
class IndicatorResult:
    name: str
    filter_mode: str
    sample_n_train: int
    sample_n_test: int
    lo: float
    hi: float
    best_interval_sum: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    n_trades_total: float
    sorted_x_train: np.ndarray
    sorted_cumret_train: np.ndarray
    sorted_x_test: np.ndarray
    sorted_cumret_test: np.ndarray
    filtered_portfolio_bars: pd.DataFrame


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified portfolio filter study on top of the existing backtest engine.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--symbol-specs", type=Path, default=Path("symbol_specs.json"))
    p.add_argument("--fx-daily", type=Path, default=Path("data/fx_daily/fx_daily_2012_2025.csv"))
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", type=str, default="20160101")
    p.add_argument("--end", type=str, default="20220101")
    p.add_argument("--timeframe", type=str, default="4H")
    p.add_argument("--commission-bps", type=float, default=0.35)
    p.add_argument("--slippage-bps", type=float, default=0.3)
    p.add_argument("--overnight-long-rate", type=float, default=0.0)
    p.add_argument("--overnight-short-rate", type=float, default=0.0)
    p.add_argument("--overnight-day-count", type=int, default=360)
    p.add_argument("--initial-margin-ratio", type=float, default=0.01)
    p.add_argument("--maintenance-margin-ratio", type=float, default=0.005)
    p.add_argument("--strategy", type=str, default="ma_atr_breakout", choices=sorted(STRATEGIES))
    p.add_argument("--ma-len", type=int, default=14)
    p.add_argument("--atr-len", type=int, default=30)
    p.add_argument("--atr-mult", type=float, default=3.0)
    p.add_argument("--stop-lookback", type=int, default=10)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--divergence-lookback", type=int, default=8)
    p.add_argument("--slope-len", type=int, default=6)
    p.add_argument("--slope-threshold", type=float, default=0.05)
    p.add_argument("--slope-norm-mode", type=str, default="atr", choices=["atr", "close_pct", "none"])
    p.add_argument("--anchor-lookback", type=int, default=5)
    p.add_argument("--session-start-hour-utc", type=int, default=0)
    p.add_argument("--session-end-hour-utc", type=int, default=0)
    p.add_argument(
        "--portfolio-mode",
        type=str,
        default="fixed_risk_pct",
        choices=["fixed_cash", "fixed_risk", "fixed_risk_pct"],
    )
    p.add_argument("--cash-per-trade", type=float, default=1.0)
    p.add_argument("--risk-per-trade", type=float, default=1.0)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.0001)
    p.add_argument("--initial-equity", type=float, default=10000.0)
    p.add_argument("--split-date", type=str, default="")
    p.add_argument("--indicators", type=str, default="")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--bootstrap-b", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("results/ma_atr_breakout_filter_study"))
    return p.parse_args()


def _build_params(args: argparse.Namespace) -> Any:
    strategy = get_strategy(args.strategy)
    kwargs = {name: getattr(args, name) for name in strategy.param_names if hasattr(args, name)}
    return strategy.make_params(**kwargs)


def _load_markets(cfg: StudyConfig) -> dict[str, MarketDataSlice]:
    out: dict[str, MarketDataSlice] = {}
    for symbol in tqdm(cfg.symbols, desc="Loading market data"):
        out[symbol.upper()] = load_symbol_market_data(
            data_dir=cfg.data_dir,
            symbol=symbol.upper(),
            timeframe=cfg.timeframe,
            start=cfg.start,
            end=cfg.end,
        )
    return out


def _rolling_zscore(s: pd.Series, n: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    mu = x.rolling(n, min_periods=n).mean()
    sd = x.rolling(n, min_periods=n).std(ddof=1)
    return (x - mu) / sd


def _rsi_wilder(close: pd.Series, n: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce").astype("float64")
    d = c.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / float(n), adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1.0 / float(n), adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_indicator_series(name: str, bars: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(bars["close"], errors="coerce").astype("float64")
    if name.startswith("rsi_"):
        n = int(name.split("_")[1])
        return _rsi_wilder(close, n)
    if name.startswith("vol_z_"):
        n = int(name.split("_")[2])
        ret = close.pct_change()
        realized_vol = ret.rolling(n, min_periods=n).std(ddof=1)
        return _rolling_zscore(realized_vol, n)
    if name.startswith("atr_pct_z_"):
        n = int(name.split("_")[3])
        atr = average_true_range(bars, length=n)
        atr_pct = atr / close.replace(0.0, np.nan)
        return _rolling_zscore(atr_pct, n)
    if name.startswith("atr_z_"):
        n = int(name.split("_")[2])
        atr = average_true_range(bars, length=n)
        return _rolling_zscore(atr, n)
    raise ValueError(f"Unsupported indicator: {name}")


def _build_indicator_bank(bars: pd.DataFrame, indicators: list[str]) -> dict[str, pd.Series]:
    return {name: _compute_indicator_series(name, bars) for name in indicators}


def _normalize_trade_return(trades: pd.DataFrame) -> pd.Series:
    entry_equity = pd.to_numeric(trades["entry_equity"], errors="coerce").astype("float64")
    net_pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").astype("float64")
    out = net_pnl / entry_equity.replace(0.0, np.nan)
    return out.astype("float64")


def _map_trade_entries_to_schedule(trades: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["entry_time", "window_start", "window_end", "signal_bar_end"])
    if schedule.empty:
        raise RuntimeError("Trade-to-schedule mapping failed because the schedule is empty while trades exist.")
    left = trades[["entry_time"]].copy().sort_values("entry_time")
    right = schedule[["window_start", "window_end", "signal_bar_end"]].copy().sort_values("window_start")
    merged = pd.merge_asof(left, right, left_on="entry_time", right_on="window_start", direction="backward")
    valid = (merged["entry_time"] >= merged["window_start"]) & (merged["entry_time"] <= merged["window_end"])
    if not bool(valid.fillna(False).all()):
        bad = merged.loc[~valid.fillna(False), ["entry_time", "window_start", "window_end"]]
        raise RuntimeError(f"Trade-to-schedule mapping produced invalid rows:\n{bad.to_string(index=False)}")
    return merged


def _build_trade_samples(
    markets: dict[str, MarketDataSlice],
    baseline_symbol_results: dict[str, dict],
    indicators: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, market in tqdm(markets.items(), total=len(markets), desc="Building trade samples"):
        result = baseline_symbol_results[symbol]
        trades = result["trades"].copy()
        if trades.empty:
            continue
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades = trades.sort_values("entry_time").reset_index(drop=True)
        mapping = _map_trade_entries_to_schedule(trades, result["schedule"])
        bank = _build_indicator_bank(market.bars, indicators)
        sample = trades.copy()
        sample["symbol"] = symbol
        sample["trade_return"] = _normalize_trade_return(sample)
        signal_bar_end = pd.DatetimeIndex(mapping["signal_bar_end"]).tz_convert("UTC")
        sample["signal_bar_end"] = signal_bar_end
        for name, series in bank.items():
            aligned = series.reindex(signal_bar_end)
            sample[name] = aligned.to_numpy(dtype="float64", copy=False)
        frames.append(sample)
    if not frames:
        raise RuntimeError("No baseline trades were produced; filter study cannot proceed.")
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def _best_interval_from_pairs(x: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    order = np.argsort(x)
    xs = x[order]
    rs = r[order]
    best_sum = -np.inf
    cur_sum = 0.0
    cur_i = 0
    best_i = 0
    best_j = 0
    for j, value in enumerate(rs):
        if cur_sum <= 0.0:
            cur_sum = float(value)
            cur_i = j
        else:
            cur_sum += float(value)
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_i = cur_i
            best_j = j
    lo = float(xs[best_i])
    hi = float(xs[best_j])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi, float(best_sum)


def _sorted_cum_curve(x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.array([], dtype="float64"), np.array([], dtype="float64")
    order = np.argsort(x)
    xs = x[order]
    rs = r[order]
    return xs, np.cumsum(rs)


def _stable_interval_bootstrap(
    x: np.ndarray,
    r: np.ndarray,
    *,
    b: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    los: list[float] = []
    his: list[float] = []
    for _ in range(b):
        idx = rng.integers(0, len(x), size=len(x), endpoint=False)
        lo_b, hi_b, _ = _best_interval_from_pairs(x[idx], r[idx])
        if np.isfinite(lo_b) and np.isfinite(hi_b):
            los.append(float(lo_b))
            his.append(float(hi_b))
    if not los or not his:
        return np.nan, np.nan, np.nan
    lo = float(np.median(np.asarray(los, dtype="float64")))
    hi = float(np.median(np.asarray(his, dtype="float64")))
    if lo > hi:
        lo, hi = hi, lo
    mask = (x >= lo) & (x <= hi)
    interval_sum = float(np.nansum(r[mask])) if int(mask.sum()) > 0 else np.nan
    return lo, hi, interval_sum


def _stable_interval_bootstrap_min(
    x: np.ndarray,
    r: np.ndarray,
    *,
    b: int,
    seed: int,
) -> tuple[float, float, float]:
    lo, hi, neg_interval_sum = _stable_interval_bootstrap(
        x,
        -r,
        b=b,
        seed=seed,
    )
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan, np.nan, np.nan
    mask = (x >= lo) & (x <= hi)
    interval_sum = float(np.nansum(r[mask])) if int(mask.sum()) > 0 else np.nan
    return lo, hi, interval_sum


def _estimate_strategy_warmup_bars(strategy_name: str, params: Any) -> int:
    if strategy_name == "ma_atr_breakout":
        return int(max(params.ma_len, params.atr_len, params.stop_lookback))
    if strategy_name == "ma_divergence_momentum_confirm":
        return int(max(4 * params.n, params.divergence_lookback + 1, params.stop_lookback))
    if strategy_name == "lr_slope_anchor_breakout":
        return int(max(params.slope_len, params.anchor_lookback) + 2)
    raise ValueError(f"Unsupported strategy for warmup estimation: {strategy_name}")


def _estimate_required_warmup_bars(strategy_name: str, params: Any, indicators: list[str]) -> int:
    strategy_warmup = _estimate_strategy_warmup_bars(strategy_name, params)
    indicator_warmup = 0
    for name in indicators:
        if name.startswith("rsi_"):
            indicator_warmup = max(indicator_warmup, int(name.split("_")[1]))
        elif name.startswith("vol_z_"):
            n = int(name.split("_")[2])
            indicator_warmup = max(indicator_warmup, 2 * n)
        elif name.startswith("atr_pct_z_"):
            n = int(name.split("_")[3])
            indicator_warmup = max(indicator_warmup, 2 * n)
        elif name.startswith("atr_z_"):
            n = int(name.split("_")[2])
            indicator_warmup = max(indicator_warmup, 2 * n)
        else:
            raise ValueError(f"Unsupported indicator for warmup estimation: {name}")
    return max(strategy_warmup, indicator_warmup)


def _build_test_markets(
    markets: dict[str, MarketDataSlice],
    split_dt: pd.Timestamp,
    warmup_bars: int,
) -> dict[str, MarketDataSlice]:
    out: dict[str, MarketDataSlice] = {}
    for symbol, market in markets.items():
        bars = market.bars.sort_index()
        split_pos = int(bars.index.searchsorted(split_dt, side="left"))
        if split_pos >= len(bars):
            raise RuntimeError(f"Split date {split_dt} is beyond the available bars for {symbol}.")
        start_pos = max(0, split_pos - warmup_bars)
        bars_slice = bars.iloc[start_pos:].copy()
        first_bar_end = pd.Timestamp(bars_slice.index[0]).tz_convert("UTC")
        m1_slice = market.m1.loc[market.m1.index >= first_bar_end].copy()
        if m1_slice.empty:
            raise RuntimeError(f"No M1 data remained for the test slice of {symbol}.")
        out[symbol] = MarketDataSlice(symbol=symbol, m1=m1_slice, bars=bars_slice)
    return out


def _make_filtered_strategy(
    base_strategy: StrategyDefinition,
    *,
    split_dt: pd.Timestamp,
    indicator_name: str | None,
    filter_mode: str,
    lo: float | None,
    hi: float | None,
) -> StrategyDefinition:
    def build_signal_schedule(bars: pd.DataFrame, features: pd.DataFrame, params: Any) -> pd.DataFrame:
        schedule = base_strategy.build_signal_schedule(bars, features, params)
        if schedule.empty:
            return schedule
        signal_ts = pd.DatetimeIndex(schedule["signal_bar_end"]).tz_convert("UTC")
        entry_allowed = signal_ts >= split_dt
        if indicator_name is not None:
            indicator = _compute_indicator_series(indicator_name, bars)
            signal_values = indicator.reindex(signal_ts)
            in_interval = signal_values.between(float(lo), float(hi), inclusive="both").to_numpy()
            indicator_ok = signal_values.notna().to_numpy()
            if filter_mode == "allow_inside":
                entry_allowed = entry_allowed & indicator_ok & in_interval
            elif filter_mode == "block_inside":
                entry_allowed = entry_allowed & indicator_ok & (~in_interval)
            else:
                raise ValueError(f"Unsupported filter_mode: {filter_mode}")
        out = schedule.copy()
        out["allow_long_entry"] = np.asarray(entry_allowed, dtype=bool)
        out["allow_short_entry"] = np.asarray(entry_allowed, dtype=bool)
        return out.reset_index(drop=True)

    return StrategyDefinition(
        name=f"{base_strategy.name}_filtered",
        params_type=base_strategy.params_type,
        execution_style=base_strategy.execution_style,
        compute_features=base_strategy.compute_features,
        build_signal_schedule=build_signal_schedule,
        default_grid=base_strategy.default_grid,
        is_entry_allowed=base_strategy.is_entry_allowed,
    )


def _run_study_backtest(
    markets: dict[str, MarketDataSlice],
    params: Any,
    config: EngineConfig,
    base_strategy: StrategyDefinition,
    cfg: StudyConfig,
    *,
    split_dt: pd.Timestamp,
    indicator_name: str | None,
    filter_mode: str = "allow_inside",
    lo: float | None,
    hi: float | None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    study_strategy = _make_filtered_strategy(
        base_strategy,
        split_dt=split_dt,
        indicator_name=indicator_name,
        filter_mode=filter_mode,
        lo=lo,
        hi=hi,
    )
    return run_portfolio_backtest(
        market_by_symbol=markets,
        params=params,
        strategy=study_strategy,
        config=config,
        show_progress=show_progress,
        progress_desc=progress_desc,
        collect_events=False,
        portfolio_mode=cfg.portfolio_mode,
        cash_per_trade=cfg.cash_per_trade,
        risk_per_trade=cfg.risk_per_trade,
        risk_per_trade_pct=cfg.risk_per_trade_pct,
    )


def _portfolio_stats_to_row(name: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": name,
        "total_return": stats.get("total_return"),
        "annualized_return": stats.get("annualized_return"),
        "annualized_vol": stats.get("annualized_vol"),
        "sharpe": stats.get("sharpe"),
        "max_drawdown": stats.get("max_drawdown"),
        "calmar": stats.get("calmar"),
        "n_trades_total": stats.get("n_trades_total"),
    }


def _rebase_equity(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars.copy()
    out = bars.copy()
    eq = pd.to_numeric(out["equity"], errors="coerce").astype("float64")
    base = float(eq.iloc[0])
    out["equity_rebased"] = eq / base if base != 0.0 else np.nan
    out["bar_end"] = pd.to_datetime(out["bar_end"], utc=True)
    return out


def _evaluate_single_filter(
    name: str,
    filter_mode: str,
    trade_samples: pd.DataFrame,
    test_markets: dict[str, MarketDataSlice],
    params: Any,
    config: EngineConfig,
    base_strategy: StrategyDefinition,
    cfg: StudyConfig,
    split_dt: pd.Timestamp,
) -> IndicatorResult:
    valid = trade_samples[["entry_time", "trade_return", name]].dropna().copy()
    train = valid[valid["entry_time"] < split_dt].copy()
    test = valid[valid["entry_time"] >= split_dt].copy()
    x_train = train[name].to_numpy(dtype="float64")
    r_train = train["trade_return"].to_numpy(dtype="float64")
    x_test = test[name].to_numpy(dtype="float64")
    r_test = test["trade_return"].to_numpy(dtype="float64")
    xs_train, cum_train = _sorted_cum_curve(x_train, r_train)
    xs_test, cum_test = _sorted_cum_curve(x_test, r_test)

    if filter_mode == "allow_inside":
        lo, hi, best_interval_sum = _stable_interval_bootstrap(
            x_train,
            r_train,
            b=cfg.bootstrap_b,
            seed=cfg.bootstrap_seed,
        )
    elif filter_mode == "block_inside":
        lo, hi, best_interval_sum = _stable_interval_bootstrap_min(
            x_train,
            r_train,
            b=cfg.bootstrap_b,
            seed=cfg.bootstrap_seed,
        )
    else:
        raise ValueError(f"Unsupported filter_mode: {filter_mode}")
    if not (np.isfinite(lo) and np.isfinite(hi)):
        filtered_bars = pd.DataFrame(columns=["bar_end", "equity_rebased"])
        return IndicatorResult(
            name=name,
            filter_mode=filter_mode,
            sample_n_train=int(len(train)),
            sample_n_test=int(len(test)),
            lo=np.nan,
            hi=np.nan,
            best_interval_sum=np.nan,
            annualized_return=np.nan,
            sharpe=np.nan,
            max_drawdown=np.nan,
            n_trades_total=np.nan,
            sorted_x_train=xs_train,
            sorted_cumret_train=cum_train,
            sorted_x_test=xs_test,
            sorted_cumret_test=cum_test,
            filtered_portfolio_bars=filtered_bars,
        )

    filtered = _run_study_backtest(
        markets=test_markets,
        params=params,
        config=config,
        base_strategy=base_strategy,
        cfg=cfg,
        split_dt=split_dt,
        indicator_name=name,
        filter_mode=filter_mode,
        lo=lo,
        hi=hi,
        show_progress=False,
    )
    filtered_stats = filtered["portfolio_stats"]
    filtered_bars = _rebase_equity(filtered["portfolio_bars"])
    return IndicatorResult(
        name=name,
        filter_mode=filter_mode,
        sample_n_train=int(len(train)),
        sample_n_test=int(len(test)),
        lo=float(lo),
        hi=float(hi),
        best_interval_sum=float(best_interval_sum),
        annualized_return=float(filtered_stats.get("annualized_return", np.nan)),
        sharpe=float(filtered_stats.get("sharpe", np.nan)),
        max_drawdown=float(filtered_stats.get("max_drawdown", np.nan)),
        n_trades_total=float(filtered_stats.get("n_trades_total", np.nan)),
        sorted_x_train=xs_train,
        sorted_cumret_train=cum_train,
        sorted_x_test=xs_test,
        sorted_cumret_test=cum_test,
        filtered_portfolio_bars=filtered_bars,
    )


def _evaluate_indicator(
    name: str,
    trade_samples: pd.DataFrame,
    test_markets: dict[str, MarketDataSlice],
    params: Any,
    config: EngineConfig,
    base_strategy: StrategyDefinition,
    cfg: StudyConfig,
    split_dt: pd.Timestamp,
) -> list[IndicatorResult]:
    return [
        _evaluate_single_filter(
            name,
            "allow_inside",
            trade_samples,
            test_markets,
            params,
            config,
            base_strategy,
            cfg,
            split_dt,
        ),
        _evaluate_single_filter(
            name,
            "block_inside",
            trade_samples,
            test_markets,
            params,
            config,
            base_strategy,
            cfg,
            split_dt,
        ),
    ]


def _build_pdf(
    out_path: Path,
    results: list[IndicatorResult],
    baseline_test_bars: pd.DataFrame,
) -> None:
    if not results:
        return
    per_page = 4
    baseline_eq = baseline_test_bars.copy()
    ordered_results = sorted(
        results,
        key=lambda res: (
            res.name,
            -np.inf if not np.isfinite(res.annualized_return) else -float(res.annualized_return),
            res.filter_mode,
        ),
    )
    with PdfPages(out_path) as pdf:
        for i in range(0, len(ordered_results), per_page):
            chunk = ordered_results[i : i + per_page]
            fig, axes = plt.subplots(len(chunk), 3, figsize=(20, 4.5 * len(chunk)))
            if len(chunk) == 1:
                axes = np.array([axes])
            for row_idx, res in enumerate(chunk):
                ax1, ax2, ax3 = axes[row_idx]
                if len(res.sorted_x_train) > 0:
                    ax1.plot(res.sorted_x_train, res.sorted_cumret_train, color="tab:blue", lw=1.2)
                    if np.isfinite(res.lo):
                        ax1.axvline(res.lo, color="tab:red", ls="--", lw=1.0)
                    if np.isfinite(res.hi):
                        ax1.axvline(res.hi, color="tab:red", ls="--", lw=1.0)
                ax1.set_title(
                    f"{res.name} [{res.filter_mode}] | train={res.sample_n_train} | ann={res.annualized_return:.2%} | mdd={res.max_drawdown:.2%}"
                )
                ax1.set_xlabel("Indicator at Entry (Train)")
                ax1.set_ylabel("Cumulative Normalized Trade Return")
                ax1.grid(alpha=0.2)

                if len(res.sorted_x_test) > 0:
                    ax2.plot(res.sorted_x_test, res.sorted_cumret_test, color="tab:orange", lw=1.2)
                    if np.isfinite(res.lo):
                        ax2.axvline(res.lo, color="tab:red", ls="--", lw=1.0)
                    if np.isfinite(res.hi):
                        ax2.axvline(res.hi, color="tab:red", ls="--", lw=1.0)
                ax2.set_title(f"Baseline Test Trades | n={res.sample_n_test}")
                ax2.set_xlabel("Indicator at Entry (Test)")
                ax2.set_ylabel("Cumulative Normalized Trade Return")
                ax2.grid(alpha=0.2)

                if not baseline_eq.empty:
                    ax3.plot(
                        baseline_eq["bar_end"],
                        baseline_eq["equity_rebased"],
                        color="tab:gray",
                        lw=1.1,
                        label="Baseline test",
                    )
                if not res.filtered_portfolio_bars.empty:
                    ax3.plot(
                        res.filtered_portfolio_bars["bar_end"],
                        res.filtered_portfolio_bars["equity_rebased"],
                        color="tab:green",
                        lw=1.2,
                        label="Filtered test",
                    )
                ax3.set_title(f"Test Portfolio Equity [{res.filter_mode}] | trades={res.n_trades_total:.0f}")
                ax3.set_xlabel("Time")
                ax3.set_ylabel("Equity (rebased)")
                ax3.grid(alpha=0.2)
                ax3.legend(fontsize=8, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_records_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in df.to_dict(orient="records"):
            fh.write(json.dumps(record, default=str))
            fh.write("\n")


def main() -> int:
    args = _parse_args()
    indicators = DEFAULT_INDICATORS if not str(args.indicators).strip() else [x.strip() for x in str(args.indicators).split(",") if x.strip()]
    unknown = [name for name in indicators if name not in DEFAULT_INDICATORS]
    if unknown:
        raise ValueError(f"Unknown indicators requested: {unknown}")

    params = _build_params(args)
    strategy = get_strategy(args.strategy)
    cfg = StudyConfig(
        data_dir=args.data_dir.resolve(),
        symbol_specs=args.symbol_specs.resolve(),
        fx_daily=args.fx_daily.resolve(),
        symbols=[str(symbol).upper() for symbol in args.symbols],
        start=str(args.start),
        end=str(args.end),
        timeframe=str(args.timeframe),
        commission_bps=float(args.commission_bps),
        slippage_bps=float(args.slippage_bps),
        overnight_long_rate=float(args.overnight_long_rate),
        overnight_short_rate=float(args.overnight_short_rate),
        overnight_day_count=int(args.overnight_day_count),
        initial_margin_ratio=float(args.initial_margin_ratio),
        maintenance_margin_ratio=float(args.maintenance_margin_ratio),
        strategy=strategy.name,
        strategy_params=asdict(params) if is_dataclass(params) else dict(params),
        portfolio_mode=str(args.portfolio_mode),
        cash_per_trade=float(args.cash_per_trade),
        risk_per_trade=float(args.risk_per_trade),
        risk_per_trade_pct=float(args.risk_per_trade_pct),
        initial_equity=float(args.initial_equity),
        split_date=str(args.split_date).strip() or None,
        indicators=indicators,
        out_dir=args.out_dir.resolve(),
        workers=max(1, int(args.workers)),
        bootstrap_b=max(1, int(args.bootstrap_b)),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.out_dir / "study_config.json", asdict(cfg))

    symbol_specs = _load_symbol_specs(cfg.symbol_specs)
    fx_daily = _load_fx_daily(cfg.fx_daily)
    config = build_engine_config(
        default_timeframe=cfg.timeframe,
        timeframe_by_symbol={symbol: cfg.timeframe for symbol in cfg.symbols},
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        overnight_long_rate=cfg.overnight_long_rate,
        overnight_short_rate=cfg.overnight_short_rate,
        overnight_day_count=cfg.overnight_day_count,
        initial_equity=cfg.initial_equity,
        initial_margin_ratio=cfg.initial_margin_ratio,
        maintenance_margin_ratio=cfg.maintenance_margin_ratio,
        symbol_specs=symbol_specs,
        fx_daily=fx_daily,
    )
    markets = _load_markets(cfg)
    print(f"[info] loaded symbols={cfg.symbols}")
    print("[info] running baseline full backtest...")
    baseline_full = run_portfolio_backtest(
        market_by_symbol=markets,
        params=params,
        strategy=strategy,
        config=config,
        show_progress=True,
        progress_desc="Baseline full backtest",
        collect_events=False,
        portfolio_mode=cfg.portfolio_mode,
        cash_per_trade=cfg.cash_per_trade,
        risk_per_trade=cfg.risk_per_trade,
        risk_per_trade_pct=cfg.risk_per_trade_pct,
    )
    baseline_full_bars = baseline_full["portfolio_bars"].copy()
    baseline_full_bars["bar_end"] = pd.to_datetime(baseline_full_bars["bar_end"], utc=True)
    if baseline_full_bars.empty:
        raise RuntimeError("Baseline full backtest produced no portfolio bars.")

    split_dt = pd.Timestamp(cfg.split_date, tz="UTC") if cfg.split_date else pd.Timestamp(baseline_full_bars["bar_end"].iloc[len(baseline_full_bars) // 2])
    trade_samples = _build_trade_samples(
        markets=markets,
        baseline_symbol_results=baseline_full["symbol_results"],
        indicators=cfg.indicators,
    )
    warmup_bars = _estimate_required_warmup_bars(strategy.name, params, cfg.indicators)
    test_markets = _build_test_markets(markets, split_dt=split_dt, warmup_bars=warmup_bars)
    print("[info] running baseline test-period backtest...")
    baseline_test = _run_study_backtest(
        markets=test_markets,
        params=params,
        config=config,
        base_strategy=strategy,
        cfg=cfg,
        split_dt=split_dt,
        indicator_name=None,
        lo=None,
        hi=None,
        show_progress=True,
        progress_desc="Baseline test backtest",
    )
    baseline_test_bars = _rebase_equity(baseline_test["portfolio_bars"])
    baseline_test_stats = baseline_test["portfolio_stats"]
    baseline_full_stats = baseline_full["portfolio_stats"]

    print(
        f"[info] split_date={split_dt.isoformat()} indicators={cfg.indicators} "
        f"warmup_bars={warmup_bars} baseline_trades={len(trade_samples)}"
    )

    results: list[IndicatorResult] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {
            ex.submit(
                _evaluate_indicator,
                name,
                trade_samples,
                test_markets,
                params,
                config,
                strategy,
                cfg,
                split_dt,
            ): name
            for name in cfg.indicators
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating indicators"):
            result_batch = fut.result()
            results.extend(result_batch)
    results.sort(key=lambda x: (x.name, x.filter_mode))

    summary = pd.DataFrame(
        [
            {
                "strategy": cfg.strategy,
                "strategy_params": json.dumps(cfg.strategy_params, sort_keys=True),
                "indicator": res.name,
                "filter_mode": res.filter_mode,
                "sample_n_train": res.sample_n_train,
                "sample_n_test": res.sample_n_test,
                "filter_lo": res.lo,
                "filter_hi": res.hi,
                "best_interval_sum": res.best_interval_sum,
                "annualized_return_filtered": res.annualized_return,
                "sharpe_filtered": res.sharpe,
                "max_drawdown_filtered": res.max_drawdown,
                "n_trades_total_filtered": res.n_trades_total,
            }
            for res in results
        ]
    )
    _write_records_jsonl(summary, cfg.out_dir / "filter_summary.jsonl")
    _write_records_jsonl(baseline_full_bars, cfg.out_dir / "baseline_full_portfolio_bars.jsonl")
    _write_records_jsonl(baseline_test_bars, cfg.out_dir / "baseline_test_portfolio_bars.jsonl")
    _write_records_jsonl(
        pd.DataFrame([_portfolio_stats_to_row("baseline_full", baseline_full_stats)]),
        cfg.out_dir / "baseline_full_portfolio_stats.jsonl",
    )
    _write_records_jsonl(
        pd.DataFrame([_portfolio_stats_to_row("baseline_test", baseline_test_stats)]),
        cfg.out_dir / "baseline_test_portfolio_stats.jsonl",
    )
    _write_json(cfg.out_dir / "baseline_full_portfolio_stats.json", baseline_full_stats)
    _write_json(cfg.out_dir / "baseline_test_portfolio_stats.json", baseline_test_stats)

    report_path = cfg.out_dir / "portfolio_filter_study_report.pdf"
    _build_pdf(report_path, results, baseline_test_bars)
    print(f"[done] report: {report_path}")
    print(f"[done] summary: {cfg.out_dir / 'filter_summary.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
