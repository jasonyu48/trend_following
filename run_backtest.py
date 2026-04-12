from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from execution_engine import EngineConfig
from market_data import load_symbol_market_data
from search_params import run_grid_search, run_portfolio_backtest
from strategies import STRATEGIES, get_strategy

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SYMBOLS = ["GBPJPY", "USDJPY", "USDCHF", "XAUUSD", "XAGUSD", "HK50", "JP225", "USDCNH"] #, "USDCNH"


def normalize_symbols(symbols: list[str]) -> list[str]:
    return [str(symbol).upper() for symbol in symbols]


def load_symbol_timeframes(
    symbols: list[str],
    default_timeframe: str,
    symbol_timeframes_path: Path | None = None,
) -> dict[str, str]:
    normalized_symbols = normalize_symbols(symbols)
    timeframe_by_symbol = {symbol: str(default_timeframe) for symbol in normalized_symbols}
    if symbol_timeframes_path is None:
        return timeframe_by_symbol
    payload = json.loads(symbol_timeframes_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in symbol timeframe file: {symbol_timeframes_path}")
    for symbol, timeframe in payload.items():
        symbol_key = str(symbol).upper()
        if symbol_key in timeframe_by_symbol:
            timeframe_by_symbol[symbol_key] = str(timeframe)
    return timeframe_by_symbol


def load_markets_for_symbols(
    data_dir: Path,
    symbols: list[str],
    timeframe_by_symbol: dict[str, str],
    start: str,
    end: str,
    progress_desc: str = "Loading market data",
) -> dict[str, object]:
    out: dict[str, object] = {}
    for symbol in tqdm(normalize_symbols(symbols), desc=progress_desc):
        out[symbol] = load_symbol_market_data(
            data_dir=data_dir,
            symbol=symbol,
            timeframe=timeframe_by_symbol[symbol],
            start=start,
            end=end,
        )
    return out


def build_strategy_params(strategy_name: str, param_values: dict[str, Any]) -> object:
    strategy = get_strategy(strategy_name)
    kwargs = {name: param_values[name] for name in strategy.param_names if name in param_values}
    return strategy.make_params(**kwargs)


def build_engine_config(
    *,
    default_timeframe: str,
    timeframe_by_symbol: dict[str, str],
    commission_bps: float,
    slippage_bps: float,
    overnight_long_rate: float,
    overnight_short_rate: float,
    overnight_day_count: int,
    initial_equity: float,
    initial_margin_ratio: float,
    maintenance_margin_ratio: float,
    symbol_specs: dict[str, dict[str, object]],
    fx_daily: pd.DataFrame,
) -> EngineConfig:
    return EngineConfig(
        timeframe=default_timeframe,
        timeframe_by_symbol=dict(timeframe_by_symbol),
        commission_bps=commission_bps,
        commission_bps_by_symbol={
            symbol: float(spec["commission_bps"])
            for symbol, spec in symbol_specs.items()
            if "commission_bps" in spec
        },
        slippage_bps=slippage_bps,
        overnight_long_rate=overnight_long_rate,
        overnight_short_rate=overnight_short_rate,
        overnight_long_rate_by_symbol={
            symbol: float(spec["overnight_long_rate"])
            for symbol, spec in symbol_specs.items()
            if "overnight_long_rate" in spec
        },
        overnight_short_rate_by_symbol={
            symbol: float(spec["overnight_short_rate"])
            for symbol, spec in symbol_specs.items()
            if "overnight_short_rate" in spec
        },
        overnight_day_count=overnight_day_count,
        initial_equity=initial_equity,
        initial_margin_ratio=initial_margin_ratio,
        maintenance_margin_ratio=maintenance_margin_ratio,
        initial_margin_ratio_by_symbol={
            symbol: float(spec["initial_margin_ratio"])
            for symbol, spec in symbol_specs.items()
            if "initial_margin_ratio" in spec
        },
        maintenance_margin_ratio_by_symbol={
            symbol: float(spec["maintenance_margin_ratio"])
            for symbol, spec in symbol_specs.items()
            if "maintenance_margin_ratio" in spec
        },
        account_currency="USD",
        symbol_specs=symbol_specs,
        fx_daily=fx_daily,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trend-following backtest with configurable signal timeframe and M1 bid/ask execution.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--symbol-specs", type=Path, default=Path("symbol_specs.json"))
    p.add_argument("--fx-daily", type=Path, default=Path("data/fx_daily/fx_daily_2012_2025.csv"))
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", type=str, default="20120101")
    p.add_argument("--end", type=str, default="20250101")
    p.add_argument("--timeframe", type=str, default="4H")
    p.add_argument("--symbol-timeframes", type=Path, default=None, help="optional JSON mapping of symbol -> timeframe")
    p.add_argument("--commission-bps", type=float, default=0.35)
    p.add_argument("--slippage-bps", type=float, default=0.3)
    p.add_argument("--overnight-long-rate", type=float, default=0.0, help="annualized overnight financing rate applied to long positions; positive=cost, negative=credit")
    p.add_argument("--overnight-short-rate", type=float, default=0.0, help="annualized overnight financing rate applied to short positions; positive=cost, negative=credit")
    p.add_argument("--overnight-day-count", type=int, default=360, help="day-count basis used for overnight financing accrual")
    p.add_argument("--initial-margin-ratio", type=float, default=0.01, help="fallback initial margin ratio if not specified in symbol specs")
    p.add_argument("--maintenance-margin-ratio", type=float, default=0.005, help="fallback maintenance margin ratio if not specified in symbol specs")
    p.add_argument("--strategy", type=str, default="ma_atr_breakout", choices=sorted(STRATEGIES))
    p.add_argument("--ma-len", type=int, default=5)
    p.add_argument("--atr-len", type=int, default=60)
    p.add_argument("--atr-mult", type=float, default=1.5)
    p.add_argument("--stop-lookback", type=int, default=50)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--divergence-lookback", type=int, default=8)
    p.add_argument("--slope-len", type=int, default=6)
    p.add_argument("--slope-threshold", type=float, default=0.05)
    p.add_argument("--slope-norm-mode", type=str, default="atr", choices=["atr", "close_pct", "none"])
    p.add_argument("--anchor-lookback", type=int, default=5)
    p.add_argument("--session-start-hour-utc", type=int, default=0)
    p.add_argument("--session-end-hour-utc", type=int, default=0)
    p.add_argument("--search", action="store_true")
    p.add_argument(
        "--neighbor-radius",
        type=int,
        default=2,
        help="taxi-cab distance used when computing neighborhood median scores during parameter search",
    )
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--initial-equity", type=float, default=1000000.0)
    p.add_argument(
        "--portfolio-mode",
        type=str,
        default="fixed_risk_pct",
        choices=["fixed_cash", "fixed_risk", "fixed_risk_pct"],
    )
    p.add_argument("--cash-per-trade", type=float, default=10000.0)
    p.add_argument("--risk-per-trade", type=float, default=10000.0)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    p.add_argument("--out-dir", type=Path, default=Path("results/ma_atr_breakout_best5_60_1.5_50"))
    return p.parse_args()


def _load_markets(args: argparse.Namespace) -> dict[str, object]:
    return load_markets_for_symbols(
        data_dir=args.data_dir,
        symbols=args.symbols,
        timeframe_by_symbol=args.timeframe_by_symbol,
        start=args.start,
        end=args.end,
    )


def _default_grid(strategy_name: str) -> dict[str, list]:
    return get_strategy(strategy_name).default_grid()


def _build_params(args: argparse.Namespace) -> object:
    return build_strategy_params(args.strategy, vars(args))


def _load_symbol_specs(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(symbol).upper(): dict(spec)
        for symbol, spec in payload.items()
        if str(symbol) != "_meta" and isinstance(spec, dict)
    }


def _load_fx_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError(f"FX daily file missing 'date' column: {path}")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.set_index("date").sort_index()
    return df


def _save_portfolio_pnl_plot(portfolio_bars: pd.DataFrame, out_dir: Path, initial_equity: float) -> None:
    if portfolio_bars.empty:
        return
    df = portfolio_bars.copy()
    df["bar_end"] = pd.to_datetime(df["bar_end"], utc=True)
    df["pnl"] = df["equity"] - float(initial_equity)
    dd = df["equity"] / df["equity"].cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(df["bar_end"], df["pnl"], linewidth=1.5, color="#1565C0", label="Portfolio PnL")
    axes[0].axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    axes[0].set_title("Portfolio PnL")
    axes[0].set_ylabel("PnL")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(df["bar_end"], dd * 100, 0, color="#E53935", alpha=0.4)
    axes[1].plot(df["bar_end"], dd * 100, color="#E53935", linewidth=0.8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "portfolio_pnl.png", dpi=150)
    plt.close(fig)


def _save_symbol_chart(
    symbol: str,
    market_bars: pd.DataFrame,
    sym_bars: pd.DataFrame,
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if market_bars.empty:
        return

    t_bars = pd.to_datetime(market_bars.index, utc=True)

    feat = features_df.copy()
    feat["bar_end"] = pd.to_datetime(feat["bar_end"], utc=True)
    feat = feat.set_index("bar_end")

    sb = sym_bars.copy()
    if not sb.empty:
        sb["bar_end"] = pd.to_datetime(sb["bar_end"], utc=True)
        sb = sb.set_index("bar_end").sort_index()

    fig, axes = plt.subplots(3, 1, figsize=(70, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    ax_p, ax_e, ax_pos = axes

    # ── price + channel ──────────────────────────────────────────────────────
    ax_p.plot(t_bars, market_bars["close"].to_numpy(), color="#424242",
              linewidth=0.7, label="Close", zorder=2)

    upper = feat["upper"].reindex(t_bars) if "upper" in feat.columns else None
    lower = feat["lower"].reindex(t_bars) if "lower" in feat.columns else None
    if upper is not None and lower is not None:
        ax_p.step(t_bars, upper.to_numpy(), where="post", color="#1565C0",
                  linewidth=1.0, linestyle="--", alpha=0.8, label="Upper")
        ax_p.step(t_bars, lower.to_numpy(), where="post", color="#B71C1C",
                  linewidth=1.0, linestyle="--", alpha=0.8, label="Lower")
        ax_p.fill_between(t_bars, upper.to_numpy(), lower.to_numpy(),
                          alpha=0.04, color="#1565C0", step="post")
    if "ma" in feat.columns:
        ax_p.step(t_bars, feat["ma"].reindex(t_bars).to_numpy(), where="post", color="#6A1B9A",
                  linewidth=0.8, linestyle=":", alpha=0.8, label="MA")

    # trailing stop — draw as a 4H staircase and extend to trade entry/exit.
    if not sb.empty and "stop_price" in sb.columns and "position_sign" in sb.columns and not trades_df.empty:
        plot_stop_label = True
        trades_plot = trades_df.copy()
        trades_plot["entry_time"] = pd.to_datetime(trades_plot["entry_time"], utc=True)
        trades_plot["exit_time"] = pd.to_datetime(trades_plot["exit_time"], utc=True, errors="coerce")
        for trade in trades_plot.itertuples(index=False):
            if pd.isna(trade.exit_time):
                continue
            side_sign = 1 if trade.side == "long" else -1
            trade_stop = sb.loc[
                (sb.index >= trade.entry_time)
                & (sb.index <= trade.exit_time)
                & (sb["position_sign"] == side_sign),
                "stop_price",
            ].dropna()
            if trade_stop.empty:
                continue
            stop_times = [trade.entry_time, *trade_stop.index.to_list(), trade.exit_time]
            stop_values = [float(trade_stop.iloc[0]), *trade_stop.astype("float64").to_list(), float(trade_stop.iloc[-1])]
            ax_p.step(
                stop_times,
                stop_values,
                where="post",
                color="#EF6C00",
                linewidth=1.3,
                alpha=0.9,
                label="Stop" if plot_stop_label else None,
                zorder=3,
            )
            plot_stop_label = False

    # entry / exit markers — vertical lines spanning the full price axis
    if not trades_df.empty:
        for side, color in [("long", "#00897B"), ("short", "#E53935")]:
            subset = trades_df[trades_df["side"] == side]
            if subset.empty:
                continue
            for t in pd.to_datetime(subset["entry_time"], utc=True):
                ax_p.axvline(t, color=color, linewidth=0.6, linestyle="-", alpha=0.6, zorder=3)
            valid = subset.dropna(subset=["exit_time"])
            for t in pd.to_datetime(valid["exit_time"], utc=True):
                ax_p.axvline(t, color=color, linewidth=0.6, linestyle="--", alpha=0.6, zorder=3)

    ax_p.set_title(symbol, fontsize=11)
    ax_p.set_ylabel("Price")
    ax_p.legend(fontsize=7, ncol=6, loc="upper left")
    ax_p.grid(True, alpha=0.25)

    # ── equity ───────────────────────────────────────────────────────────────
    if not sb.empty and "equity" in sb.columns:
        ax_e.plot(sb.index, sb["equity"].to_numpy(), color="#1565C0", linewidth=1.0)
        ax_e.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
        ax_e.set_ylabel("Equity")
        ax_e.grid(True, alpha=0.25)

    # ── position indicator ───────────────────────────────────────────────────
    if not sb.empty and "position_sign" in sb.columns:
        pos = sb["position_sign"].fillna(0).to_numpy()
        ax_pos.fill_between(sb.index, pos, 0, where=pos > 0,
                            color="#00897B", alpha=0.5, label="Long")
        ax_pos.fill_between(sb.index, pos, 0, where=pos < 0,
                            color="#E53935", alpha=0.5, label="Short")
        ax_pos.set_yticks([-1, 0, 1])
        ax_pos.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)
        ax_pos.set_ylabel("Position")
        ax_pos.legend(fontsize=7, loc="upper left")
        ax_pos.grid(True, alpha=0.25)

    ax_pos.set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(out_dir / f"{symbol}_chart.png", dpi=300)
    plt.close(fig)


def _save_symbol_equity_chart(symbol_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol, result in symbol_results.items():
        bars = result["bars"]
        if bars.empty:
            continue
        t = pd.to_datetime(bars["bar_end"], utc=True)
        ax.plot(t, bars["equity"].to_numpy(), linewidth=1.0, label=symbol, alpha=0.85)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Symbol Equity Curves")
    ax.set_ylabel("Equity (normalized)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "symbol_equity.png", dpi=150)
    plt.close(fig)


def _save_monthly_heatmap(portfolio_bars: pd.DataFrame, out_dir: Path) -> None:
    if portfolio_bars.empty:
        return
    df = portfolio_bars.copy()
    df["bar_end"] = pd.to_datetime(df["bar_end"], utc=True)
    df = df.set_index("bar_end")
    monthly = df["equity"].resample("ME").last()
    monthly_ret = monthly.pct_change().dropna() * 100  # percent

    years = sorted(monthly_ret.index.year.unique())
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grid = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for ts, val in monthly_ret.items():
            if ts.year == year:
                grid[i, ts.month - 1] = float(val)

    vmax = float(np.nanmax(np.abs(grid))) or 1.0
    fig, ax = plt.subplots(figsize=(13, max(3, len(years) * 0.55 + 1.5)))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    for i in range(len(years)):
        for j in range(12):
            val = grid[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, label="Monthly Return (%)")
    ax.set_title("Monthly Returns Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_heatmap.png", dpi=150)
    plt.close(fig)


def _save_trade_analysis(symbol_results: dict, out_dir: Path) -> None:
    frames = []
    for symbol, result in symbol_results.items():
        t = result["trades"]
        if not t.empty:
            f = t.copy()
            f["symbol"] = symbol
            frames.append(f)
    if not frames:
        return
    trades = pd.concat(frames, ignore_index=True)
    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").dropna()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # PnL distribution
    axes[0].hist(pnl, bins=40, color="#1565C0", edgecolor="white", linewidth=0.4)
    axes[0].axvline(0, color="#E53935", linewidth=1.2, linestyle="--")
    axes[0].set_title("Trade PnL Distribution")
    axes[0].set_xlabel("Net PnL")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Win rate by symbol
    sym_stats = (
        trades.groupby("symbol")["net_pnl"]
        .apply(lambda g: float((pd.to_numeric(g, errors="coerce") > 0).mean()) * 100)
        .reset_index(name="win_rate")
        .sort_values("symbol")
    )
    x = range(len(sym_stats))
    bars = axes[1].bar(x, sym_stats["win_rate"], color="#4CAF50", alpha=0.75, width=0.6)
    axes[1].axhline(50, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(sym_stats["symbol"], rotation=40, ha="right", fontsize=8)
    axes[1].set_title("Win Rate by Symbol (%)")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, sym_stats["win_rate"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    # Long vs short trade count by symbol
    counts = trades.groupby(["symbol", "side"]).size().unstack(fill_value=0).sort_index()
    x2 = np.arange(len(counts))
    w = 0.35
    if "long" in counts.columns:
        axes[2].bar(x2 - w / 2, counts["long"], width=w, color="#00897B", alpha=0.75, label="Long")
    if "short" in counts.columns:
        axes[2].bar(x2 + w / 2, counts["short"], width=w, color="#E53935", alpha=0.75, label="Short")
    axes[2].set_xticks(list(x2))
    axes[2].set_xticklabels(counts.index, rotation=40, ha="right", fontsize=8)
    axes[2].set_title("Trade Count by Symbol")
    axes[2].set_ylabel("Count")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "trade_analysis.png", dpi=150)
    plt.close(fig)


def _portfolio_construction_metadata(args: argparse.Namespace) -> dict[str, object]:
    if args.portfolio_mode == "fixed_cash":
        return {
            "mode": "fixed_cash",
            "cash_per_trade": float(args.cash_per_trade),
            "initial_equity": float(args.initial_equity),
            "initial_margin_ratio": float(args.initial_margin_ratio),
            "maintenance_margin_ratio": float(args.maintenance_margin_ratio),
            "description": "Integrated portfolio-level simple-interest mode. Each entry consumes a fixed amount of initial margin cash, with position notional determined by the configured initial margin ratio; the portfolio is checked on every M1 bar against maintenance margin and any breach is liquidated on the next minute open.",
        }
    if args.portfolio_mode == "fixed_risk":
        return {
            "mode": "fixed_risk",
            "initial_equity": float(args.initial_equity),
            "risk_per_trade": float(args.risk_per_trade),
            "initial_margin_ratio": float(args.initial_margin_ratio),
            "maintenance_margin_ratio": float(args.maintenance_margin_ratio),
            "description": "Integrated portfolio-level cash mode. Each entry sizes quantity from the strategy's initial risk reference so the planned loss is capped at a fixed cash amount; the portfolio is checked on every M1 bar against maintenance margin and any breach is liquidated on the next minute open.",
        }
    if args.portfolio_mode == "fixed_risk_pct":
        return {
            "mode": "fixed_risk_pct",
            "initial_equity": float(args.initial_equity),
            "risk_per_trade_pct": float(args.risk_per_trade_pct),
            "initial_margin_ratio": float(args.initial_margin_ratio),
            "maintenance_margin_ratio": float(args.maintenance_margin_ratio),
            "description": "Integrated portfolio-level compounding mode. Each entry sizes quantity from the strategy's initial risk reference so the planned loss is capped at a fixed percentage of current portfolio equity; the portfolio is checked on every M1 bar against maintenance margin and any breach is liquidated on the next minute open.",
        }
    raise ValueError(f"Unsupported portfolio mode: {args.portfolio_mode}")


def _save_run_args(args: argparse.Namespace, out_dir: Path) -> None:
    params = _build_params(args)
    payload = {
        "argv": sys.argv,
        "args": vars(args),
        "symbol_timeframes": dict(args.timeframe_by_symbol),
        "strategy": args.strategy,
        "strategy_params": asdict(params) if is_dataclass(params) else params,
        "portfolio_construction": _portfolio_construction_metadata(args),
    }
    (out_dir / "run_args.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_records_json(df: pd.DataFrame, path: Path) -> None:
    path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, default=str), encoding="utf-8")


def _write_records_jsonl(df: pd.DataFrame, path: Path) -> None:
    records = df.to_dict(orient="records")
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str))
            fh.write("\n")


def save_backtest_outputs(
    *,
    result: dict[str, Any],
    markets: dict[str, object],
    out_dir: Path,
    initial_equity: float,
) -> None:
    portfolio_bars = result["portfolio_bars"]
    symbol_stats = result["symbol_stats"]
    portfolio_stats = pd.DataFrame([result["portfolio_stats"]])

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_records_jsonl(portfolio_bars, out_dir / "portfolio_bars.jsonl")
    _write_records_jsonl(symbol_stats, out_dir / "symbol_stats.jsonl")
    _write_records_json(portfolio_stats, out_dir / "portfolio_stats.json")

    _save_portfolio_pnl_plot(portfolio_bars, out_dir, initial_equity)
    _save_symbol_equity_chart(result["symbol_results"], out_dir)
    _save_monthly_heatmap(portfolio_bars, out_dir)
    _save_trade_analysis(result["symbol_results"], out_dir)

    for symbol, sym_result in tqdm(result["symbol_results"].items(), desc="Writing outputs"):
        _write_records_jsonl(sym_result["bars"], out_dir / f"{symbol}_bars.jsonl")
        _write_records_jsonl(sym_result["trades"], out_dir / f"{symbol}_trades.jsonl")
        _save_symbol_chart(
            symbol=symbol,
            market_bars=markets[symbol].bars,
            sym_bars=sym_result["bars"],
            trades_df=sym_result["trades"],
            features_df=sym_result["features"],
            out_dir=out_dir,
        )


def main() -> int:
    args = _parse_args()
    args.symbols = normalize_symbols(args.symbols)
    args.timeframe_by_symbol = load_symbol_timeframes(
        symbols=args.symbols,
        default_timeframe=args.timeframe,
        symbol_timeframes_path=args.symbol_timeframes,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _save_run_args(args, args.out_dir)

    markets = _load_markets(args)
    symbol_specs = _load_symbol_specs(args.symbol_specs)
    fx_daily = _load_fx_daily(args.fx_daily)
    config = build_engine_config(
        default_timeframe=args.timeframe,
        timeframe_by_symbol=args.timeframe_by_symbol,
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
    strategy = get_strategy(args.strategy)

    if args.search:
        result = run_grid_search(
            market_by_symbol=markets,
            strategy=strategy,
            param_grid=_default_grid(args.strategy),
            config=config,
            neighbor_radius=args.neighbor_radius,
            max_workers=args.max_workers,
            portfolio_mode=args.portfolio_mode,
            cash_per_trade=args.cash_per_trade,
            risk_per_trade=args.risk_per_trade,
            risk_per_trade_pct=args.risk_per_trade_pct,
        )
        path = args.out_dir / "grid_search_results.jsonl"
        _write_records_jsonl(result, path)
        preview_n = min(20, len(result))
        payload = {
            "grid_search_results_path": str(path),
            "n_rows_total": int(len(result)),
            "preview_n_rows": preview_n,
            "preview": result.head(preview_n).to_dict(orient="records"),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    params = _build_params(args)
    result = run_portfolio_backtest(
        market_by_symbol=markets,
        params=params,
        strategy=strategy,
        config=config,
        show_progress=True,
        progress_desc="Backtesting symbols",
        portfolio_mode=args.portfolio_mode,
        cash_per_trade=args.cash_per_trade,
        risk_per_trade=args.risk_per_trade,
        risk_per_trade_pct=args.risk_per_trade_pct,
    )

    save_backtest_outputs(
        result=result,
        markets=markets,
        out_dir=args.out_dir,
        initial_equity=args.initial_equity,
    )

    print(json.dumps(result["portfolio_stats"], indent=2, default=str))
    print(f"\nSaved outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
