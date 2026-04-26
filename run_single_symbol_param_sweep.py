from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import MISSING, fields
import json
from pathlib import Path
from typing import Any, get_type_hints

import matplotlib
import pandas as pd
from tqdm.auto import tqdm

from data_paths import DEFAULT_DATA_DIR
import search_params as sp
from market_data import load_symbol_market_data
from run_backtest import _load_fx_daily, _load_symbol_specs, _save_portfolio_pnl_plot, build_engine_config
from search_params import run_portfolio_backtest
from strategies import STRATEGIES, get_strategy

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run single-symbol parameter scans using either one-at-a-time sweeps or a full-grid heatmap."
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--symbol-specs", type=Path, default=Path("symbol_specs.json"))
    p.add_argument("--fx-daily", type=Path, default=Path("data/fx_daily/fx_daily_2012_2025.csv"))
    p.add_argument("--symbol", type=str, default="GBPJPY")
    p.add_argument("--start", type=str, default="20120101")
    p.add_argument("--end", type=str, default="20200101")
    p.add_argument("--timeframe", type=str, default="3H")
    p.add_argument("--strategy", type=str, default="supertrend", choices=sorted(STRATEGIES))
    p.add_argument(
        "--mode",
        type=str,
        default="sweep",
        choices=["sweep", "heatmap"],
        help="sweep scans one default-grid parameter at a time; heatmap scans the full supertrend grid and plots max_recovery_time",
    )
    p.add_argument(
        "--base-param",
        action="append",
        default=[],
        help="Base strategy param in the form name=value. Used only in sweep mode, where each default-grid parameter is scanned one at a time while the others stay fixed.",
    )
    p.add_argument("--commission-bps", type=float, default=0.35)
    p.add_argument("--slippage-bps", type=float, default=0.3)
    p.add_argument("--overnight-long-rate", type=float, default=0.0)
    p.add_argument("--overnight-short-rate", type=float, default=0.0)
    p.add_argument("--overnight-day-count", type=int, default=360)
    p.add_argument("--initial-margin-ratio", type=float, default=0.01)
    p.add_argument("--maintenance-margin-ratio", type=float, default=0.005)
    p.add_argument("--initial-equity", type=float, default=1000000.0)
    p.add_argument(
        "--portfolio-mode",
        type=str,
        default="fixed_risk",
        choices=["fixed_cash", "fixed_risk", "fixed_risk_pct"],
    )
    p.add_argument("--cash-per-trade", type=float, default=10000.0)
    p.add_argument("--risk-per-trade", type=float, default=10000.0)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    p.add_argument(
        "--opposite-signal-action",
        type=str,
        default="close_and_reverse",
        choices=["close_only", "close_and_reverse"],
    )
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _parse_param_assignments(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        text = str(item).strip()
        if "=" not in text:
            raise ValueError(f"Invalid --base-param {item!r}; expected name=value")
        name, raw = text.split("=", 1)
        key = name.strip()
        if not key:
            raise ValueError(f"Invalid --param {item!r}; missing name")
        out[key] = raw.strip()
    return out


def _coerce_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _coerce_value(raw: str, target_type: Any) -> Any:
    if target_type is bool:
        return _coerce_bool(raw)
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    if target_type is str:
        return str(raw)
    return raw


def _resolve_base_param_values(strategy_name: str, overrides: dict[str, str]) -> dict[str, Any]:
    strategy = get_strategy(strategy_name)
    param_type_hints = get_type_hints(strategy.params_type)
    valid_names = {field.name for field in fields(strategy.params_type)}
    extra = sorted(set(overrides) - valid_names)
    if extra:
        raise ValueError(f"Unknown strategy params for {strategy_name!r}: {extra}")

    values: dict[str, Any] = {}
    missing_required: list[str] = []
    for field in fields(strategy.params_type):
        if field.name in overrides:
            target_type = param_type_hints.get(field.name, str)
            values[field.name] = _coerce_value(overrides[field.name], target_type)
        elif field.default is not MISSING:
            values[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            values[field.name] = field.default_factory()  # type: ignore[misc]
        else:
            missing_required.append(field.name)
    if missing_required:
        raise ValueError(
            "Missing required base params for single-symbol sweep: "
            + ", ".join(missing_required)
            + ". Provide them via repeated --base-param name=value."
        )
    strategy.make_params(**values)
    return values


def _slugify_value(value: Any) -> str:
    text = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe.replace(".", "_")


def _default_out_dir(strategy_name: str, mode: str, timeframe: str, base_params: dict[str, Any]) -> Path:
    ordered_values = [_slugify_value(base_params[name]) for name in base_params]
    suffix = "_".join(ordered_values) if ordered_values else "default"
    return Path("results") / (
        f"single_symbol_param_{_slugify_value(mode)}_{strategy_name}_{_slugify_value(timeframe)}_{suffix}"
    )


def _default_heatmap_out_dir(strategy_name: str, mode: str, symbol: str, timeframe: str) -> Path:
    return Path("results") / (
        f"single_symbol_param_{_slugify_value(mode)}_{strategy_name}_{str(symbol).upper()}_{_slugify_value(timeframe)}"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_records_jsonl(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in frame.to_dict(orient="records"):
            fh.write(json.dumps(record, default=str))
            fh.write("\n")


def _load_records_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _value_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    text = str(value)
    try:
        offset = pd.tseries.frequencies.to_offset(text.strip().lower())
        delta = pd.Timedelta(offset)
        return (1, float(delta / pd.Timedelta(minutes=1)))
    except (ValueError, TypeError):
        return (2, text)


def _build_pnl_comparison_pdf(results: pd.DataFrame, output_path: Path, initial_equity: float) -> None:
    if results.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        cover = plt.figure(figsize=(11, 8.5))
        cover.text(0.05, 0.92, "Single-Parameter Sweep PnL Comparison", fontsize=18, weight="bold")
        cover.text(0.05, 0.86, f"Runs: {len(results)}", fontsize=11)
        cover.text(
            0.05,
            0.80,
            "Each following page overlays portfolio PnL curves for one parameter while all other base params stay fixed.",
            fontsize=10,
        )
        grouped = results.groupby("varied_param", sort=False)["varied_value"].apply(list).to_dict()
        y = 0.72
        for param_name, values in grouped.items():
            values_text = ", ".join(str(v) for v in sorted(values, key=_value_sort_key))
            cover.text(0.05, y, f"{param_name}: {values_text}", fontsize=9)
            y -= 0.04
            if y < 0.08:
                break
        cover.tight_layout()
        pdf.savefig(cover)
        plt.close(cover)

        for varied_param, group in results.groupby("varied_param", sort=False):
            fig, ax = plt.subplots(figsize=(12, 6))
            for row in group.sort_values("varied_value", key=lambda s: s.map(_value_sort_key)).itertuples(index=False):
                pnl_path = Path(row.run_dir) / "portfolio_bars.jsonl"
                bars = _load_records_jsonl(pnl_path)
                if bars.empty:
                    continue
                bars["bar_end"] = pd.to_datetime(bars["bar_end"], utc=True)
                bars["pnl"] = pd.to_numeric(bars["equity"], errors="coerce") - float(initial_equity)
                ax.plot(bars["bar_end"], bars["pnl"], linewidth=1.1, label=str(row.varied_value), alpha=0.9)
            ax.axhline(0.0, color="gray", linewidth=0.9, linestyle="--")
            ax.set_title(f"{varied_param} sweep")
            ax.set_xlabel("Time")
            ax.set_ylabel("Portfolio PnL")
            ax.grid(True, alpha=0.3)
            ax.legend(title=varied_param, fontsize=8, title_fontsize=9, loc="best", ncol=2)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _build_max_recovery_time_heatmap(results: pd.DataFrame, output_path: Path) -> None:
    required_cols = {"atr_len", "atr_mult", "max_recovery_time"}
    if results.empty or not required_cols.issubset(results.columns):
        return
    grid = (
        results.loc[:, ["atr_len", "atr_mult", "max_recovery_time"]]
        .dropna(subset=["atr_len", "atr_mult", "max_recovery_time"])
        .copy()
    )
    if grid.empty:
        return
    grid["atr_len"] = pd.to_numeric(grid["atr_len"], errors="coerce")
    grid["atr_mult"] = pd.to_numeric(grid["atr_mult"], errors="coerce")
    grid["max_recovery_time"] = pd.to_numeric(grid["max_recovery_time"], errors="coerce")
    grid = grid.dropna(subset=["atr_len", "atr_mult", "max_recovery_time"])
    if grid.empty:
        return
    pivot = grid.pivot(index="atr_mult", columns="atr_len", values="max_recovery_time").sort_index().sort_index(axis=1)
    if pivot.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Supertrend max_recovery_time heatmap")
    ax.set_xlabel("atr_len")
    ax.set_ylabel("atr_mult")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("max_recovery_time")
    for row_idx, atr_mult in enumerate(pivot.index):
        for col_idx, atr_len in enumerate(pivot.columns):
            value = pivot.loc[atr_mult, atr_len]
            if pd.isna(value):
                continue
            ax.text(col_idx, row_idx, f"{float(value):.0f}", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _rebuild_worker_market() -> dict[str, object]:
    if sp._GRID_WORKER_MARKET_BY_SYMBOL is None or sp._GRID_WORKER_CONFIG is None:  # type: ignore[attr-defined]
        raise RuntimeError("Shared market context is not initialized")
    if sp._GRID_WORKER_MARKET_SHARED is None:  # type: ignore[attr-defined]
        return sp._GRID_WORKER_MARKET_BY_SYMBOL  # type: ignore[attr-defined]

    market_by_symbol: dict[str, object] = {}
    for symbol, bars in sp._GRID_WORKER_MARKET_BY_SYMBOL.items():  # type: ignore[attr-defined]
        shared = sp._GRID_WORKER_MARKET_SHARED[symbol]  # type: ignore[attr-defined]
        data = shared["data"]
        m1 = pd.DataFrame(
            data,
            columns=["bid_open", "bid_low", "bid_close", "ask_open", "ask_high", "ask_close"],
            index=pd.to_datetime(shared["ts_ns"], unit="ns", utc=True),
        )
        market_by_symbol[symbol] = type("SharedMarketSlice", (), {"m1": m1, "bars": bars})()
    return market_by_symbol


def _run_single_sweep_task(task: dict[str, Any]) -> dict[str, Any]:
    strategy = get_strategy(sp._GRID_WORKER_STRATEGY_NAME)  # type: ignore[attr-defined]
    params = strategy.make_params(**task["param_values"])
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

    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_records_jsonl(result["portfolio_bars"], run_dir / "portfolio_bars.jsonl")
    _save_portfolio_pnl_plot(result["portfolio_bars"], run_dir, float(sp._GRID_WORKER_CONFIG.initial_equity))  # type: ignore[attr-defined]
    _write_json(run_dir / "portfolio_stats.json", result["portfolio_stats"])
    _write_json(run_dir / "symbol_stats.json", result["symbol_stats"].to_dict(orient="records"))
    _write_json(
        run_dir / "run_args.json",
        {
            "symbol": task["symbol"],
            "timeframe": task["timeframe"],
            "strategy": strategy.name,
            "varied_param": task["varied_param"],
            "varied_value": task["varied_value"],
            "base_params": task["base_params"],
            "strategy_params": task["param_values"],
        },
    )

    row = {
        "symbol": task["symbol"],
        "timeframe": task["timeframe"],
        "strategy": strategy.name,
        "varied_param": task["varied_param"],
        "varied_value": task["varied_value"],
        "run_dir": str(run_dir),
        **result["portfolio_stats"],
    }
    return row


def _run_single_heatmap_task(task: dict[str, Any]) -> dict[str, Any]:
    strategy = get_strategy(sp._GRID_WORKER_STRATEGY_NAME)  # type: ignore[attr-defined]
    params = strategy.make_params(**task["param_values"])
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
        "symbol": task["symbol"],
        "timeframe": task["timeframe"],
        "strategy": strategy.name,
        **task["param_values"],
        **result["portfolio_stats"],
    }


def _build_tasks(
    *,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    base_params: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    strategy = get_strategy(strategy_name)
    tasks: list[dict[str, Any]] = []
    for param_name, values in strategy.default_grid().items():
        for value in values:
            param_values = dict(base_params)
            param_values[param_name] = value
            run_dir = out_dir / param_name / _slugify_value(value)
            tasks.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "varied_param": param_name,
                    "varied_value": value,
                    "base_params": dict(base_params),
                    "param_values": param_values,
                    "run_dir": str(run_dir),
                }
            )
    return tasks


def _build_heatmap_tasks(
    *,
    strategy_name: str,
    symbol: str,
    timeframe: str,
) -> list[dict[str, Any]]:
    if strategy_name != "supertrend":
        raise ValueError("Heatmap mode only supports strategy='supertrend'")
    grid = get_strategy(strategy_name).default_grid()
    atr_lens = list(grid.get("atr_len", []))
    atr_mults = list(grid.get("atr_mult", []))
    if not atr_lens or not atr_mults:
        raise ValueError("Supertrend default_grid must include atr_len and atr_mult for heatmap mode")
    tasks: list[dict[str, Any]] = []
    for atr_len in atr_lens:
        for atr_mult in atr_mults:
            tasks.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "param_values": {
                        "atr_len": atr_len,
                        "atr_mult": atr_mult,
                    },
                }
            )
    return tasks


def main() -> int:
    args = _parse_args()
    args.symbol = str(args.symbol).upper()

    strategy = get_strategy(args.strategy)
    base_param_overrides = _parse_param_assignments(args.base_param)
    if args.mode == "heatmap":
        if args.strategy != "supertrend":
            raise ValueError("Heatmap mode only supports --strategy supertrend")
        if base_param_overrides:
            raise ValueError("Heatmap mode scans the full supertrend grid; do not pass --base-param")
        base_params: dict[str, Any] = {}
        if args.out_dir is None:
            args.out_dir = _default_heatmap_out_dir(args.strategy, args.mode, args.symbol, args.timeframe)
    else:
        base_params = _resolve_base_param_values(args.strategy, base_param_overrides)
        if args.out_dir is None:
            args.out_dir = _default_out_dir(args.strategy, args.mode, args.timeframe, base_params)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    market = load_symbol_market_data(
        data_dir=args.data_dir,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
    )
    markets = {args.symbol: market}
    symbol_specs = _load_symbol_specs(args.symbol_specs)
    fx_daily = _load_fx_daily(args.fx_daily)
    config = build_engine_config(
        default_timeframe=args.timeframe,
        timeframe_by_symbol={args.symbol: args.timeframe},
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
    if args.mode == "heatmap":
        tasks = _build_heatmap_tasks(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
    else:
        tasks = _build_tasks(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            base_params=base_params,
            out_dir=args.out_dir,
        )

    _write_json(
        args.out_dir / "run_args.json",
        {
            "mode": args.mode,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "strategy": args.strategy,
            "start": args.start,
            "end": args.end,
            "base_params": base_params,
            "portfolio_mode": args.portfolio_mode,
            "cash_per_trade": args.cash_per_trade,
            "risk_per_trade": args.risk_per_trade,
            "risk_per_trade_pct": args.risk_per_trade_pct,
            "max_workers": args.max_workers,
            "opposite_signal_action": args.opposite_signal_action,
        },
    )

    rows: list[dict[str, Any]] = []
    progress_desc = "Running supertrend heatmap grid" if args.mode == "heatmap" else "Running single-parameter sweeps"
    progress = tqdm(total=len(tasks), desc=progress_desc)
    shared_blocks = []
    try:
        if args.max_workers == 1:
            sp._init_grid_worker(
                markets,
                config,
                strategy.name,
                args.portfolio_mode,
                args.cash_per_trade,
                args.risk_per_trade,
                args.risk_per_trade_pct,
            )
            for task in tasks:
                rows.append(_run_single_heatmap_task(task) if args.mode == "heatmap" else _run_single_sweep_task(task))
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
                    strategy.name,
                    args.portfolio_mode,
                    args.cash_per_trade,
                    args.risk_per_trade,
                    args.risk_per_trade_pct,
                ),
            ) as executor:
                task_fn = _run_single_heatmap_task if args.mode == "heatmap" else _run_single_sweep_task
                futures = [executor.submit(task_fn, task) for task in tasks]
                for future in as_completed(futures):
                    rows.append(future.result())
                    progress.update(1)
    finally:
        progress.close()
        for shm in shared_blocks:
            try:
                shm.close()
            finally:
                shm.unlink()

    results = pd.DataFrame(rows)
    if not results.empty:
        if args.mode == "heatmap":
            results = results.sort_values(["atr_len", "atr_mult"]).reset_index(drop=True)
            results_path = args.out_dir / "heatmap_results.jsonl"
        else:
            results = results.sort_values(["varied_param", "varied_value"]).reset_index(drop=True)
            results_path = args.out_dir / "sweep_results.jsonl"
        with results_path.open("w", encoding="utf-8") as fh:
            for record in results.to_dict(orient="records"):
                fh.write(json.dumps(record, default=str))
                fh.write("\n")
        best_by_annualized_return = (
            results.sort_values("annualized_return", ascending=False).iloc[0].to_dict()
            if "annualized_return" in results.columns
            else None
        )
        best_by_calmar = (
            results.sort_values("calmar", ascending=False).iloc[0].to_dict()
            if "calmar" in results.columns
            else None
        )
        best_by_min_recovery_time = (
            results.sort_values("max_recovery_time", ascending=True).iloc[0].to_dict()
            if "max_recovery_time" in results.columns
            else None
        )
        _write_json(
            args.out_dir / "summary.json",
            {
                "mode": args.mode,
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "strategy": args.strategy,
                "start": args.start,
                "end": args.end,
                "base_params": base_params,
                "n_runs": int(len(results)),
                "best_by_annualized_return": best_by_annualized_return,
                "best_by_calmar": best_by_calmar,
                "best_by_min_recovery_time": best_by_min_recovery_time,
            },
        )
        if args.mode == "heatmap":
            _build_max_recovery_time_heatmap(
                results,
                args.out_dir / "max_recovery_time_heatmap.png",
            )
        else:
            _build_pnl_comparison_pdf(
                results,
                args.out_dir / "pnl_comparison.pdf",
                initial_equity=float(args.initial_equity),
            )
    else:
        _write_json(
            args.out_dir / "summary.json",
            {"mode": args.mode, "symbol": args.symbol, "strategy": args.strategy, "n_runs": 0},
        )

    print(json.dumps({"out_dir": str(args.out_dir), "n_runs": int(len(rows))}, indent=2, default=str))
    print(f"\nSaved outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
