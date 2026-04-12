from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import MISSING, asdict, fields, is_dataclass
import heapq
from itertools import product
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from execution_engine import (
    EngineConfig,
    Trade,
    _apply_adverse_slippage,
    _bars_per_year,
    _commission_rate_for_symbol,
    _entry_fill_from_trigger,
    _initial_margin_ratio_for_symbol,
    _maintenance_margin_ratio_for_symbol,
    _overnight_day_count,
    _overnight_rate_for_side,
    _stop_fill,
    _timeframe_for_symbol,
)
from indicators import bar_performance_stats, resampled_bar_performance_stats, trade_performance_stats
from strategies.base import StrategyDefinition
from strategies.registry import get_strategy

_GRID_WORKER_MARKET_BY_SYMBOL: dict[str, object] | None = None
_GRID_WORKER_CONFIG: EngineConfig | None = None
_GRID_WORKER_MARKET_SHARED: dict[str, dict] | None = None
_GRID_WORKER_SHARED_HANDLES: list[SharedMemory] = []
_GRID_WORKER_STRATEGY_NAME: str = "ma_atr_breakout"
_GRID_WORKER_PORTFOLIO_MODE: str = "fixed_risk"
_GRID_WORKER_CASH_PER_TRADE: float = 1.0
_GRID_WORKER_RISK_PER_TRADE: float = 0.01
_GRID_WORKER_RISK_PER_TRADE_PCT: float = 0.01
USDJPY_RANGE_RETURN_COLUMN = "usd_jpy_range_return_20140201_20140815"
USDJPY_RANGE_PENALTY_COLUMN = "usd_jpy_range_penalty"
USDJPY_RANGE_PENALTY_POINTS = 0.15
TRADE_COUNT_PENALTY_COLUMN = "trade_count_penalty"
TRADE_COUNT_BOTTOM_FRACTION = 0.30
TRADE_COUNT_BOTTOM_PENALTY = 0.08
USDJPY_RANGE_START = pd.Timestamp("2014-02-01", tz="UTC")
USDJPY_RANGE_END = pd.Timestamp("2014-08-15 23:59:59", tz="UTC")


def _params_to_dict(params: Any) -> dict[str, Any]:
    if is_dataclass(params):
        return asdict(params)
    if isinstance(params, dict):
        return dict(params)
    raise TypeError(f"Unsupported params type: {type(params)!r}")


def _profitable_symbol_ratio(symbol_stats: pd.DataFrame) -> float:
    if symbol_stats.empty:
        return np.nan
    if "annualized_return" in symbol_stats.columns:
        return float((pd.to_numeric(symbol_stats["annualized_return"], errors="coerce") > 0).mean())
    if "net_pnl_total" in symbol_stats.columns:
        return float((pd.to_numeric(symbol_stats["net_pnl_total"], errors="coerce") > 0).mean())
    if "total_return" in symbol_stats.columns:
        return float((pd.to_numeric(symbol_stats["total_return"], errors="coerce") > 0).mean())
    return np.nan


def _all_trades_df(symbol_results: dict[str, dict]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, result in symbol_results.items():
        trades = result["trades"]
        if trades is None or trades.empty:
            continue
        frame = trades.copy()
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _equity_return_between(
    bars: pd.DataFrame,
    start: pd.Timestamp = USDJPY_RANGE_START,
    end: pd.Timestamp = USDJPY_RANGE_END,
) -> float:
    if bars.empty or "bar_end" not in bars or "equity" not in bars:
        return np.nan
    bar_end = pd.DatetimeIndex(pd.to_datetime(bars["bar_end"], utc=True))
    equity = pd.to_numeric(bars["equity"], errors="coerce")
    start_mask = bar_end >= start
    end_mask = bar_end <= end
    if not start_mask.any() or not end_mask.any():
        return np.nan
    start_equity = equity.loc[start_mask].iloc[0]
    end_equity = equity.loc[end_mask].iloc[-1]
    if not np.isfinite(start_equity) or not np.isfinite(end_equity) or np.isclose(start_equity, 0.0):
        return np.nan
    return float(end_equity / start_equity - 1.0)


def run_integrated_cash_backtest(
    market_by_symbol: dict[str, object],
    params: Any,
    strategy: StrategyDefinition,
    config: EngineConfig,
    show_progress: bool = False,
    progress_desc: str | None = None,
    collect_events: bool = True,
    cash_per_trade: float = 1.0,
    fixed_risk_cash: float = 0.0,
    fixed_risk_pct: float = 0.01,
    position_sizing_mode: str = "fixed_cash",
) -> dict[str, pd.DataFrame | dict[str, float]]:
    if position_sizing_mode == "fixed_cash":
        if cash_per_trade <= 0 or cash_per_trade > float(config.initial_equity):
            raise ValueError("cash_per_trade must be > 0 and <= initial_equity in fixed_cash mode")
    elif position_sizing_mode == "fixed_risk":
        if fixed_risk_cash <= 0 or fixed_risk_cash > float(config.initial_equity):
            raise ValueError("fixed_risk_cash must be > 0 and <= initial_equity in fixed_risk mode")
    elif position_sizing_mode == "fixed_risk_pct":
        if fixed_risk_pct <= 0 or fixed_risk_pct > 1.0:
            raise ValueError("fixed_risk_pct must be > 0 and <= 1.0 in fixed_risk_pct mode")
    else:
        raise ValueError(f"Unsupported position sizing mode: {position_sizing_mode}")

    overnight_day_count = _overnight_day_count(config)
    account_currency = str(config.account_currency).upper()
    symbol_specs = {
        str(symbol).upper(): dict(spec)
        for symbol, spec in (config.symbol_specs or {}).items()
    }
    fx_daily = config.fx_daily if isinstance(config.fx_daily, pd.DataFrame) else None
    fx_dates_ns = np.array([], dtype="int64")
    fx_cols: dict[str, np.ndarray] = {}
    if fx_daily is not None and not fx_daily.empty:
        fx_index = pd.DatetimeIndex(fx_daily.index)
        if fx_index.tz is None:
            fx_index = fx_index.tz_localize("UTC")
        else:
            fx_index = fx_index.tz_convert("UTC")
        fx_frame = fx_daily.copy()
        fx_frame.index = fx_index.normalize()
        fx_frame = fx_frame.sort_index()
        fx_dates_ns = fx_frame.index.asi8
        fx_cols = {
            str(col).upper(): fx_frame[col].to_numpy(dtype="float64", copy=False)
            for col in fx_frame.columns
        }

    symbol_order = {symbol: idx for idx, symbol in enumerate(market_by_symbol)}
    states: dict[str, dict] = {}
    heap: list[tuple[int, int, str]] = []
    skipped_rows: list[dict[str, object]] = []
    maintenance_margin_call_count = 0
    portfolio_rows_by_end: dict[int, dict[str, float | int]] = {}

    def _symbol_equity(state: dict, allocated_equity: float | None = None) -> float:
        open_pnl = 0.0
        if state["side"] != "flat" and allocated_equity is not None:
            open_pnl = float(allocated_equity) - float(state["entry_equity_before"])
        return 1.0 + (float(state["realized_pnl_total"]) + open_pnl) / float(config.initial_equity)

    def _spec_str(symbol: str, key: str, default: str) -> str:
        spec = symbol_specs.get(str(symbol).upper(), {})
        value = spec.get(key, default)
        return str(value).upper()

    def _spec_float(symbol: str, key: str, default: float) -> float:
        spec = symbol_specs.get(str(symbol).upper(), {})
        value = spec.get(key, default)
        out = float(value)
        if not np.isfinite(out):
            raise ValueError(f"Invalid {key} for symbol {symbol}: {value!r}")
        return out

    def _fx_rate_to_account(currency: str, ts_value: int) -> float:
        ccy = str(currency).upper()
        if ccy == account_currency:
            return 1.0
        if fx_dates_ns.size == 0:
            raise ValueError(f"No FX daily data configured for {ccy}->{account_currency}")
        col = f"{ccy}{account_currency}"
        rates = fx_cols.get(col)
        if rates is None:
            raise ValueError(f"Missing FX conversion column {col!r} in FX daily data")
        day_ns = int(pd.Timestamp(ts_value, unit="ns", tz="UTC").normalize().value)
        pos = int(np.searchsorted(fx_dates_ns, day_ns, side="right") - 1)
        if pos < 0:
            raise ValueError(f"No FX rate available on or before {pd.Timestamp(ts_value, unit='ns', tz='UTC')} for {col}")
        rate = float(rates[pos])
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"Invalid FX rate {rate!r} for {col} at index {pos}")
        return rate

    def _round_quantity_to_lot_step(quantity: float, state: dict) -> float:
        lot_step = float(state["lot_step"])
        min_lot = float(state["min_lot"])
        if not np.isfinite(quantity) or quantity <= 0:
            return 0.0
        if lot_step <= 0:
            return float(quantity) if quantity >= min_lot else 0.0
        steps = np.floor((float(quantity) + 1e-12) / lot_step)
        rounded = float(steps * lot_step)
        if rounded + 1e-12 < min_lot:
            return 0.0
        return float(np.round(rounded, 10))

    def _sync_stop(state: dict) -> None:
        if state["current_window"] >= state["n_windows"]:
            state["stop_price"] = None
            return
        if strategy.execution_style != "trailing_stop":
            state["stop_price"] = None
            return
        if state["side"] == "long":
            stop_value = state["long_stop"][state["current_window"]]
            new_stop = float(stop_value) if np.isfinite(stop_value) else None
            if new_stop is not None and state["stop_price"] is not None:
                new_stop = max(new_stop, state["stop_price"])
            state["stop_price"] = new_stop
        elif state["side"] == "short":
            stop_value = state["short_stop"][state["current_window"]]
            new_stop = float(stop_value) if np.isfinite(stop_value) else None
            if new_stop is not None and state["stop_price"] is not None:
                new_stop = min(new_stop, state["stop_price"])
            state["stop_price"] = new_stop
        else:
            state["stop_price"] = None

    def _finalize_bar(state: dict, window_idx: int, allocated_equity: float | None) -> None:
        if state["trade"] is not None:
            state["trade"].bars_held += 1
        state["bar_rows"].append(
            {
                "bar_end": int(state["window_end"][window_idx]),
                "position": state["side"],
                "position_sign": 1 if state["side"] == "long" else -1 if state["side"] == "short" else 0,
                "stop_price": np.nan if state["stop_price"] is None else float(state["stop_price"]),
                "equity": float(_symbol_equity(state, allocated_equity)),
                "allocated_equity": np.nan if allocated_equity is None else float(allocated_equity),
                "margin_in_use": 0.0 if state["side"] == "flat" else float(state["margin_required"]),
                "maintenance_margin_required": (
                    0.0 if state["side"] == "flat" else float(state["maintenance_margin_required"])
                ),
                "overnight_pnl_total": float(state["trade_overnight_pnl_total"]),
            }
        )
        _capture_portfolio_bar(int(state["window_end"][window_idx]))

    def _record_skip(symbol: str, ts_value: int, reason: str, side: str, entry_reason: str, state: dict) -> None:
        skipped_rows.append(
            {
                "symbol": symbol,
                "entry_time": pd.Timestamp(ts_value, unit="ns", tz="UTC"),
                "skip_reason": reason,
                "side": side,
                "entry_reason": entry_reason,
            }
        )
        if collect_events:
            state["event_rows"].append(
                {"timestamp": int(ts_value), "event": f"skip_{reason}", "price": np.nan, "equity": np.nan}
            )

    def _marked_equity_for_risk(state: dict, data: dict | None) -> float:
        if state["side"] == "flat":
            return 0.0
        del data
        if state["entry_equity_after_cost"] is not None:
            return float(
                state["entry_equity_after_cost"]
                + state["trade_price_pnl_total"]
                + state["trade_overnight_pnl_total"]
            )
        return float(state["entry_equity_before"])

    def _apply_overnight_financing(state: dict, mark_price: float, ts_value: int) -> None:
        nonlocal account_cash
        if state["side"] == "flat" or state["quantity"] <= 0:
            return
        current_day_ns = int(pd.Timestamp(ts_value, unit="ns", tz="UTC").normalize().value)
        last_day_ns = state["last_financing_day_ns"]
        if last_day_ns is None:
            state["last_financing_day_ns"] = current_day_ns
            return
        elapsed_days = int((current_day_ns - int(last_day_ns)) // pd.Timedelta(days=1).value)
        if elapsed_days <= 0:
            return
        overnight_rate = _overnight_rate_for_side(config, state["symbol"], state["side"])
        if overnight_rate != 0.0:
            quote_to_account = _fx_rate_to_account(state["quote_currency"], int(ts_value))
            notional_in_account = (
                abs(float(state["quantity"]))
                * float(state["contract_multiplier"])
                * float(mark_price)
                * quote_to_account
            )
            financing_pnl = -notional_in_account * float(overnight_rate) * float(elapsed_days) / float(overnight_day_count)
            if np.isfinite(financing_pnl):
                account_cash += float(financing_pnl)
                state["trade_overnight_pnl_total"] += float(financing_pnl)
        state["last_financing_day_ns"] = current_day_ns

    def _maintenance_margin_for_state(state: dict, data: dict | None = None, ts_value: int | None = None) -> float:
        if state["side"] == "flat" or state["quantity"] <= 0:
            return 0.0
        if data is not None:
            ref_price = float(data["b_close"]) if state["side"] == "long" else float(data["a_close"])
        elif state["last_maintenance_reference_price"] is not None:
            ref_price = float(state["last_maintenance_reference_price"])
        else:
            ref_price = float(state["entry_fill"])
        ref_ts = int(ts_value) if ts_value is not None else int(state["last_fx_timestamp"])
        quote_to_account = _fx_rate_to_account(state["quote_currency"], ref_ts)
        return (
            float(state["quantity"])
            * float(state["contract_multiplier"])
            * ref_price
            * float(state["maintenance_margin_ratio"])
            * quote_to_account
        )

    def _portfolio_margin_in_use() -> float:
        return float(
            sum(float(state["margin_required"]) for state in states.values() if state["side"] != "flat")
        )

    def _portfolio_free_cash() -> float:
        return float(account_cash - _portfolio_margin_in_use())

    def _capture_portfolio_bar(bar_end: int) -> None:
        margin_in_use = _portfolio_margin_in_use()
        maintenance_required = float(
            sum(
                float(state["maintenance_margin_required"])
                for state in states.values()
                if state["side"] != "flat"
            )
        )
        equity = float(account_cash)
        portfolio_rows_by_end[int(bar_end)] = {
            "bar_end": int(bar_end),
            "equity": equity,
            "n_active": int(sum(state["side"] != "flat" for state in states.values())),
            "capital_invested": float(margin_in_use / equity) if equity != 0 else np.nan,
            "cash_available": float(equity - margin_in_use),
            "margin_in_use": margin_in_use,
            "maintenance_margin_required": maintenance_required,
            "overnight_pnl_total": float(
                sum(float(state["trade_overnight_pnl_total"]) for state in states.values() if state["side"] != "flat")
            ),
        }

    def _mark_price_from_quotes(side: str, bid_price: float, ask_price: float) -> float:
        return float(bid_price) if side == "long" else float(ask_price)

    def _apply_variation_margin(state: dict, mark_price: float, ts_value: int) -> None:
        nonlocal account_cash
        if state["side"] == "flat" or state["quantity"] <= 0:
            return
        prev_mark = state["last_mark_price"]
        if prev_mark is None or not np.isfinite(prev_mark):
            state["last_mark_price"] = float(mark_price)
            state["last_fx_timestamp"] = int(ts_value)
            return
        pnl_to_account = _fx_rate_to_account(state["pnl_currency"], int(ts_value))
        delta = (
            float(state["quantity"]) * float(state["contract_multiplier"]) * (float(mark_price) - float(prev_mark)) * pnl_to_account
            if state["side"] == "long"
            else float(state["quantity"]) * float(state["contract_multiplier"]) * (float(prev_mark) - float(mark_price)) * pnl_to_account
        )
        if np.isfinite(delta):
            account_cash += float(delta)
            state["trade_price_pnl_total"] += float(delta)
        state["last_mark_price"] = float(mark_price)
        state["last_fx_timestamp"] = int(ts_value)

    def _reset_flat_state(state: dict) -> None:
        state["side"] = "flat"
        state["quantity"] = 0.0
        state["entry_fill"] = None
        state["entry_equity_before"] = 0.0
        state["entry_equity_after_cost"] = None
        state["margin_required"] = 0.0
        state["maintenance_margin_required"] = 0.0
        state["last_marked_equity"] = 0.0
        state["last_mark_price"] = None
        state["last_fx_timestamp"] = None
        state["last_financing_day_ns"] = None
        state["last_maintenance_reference_price"] = None
        state["pending_margin_liquidation"] = False
        state["stop_price"] = None
        state["trade"] = None
        state["trade_price_pnl_total"] = 0.0
        state["trade_overnight_pnl_total"] = 0.0
        state["last_allocated_equity"] = None

    def _close_open_position(
        state: dict,
        ts_value: int,
        fill: float,
        exit_reason: str,
        event_name: str | None,
    ) -> None:
        nonlocal account_cash
        _apply_variation_margin(state, float(fill), int(ts_value))
        commission_rate = _commission_rate_for_symbol(config, state["symbol"])
        quote_to_account = _fx_rate_to_account(state["quote_currency"], int(ts_value))
        gross_equity = float(
            state["entry_equity_after_cost"]
            + state["trade_price_pnl_total"]
            + state["trade_overnight_pnl_total"]
        )
        exit_cost = (
            abs(state["quantity"] * float(state["contract_multiplier"]) * fill) * commission_rate * quote_to_account
        )
        exit_equity = gross_equity - exit_cost
        state["trade"].exit_time = int(ts_value)
        state["trade"].exit_price = float(fill)
        state["trade"].exit_equity = float(exit_equity)
        state["trade"].gross_pnl = float(gross_equity - state["entry_equity_before"])
        state["trade"].net_pnl = float(exit_equity - state["entry_equity_before"])
        state["trade"].overnight_pnl = float(state["trade_overnight_pnl_total"])
        state["trade"].exit_reason = exit_reason
        state["trade_rows"].append(asdict(state["trade"]))
        if collect_events and event_name is not None:
            state["event_rows"].append(
                {"timestamp": int(ts_value), "event": event_name, "price": float(fill), "equity": float(exit_equity)}
            )
        state["realized_pnl_total"] += float(exit_equity - state["entry_equity_before"])
        account_cash -= float(exit_cost)
        _reset_flat_state(state)

    market_items = list(market_by_symbol.items())
    market_iterator = market_items
    if show_progress:
        market_iterator = tqdm(market_items, desc=f"{progress_desc or 'Integrated backtest'}: prepare", leave=False)

    for symbol, market in market_iterator:
        features = strategy.compute_features(market.bars, params)
        schedule = strategy.build_signal_schedule(market.bars, features, params)
        contract_multiplier = _spec_float(symbol, "contract_multiplier", 1.0)
        min_lot = _spec_float(symbol, "min_lot", 0.0)
        lot_step = _spec_float(symbol, "lot_step", 0.0)
        quote_currency = _spec_str(symbol, "quote_currency", account_currency)
        pnl_currency = _spec_str(symbol, "pnl_currency", quote_currency)
        state = {
            "symbol": symbol,
            "features": features.reset_index(names="bar_end"),
            "schedule": schedule,
            "bars_src": market.bars,
            "ts_ns": market.m1.index.asi8,
            "bid_open": market.m1["bid_open"].to_numpy(dtype="float64", copy=False),
            "bid_low": market.m1["bid_low"].to_numpy(dtype="float64", copy=False),
            "bid_close": market.m1["bid_close"].to_numpy(dtype="float64", copy=False),
            "ask_open": market.m1["ask_open"].to_numpy(dtype="float64", copy=False),
            "ask_high": market.m1["ask_high"].to_numpy(dtype="float64", copy=False),
            "ask_close": market.m1["ask_close"].to_numpy(dtype="float64", copy=False),
            "window_start": pd.DatetimeIndex(schedule["window_start"]).asi8 if not schedule.empty else np.array([], dtype="int64"),
            "window_end": pd.DatetimeIndex(schedule["window_end"]).asi8 if not schedule.empty else np.array([], dtype="int64"),
            "long_trigger": schedule["long_trigger"].to_numpy(dtype="float64", copy=False) if not schedule.empty else np.array([], dtype="float64"),
            "short_trigger": schedule["short_trigger"].to_numpy(dtype="float64", copy=False) if not schedule.empty else np.array([], dtype="float64"),
            "allow_long_entry": (
                schedule["allow_long_entry"].to_numpy(dtype=bool, copy=False)
                if (not schedule.empty and "allow_long_entry" in schedule.columns)
                else np.ones(len(schedule), dtype=bool)
            ),
            "allow_short_entry": (
                schedule["allow_short_entry"].to_numpy(dtype=bool, copy=False)
                if (not schedule.empty and "allow_short_entry" in schedule.columns)
                else np.ones(len(schedule), dtype=bool)
            ),
            "long_stop": (
                schedule["long_stop"].to_numpy(dtype="float64", copy=False)
                if (not schedule.empty and "long_stop" in schedule.columns)
                else np.full(len(schedule), np.nan, dtype="float64")
            ),
            "short_stop": (
                schedule["short_stop"].to_numpy(dtype="float64", copy=False)
                if (not schedule.empty and "short_stop" in schedule.columns)
                else np.full(len(schedule), np.nan, dtype="float64")
            ),
            "n_windows": len(schedule),
            "current_window": 0,
            "minute_idx": 0,
            "side": "flat",
            "quantity": 0.0,
            "entry_fill": None,
            "entry_equity_before": 0.0,
            "entry_equity_after_cost": None,
            "contract_multiplier": contract_multiplier,
            "quote_currency": quote_currency,
            "pnl_currency": pnl_currency,
            "min_lot": min_lot,
            "lot_step": lot_step,
            "initial_margin_ratio": _initial_margin_ratio_for_symbol(config, symbol),
            "maintenance_margin_ratio": _maintenance_margin_ratio_for_symbol(config, symbol),
            "margin_required": 0.0,
            "maintenance_margin_required": 0.0,
            "last_marked_equity": 0.0,
            "last_mark_price": None,
            "last_fx_timestamp": None,
            "last_financing_day_ns": None,
            "last_maintenance_reference_price": None,
            "pending_margin_liquidation": False,
            "stop_price": None,
            "trade": None,
            "trade_price_pnl_total": 0.0,
            "trade_overnight_pnl_total": 0.0,
            "realized_pnl_total": 0.0,
            "last_allocated_equity": None,
            "event_rows": [],
            "trade_rows": [],
            "bar_rows": [],
            "active_minutes": 0,
            "total_minutes": int(len(market.m1)),
        }
        if state["n_windows"] > 0:
            state["minute_idx"] = int(np.searchsorted(state["ts_ns"], state["window_start"][0], side="right"))
            _sync_stop(state)
            if state["minute_idx"] < len(state["ts_ns"]):
                heapq.heappush(heap, (int(state["ts_ns"][state["minute_idx"]]), symbol_order[symbol], symbol))
        else:
            state["minute_idx"] = len(state["ts_ns"])
        states[symbol] = state

    cash_budget = float(cash_per_trade)
    account_cash = float(config.initial_equity)
    batch_progress = None
    if show_progress:
        total_batches = max((len(market.m1) for _, market in market_items), default=0)
        batch_progress = tqdm(total=total_batches, desc=progress_desc or "Integrated backtest", leave=False)

    while heap:
        ts_value = heap[0][0]
        batch: list[str] = []
        while heap and heap[0][0] == ts_value:
            _, _, symbol = heapq.heappop(heap)
            batch.append(symbol)
        batch.sort(key=lambda sym: symbol_order[sym])
        if batch_progress is not None:
            batch_progress.update(1)

        minute_data: dict[str, dict] = {}
        eligible_symbols: list[str] = []
        for symbol in batch:
            state = states[symbol]
            while state["current_window"] < state["n_windows"] and ts_value > int(state["window_end"][state["current_window"]]):
                _finalize_bar(state, state["current_window"], state["last_allocated_equity"])
                state["current_window"] += 1
                _sync_stop(state)
            if state["current_window"] >= state["n_windows"]:
                continue
            if ts_value <= int(state["window_start"][state["current_window"]]):
                continue

            minute_idx = state["minute_idx"]
            minute_data[symbol] = {
                "b_open": float(state["bid_open"][minute_idx]),
                "b_low": float(state["bid_low"][minute_idx]),
                "b_close": float(state["bid_close"][minute_idx]),
                "a_open": float(state["ask_open"][minute_idx]),
                "a_high": float(state["ask_high"][minute_idx]),
                "a_close": float(state["ask_close"][minute_idx]),
                "had_position_before": state["side"] != "flat",
                "action_taken": False,
            }
            eligible_symbols.append(symbol)

        batch_had_margin_liquidation = False
        for symbol in eligible_symbols:
            state = states[symbol]
            data = minute_data[symbol]
            data["exited_on_opposite"] = False
            data["forced_entry_side"] = None
            if state["side"] != "flat":
                overnight_mark_price = _mark_price_from_quotes(state["side"], data["b_open"], data["a_open"])
                _apply_overnight_financing(state, overnight_mark_price, int(ts_value))
            if state["side"] != "flat" and state["pending_margin_liquidation"]:
                fill = (
                    _apply_adverse_slippage(float(data["b_open"]), "sell", config)
                    if state["side"] == "long"
                    else _apply_adverse_slippage(float(data["a_open"]), "buy", config)
                )
                event_name = "exit_long_maintenance_margin" if state["side"] == "long" else "exit_short_maintenance_margin"
                _close_open_position(
                    state=state,
                    ts_value=int(ts_value),
                    fill=float(fill),
                    exit_reason="maintenance_margin_liquidation",
                    event_name=event_name,
                )
                data["action_taken"] = True
                batch_had_margin_liquidation = True
                continue
            if state["side"] == "long":
                if strategy.execution_style == "trailing_stop" and state["stop_price"] is not None and data["b_low"] <= float(state["stop_price"]):
                    fill = _stop_fill(data["b_open"], data["a_open"], float(state["stop_price"]), "long", config)
                    _close_open_position(
                        state=state,
                        ts_value=int(ts_value),
                        fill=float(fill),
                        exit_reason="trailing_stop",
                        event_name="exit_long_stop",
                    )
                    data["action_taken"] = True
                elif strategy.execution_style == "opposite_breakout" and data["b_low"] <= float(state["short_trigger"][state["current_window"]]):
                    fill = _entry_fill_from_trigger(
                        data["a_open"],
                        data["b_open"],
                        float(state["short_trigger"][state["current_window"]]),
                        "short",
                        config,
                    )
                    _close_open_position(
                        state=state,
                        ts_value=int(ts_value),
                        fill=float(fill),
                        exit_reason="opposite_breakout",
                        event_name="exit_long_reverse",
                    )
                    data["action_taken"] = True
                    data["exited_on_opposite"] = True
                    data["forced_entry_side"] = "short"
            elif state["side"] == "short":
                if strategy.execution_style == "trailing_stop" and state["stop_price"] is not None and data["a_high"] >= float(state["stop_price"]):
                    fill = _stop_fill(data["b_open"], data["a_open"], float(state["stop_price"]), "short", config)
                    _close_open_position(
                        state=state,
                        ts_value=int(ts_value),
                        fill=float(fill),
                        exit_reason="trailing_stop",
                        event_name="exit_short_stop",
                    )
                    data["action_taken"] = True
                elif strategy.execution_style == "opposite_breakout" and data["a_high"] >= float(state["long_trigger"][state["current_window"]]):
                    fill = _entry_fill_from_trigger(
                        data["a_open"],
                        data["b_open"],
                        float(state["long_trigger"][state["current_window"]]),
                        "long",
                        config,
                    )
                    _close_open_position(
                        state=state,
                        ts_value=int(ts_value),
                        fill=float(fill),
                        exit_reason="opposite_breakout",
                        event_name="exit_short_reverse",
                    )
                    data["action_taken"] = True
                    data["exited_on_opposite"] = True
                    data["forced_entry_side"] = "long"

        batch_risk_budget = None
        if position_sizing_mode == "fixed_risk_pct":
            batch_risk_budget = float(account_cash) * float(fixed_risk_pct)

        for symbol in eligible_symbols:
            state = states[symbol]
            data = minute_data[symbol]
            if batch_had_margin_liquidation:
                continue
            if data["action_taken"] and not (config.allow_flip_same_minute or data["exited_on_opposite"]):
                continue
            if state["side"] != "flat":
                continue

            ts_utc = pd.Timestamp(ts_value, unit="ns", tz="UTC")
            forced_entry_side = data["forced_entry_side"]
            if forced_entry_side is None and not strategy.is_entry_allowed(ts_utc, params):
                continue

            if forced_entry_side is not None:
                side = forced_entry_side
                long_hit = side == "long"
                short_hit = side == "short"
            else:
                long_hit = (
                    bool(state["allow_long_entry"][state["current_window"]])
                    and data["a_high"] >= float(state["long_trigger"][state["current_window"]])
                )
                short_hit = (
                    bool(state["allow_short_entry"][state["current_window"]])
                    and data["b_low"] <= float(state["short_trigger"][state["current_window"]])
                )
                if long_hit and short_hit:
                    if collect_events:
                        state["event_rows"].append(
                            {"timestamp": int(ts_value), "event": "ambiguous_entry", "price": np.nan, "equity": np.nan}
                        )
                    continue
                if not long_hit and not short_hit:
                    continue
                side = "long" if long_hit else "short"

            entry_reason = "opposite_breakout_reverse" if forced_entry_side is not None else ("upper_breakout" if long_hit else "lower_breakout")
            trigger = float(state["long_trigger"][state["current_window"]]) if long_hit else float(state["short_trigger"][state["current_window"]])
            fill = _entry_fill_from_trigger(data["a_open"], data["b_open"], trigger, side, config)
            trail_stop_raw = (
                float(state["long_stop"][state["current_window"]]) if long_hit else float(state["short_stop"][state["current_window"]])
            )
            stop_price = trail_stop_raw
            if not np.isfinite(stop_price):
                opposite_trigger = (
                    float(state["short_trigger"][state["current_window"]]) if long_hit else float(state["long_trigger"][state["current_window"]])
                )
                print(
                    f"[stop_fallback] symbol={symbol} ts={ts_utc.isoformat()} side={side} "
                    f"trail_stop_nonfinite={trail_stop_raw!r} using_opposite_trigger={opposite_trigger!r}",
                    flush=True,
                )
                stop_price = opposite_trigger

            initial_margin_ratio = float(state["initial_margin_ratio"])
            maintenance_margin_ratio = float(state["maintenance_margin_ratio"])
            if fill <= 0 or not np.isfinite(initial_margin_ratio) or not np.isfinite(maintenance_margin_ratio):
                _record_skip(symbol, ts_value, "invalid_margin_requirement", side, entry_reason, state)
                continue
            quote_to_account = _fx_rate_to_account(state["quote_currency"], int(ts_value))
            pnl_to_account = _fx_rate_to_account(state["pnl_currency"], int(ts_value))
            notional_per_lot = float(state["contract_multiplier"]) * float(fill) * quote_to_account
            margin_per_unit = notional_per_lot * initial_margin_ratio
            commission_rate = _commission_rate_for_symbol(config, symbol)
            entry_commission_per_unit = commission_rate * notional_per_lot
            required_cash_per_unit = margin_per_unit + entry_commission_per_unit
            if not np.isfinite(required_cash_per_unit) or required_cash_per_unit <= 0:
                _record_skip(symbol, ts_value, "invalid_margin_requirement", side, entry_reason, state)
                continue

            if position_sizing_mode == "fixed_cash":
                quantity = float(cash_budget) / float(required_cash_per_unit)
            else:
                target_risk_cash = (
                    float(fixed_risk_cash)
                    if position_sizing_mode == "fixed_risk"
                    else float(batch_risk_budget)
                )
                if not np.isfinite(target_risk_cash) or target_risk_cash <= 0:
                    _record_skip(symbol, ts_value, "invalid_risk_budget", side, entry_reason, state)
                    continue
                stop_fill = _stop_fill(stop_price, stop_price, stop_price, side, config)
                price_risk = (
                    abs(float(fill) - float(stop_fill)) * float(state["contract_multiplier"]) * pnl_to_account
                )
                stop_notional_per_lot = float(state["contract_multiplier"]) * float(stop_fill) * quote_to_account
                per_unit_risk = price_risk + commission_rate * (notional_per_lot + stop_notional_per_lot)
                if fill <= 0 or not np.isfinite(per_unit_risk) or per_unit_risk <= 0:
                    _record_skip(symbol, ts_value, "invalid_stop_distance", side, entry_reason, state)
                    continue
                quantity = float(target_risk_cash) / float(per_unit_risk)
            quantity = _round_quantity_to_lot_step(float(quantity), state)
            if not np.isfinite(quantity) or quantity <= 0:
                _record_skip(symbol, ts_value, "invalid_margin_requirement", side, entry_reason, state)
                continue

            margin_required = float(quantity) * margin_per_unit
            entry_cost = float(quantity) * entry_commission_per_unit
            deployable = margin_required
            entry_equity_before = margin_required + entry_cost

            if _portfolio_free_cash() < entry_equity_before:
                _record_skip(symbol, ts_value, "no_available_cash", side, entry_reason, state)
                continue

            account_cash -= float(entry_cost)

            state["quantity"] = float(quantity)
            state["entry_fill"] = float(fill)
            state["entry_equity_before"] = float(entry_equity_before)
            state["entry_equity_after_cost"] = float(deployable)
            state["margin_required"] = float(margin_required)
            state["maintenance_margin_required"] = (
                float(quantity) * float(state["contract_multiplier"]) * float(fill) * maintenance_margin_ratio * quote_to_account
            )
            state["last_marked_equity"] = float(deployable)
            state["last_mark_price"] = float(fill)
            state["last_fx_timestamp"] = int(ts_value)
            state["last_financing_day_ns"] = int(pd.Timestamp(ts_value, unit="ns", tz="UTC").normalize().value)
            state["last_maintenance_reference_price"] = float(fill)
            state["pending_margin_liquidation"] = False
            state["stop_price"] = stop_price if strategy.execution_style == "trailing_stop" else None
            state["side"] = side
            state["trade_price_pnl_total"] = 0.0
            state["trade"] = Trade(
                symbol=symbol,
                side=side,
                entry_time=int(ts_value),
                exit_time=None,
                entry_price=float(fill),
                exit_price=None,
                quantity=float(quantity),
                entry_equity=float(entry_equity_before),
                exit_equity=None,
                gross_pnl=None,
                net_pnl=None,
                entry_reason=entry_reason,
                exit_reason=None,
            )
            if collect_events:
                state["event_rows"].append(
                    {
                        "timestamp": int(ts_value),
                        "event": "enter_long" if side == "long" else "enter_short",
                        "price": float(fill),
                        "equity": float(margin_required),
                    }
                )

        for symbol in batch:
            state = states[symbol]
            minute_idx = state["minute_idx"]
            if minute_idx >= len(state["ts_ns"]):
                continue

            processed = symbol in minute_data
            if processed:
                data = minute_data[symbol]
                if data["had_position_before"] or state["side"] != "flat":
                    state["active_minutes"] += 1

                if state["side"] != "flat":
                    mark_price = _mark_price_from_quotes(state["side"], data["b_close"], data["a_close"])
                    _apply_variation_margin(state, mark_price, int(ts_value))
                    state["last_marked_equity"] = _marked_equity_for_risk(state, data)
                    state["last_maintenance_reference_price"] = float(mark_price)
                    state["maintenance_margin_required"] = _maintenance_margin_for_state(state, data, int(ts_value))
                else:
                    state["last_marked_equity"] = 0.0
                    state["last_mark_price"] = None
                    state["last_fx_timestamp"] = None
                    state["last_maintenance_reference_price"] = None
                    state["maintenance_margin_required"] = 0.0

                next_idx = minute_idx + 1
                next_ts_value = int(state["ts_ns"][next_idx]) if next_idx < len(state["ts_ns"]) else np.iinfo(np.int64).max
                if (
                    state["current_window"] < state["n_windows"]
                    and state["side"] != "flat"
                    and next_ts_value > int(state["window_end"][state["current_window"]])
                ):
                    state["last_allocated_equity"] = _marked_equity_for_risk(state, data)
                elif state["side"] == "flat":
                    state["last_allocated_equity"] = None

                while (
                    state["current_window"] < state["n_windows"]
                    and next_ts_value > int(state["window_end"][state["current_window"]])
                ):
                    allocated_equity = state["last_allocated_equity"] if state["side"] != "flat" else None
                    _finalize_bar(state, state["current_window"], allocated_equity)
                    state["current_window"] += 1
                    _sync_stop(state)
            state["minute_idx"] += 1
            if state["current_window"] < state["n_windows"] and state["minute_idx"] < len(state["ts_ns"]):
                heapq.heappush(
                    heap,
                    (int(state["ts_ns"][state["minute_idx"]]), symbol_order[symbol], symbol),
                )

        portfolio_maintenance_margin_required = 0.0
        open_states: list[dict] = []
        for state in states.values():
            if state["side"] == "flat":
                continue
            open_states.append(state)
            portfolio_maintenance_margin_required += float(state["maintenance_margin_required"])

        if (
            open_states
            and float(account_cash) < portfolio_maintenance_margin_required
            and not any(state["pending_margin_liquidation"] for state in open_states)
        ):
            maintenance_margin_call_count += 1
            for state in open_states:
                state["pending_margin_liquidation"] = True
                if collect_events:
                    state["event_rows"].append(
                        {
                            "timestamp": int(ts_value),
                            "event": "maintenance_margin_call",
                            "price": np.nan,
                            "equity": float(state["last_marked_equity"]),
                        }
                    )

    if batch_progress is not None:
        batch_progress.close()

    symbol_results: dict[str, dict] = {}
    symbol_stats_rows: list[dict] = []

    for symbol, state in states.items():
        if state["trade"] is not None and len(state["ts_ns"]) > 0:
            last_ts = int(state["ts_ns"][-1])
            recorded_exit_time = int(state["bar_rows"][-1]["bar_end"]) if state["bar_rows"] else last_ts
            if state["side"] == "long":
                fill = _apply_adverse_slippage(float(state["bid_close"][-1]), "sell", config)
            else:
                fill = _apply_adverse_slippage(float(state["ask_close"][-1]), "buy", config)
            _close_open_position(
                state=state,
                ts_value=recorded_exit_time,
                fill=float(fill),
                exit_reason="end_of_data",
                event_name=None,
            )
            if state["bar_rows"]:
                state["bar_rows"][-1]["equity"] = 1.0 + float(state["realized_pnl_total"]) / float(config.initial_equity)
                state["bar_rows"][-1]["allocated_equity"] = np.nan
                state["bar_rows"][-1]["margin_in_use"] = 0.0
                state["bar_rows"][-1]["maintenance_margin_required"] = 0.0
                state["bar_rows"][-1]["position"] = "flat"
                state["bar_rows"][-1]["position_sign"] = 0
                state["bar_rows"][-1]["stop_price"] = np.nan

    if portfolio_rows_by_end:
        _capture_portfolio_bar(max(portfolio_rows_by_end))

    for symbol, state in states.items():
        bar_df = pd.DataFrame(state["bar_rows"])
        if not bar_df.empty:
            bar_df["bar_end"] = pd.to_datetime(bar_df["bar_end"], unit="ns", utc=True)
            bar_df = bar_df.sort_values("bar_end").reset_index(drop=True)
            bar_df["bar_return"] = bar_df["equity"].pct_change().fillna(0.0)
        trade_df = pd.DataFrame(state["trade_rows"])
        if not trade_df.empty:
            trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"], unit="ns", utc=True)
            trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"], unit="ns", utc=True)
            trade_df = trade_df.sort_values("entry_time").reset_index(drop=True)
        event_df = pd.DataFrame(state["event_rows"])
        if not event_df.empty:
            event_df["timestamp"] = pd.to_datetime(event_df["timestamp"], unit="ns", utc=True)
            event_df = event_df.sort_values("timestamp").reset_index(drop=True)

        symbol_bars_per_year = _bars_per_year(_timeframe_for_symbol(config, symbol))
        stats = bar_performance_stats(
            bar_df["equity"] if not bar_df.empty else pd.Series(dtype="float64"),
            bars_per_year=symbol_bars_per_year,
        )
        stats["n_trades"] = float(len(trade_df))
        stats.update(trade_performance_stats(trade_df))
        stats["active_minutes"] = float(state["active_minutes"])
        stats["total_minutes"] = float(state["total_minutes"])
        stats["minutes_in_position_ratio"] = (
            float(state["active_minutes"]) / float(state["total_minutes"]) if state["total_minutes"] > 0 else np.nan
        )
        stats["net_pnl_total"] = float(trade_df["net_pnl"].sum()) if not trade_df.empty and "net_pnl" in trade_df else 0.0
        stats["overnight_pnl_total"] = (
            float(trade_df["overnight_pnl"].sum()) if not trade_df.empty and "overnight_pnl" in trade_df else 0.0
        )

        symbol_results[symbol] = {
            "features": state["features"],
            "schedule": state["schedule"],
            "bars": bar_df,
            "trades": trade_df,
            "events": event_df,
            "stats": stats,
        }
        symbol_stats_rows.append({"symbol": symbol, **stats})

    symbol_stats = pd.DataFrame(symbol_stats_rows).sort_values("symbol").reset_index(drop=True)
    portfolio_bars = (
        pd.DataFrame(sorted(portfolio_rows_by_end.values(), key=lambda row: int(row["bar_end"])))
        if portfolio_rows_by_end
        else pd.DataFrame(
            columns=[
                "bar_end",
                "equity",
                "bar_return",
                "n_active",
                "capital_invested",
                "cash_available",
                "margin_in_use",
                "maintenance_margin_required",
                "overnight_pnl_total",
            ]
        )
    )
    if not portfolio_bars.empty:
        portfolio_bars["bar_end"] = pd.to_datetime(portfolio_bars["bar_end"], unit="ns", utc=True)
        portfolio_bars = portfolio_bars.sort_values("bar_end").reset_index(drop=True)
        portfolio_bars["bar_return"] = portfolio_bars["equity"].pct_change().fillna(0.0)
    portfolio_stats = resampled_bar_performance_stats(
        portfolio_bars["equity"] if not portfolio_bars.empty else pd.Series(dtype="float64"),
        portfolio_bars["bar_end"] if not portfolio_bars.empty else pd.DatetimeIndex([]),
        resample_freq="1D",
        bars_per_year=252,
    )
    portfolio_stats.update(trade_performance_stats(_all_trades_df(symbol_results)))
    portfolio_stats["symbols_profitable_ratio"] = _profitable_symbol_ratio(symbol_stats)
    portfolio_stats["n_symbols"] = float(len(symbol_results))
    portfolio_stats["n_trades_total"] = float(sum(len(result["trades"]) for result in symbol_results.values()))
    if not portfolio_bars.empty and "capital_invested" in portfolio_bars:
        utilization = portfolio_bars["capital_invested"].astype("float64")
        portfolio_stats["avg_capital_invested"] = float(utilization.mean())
        portfolio_stats["capital_utilization_ratio"] = float(utilization.mean())
        portfolio_stats["capital_utilization_median"] = float(utilization.median())
        portfolio_stats["capital_utilization_peak"] = float(utilization.max())
    portfolio_stats["initial_equity"] = float(config.initial_equity)
    portfolio_stats["initial_margin_ratio"] = float(config.initial_margin_ratio)
    portfolio_stats["maintenance_margin_ratio"] = float(config.maintenance_margin_ratio)
    portfolio_stats["stats_resample_freq"] = "1D"
    portfolio_stats["position_sizing_mode"] = position_sizing_mode
    portfolio_stats["n_trades_skipped_no_cash"] = float(
        sum(row.get("skip_reason") == "no_available_cash" for row in skipped_rows)
    )
    portfolio_stats["n_trades_skipped_invalid_margin"] = float(
        sum(row.get("skip_reason") == "invalid_margin_requirement" for row in skipped_rows)
    )
    portfolio_stats["n_maintenance_margin_calls"] = float(maintenance_margin_call_count)
    all_trades = _all_trades_df(symbol_results)
    portfolio_stats["overnight_pnl_total"] = (
        float(all_trades["overnight_pnl"].sum()) if not all_trades.empty and "overnight_pnl" in all_trades else 0.0
    )
    portfolio_stats["overnight_long_rate"] = float(config.overnight_long_rate)
    portfolio_stats["overnight_short_rate"] = float(config.overnight_short_rate)
    portfolio_stats["overnight_day_count"] = float(overnight_day_count)
    portfolio_stats["n_trades_maintenance_liquidated"] = float(
        (all_trades["exit_reason"] == "maintenance_margin_liquidation").sum()
    ) if not all_trades.empty and "exit_reason" in all_trades else 0.0
    if position_sizing_mode == "fixed_cash":
        portfolio_stats["cash_per_trade"] = float(cash_budget)
    elif position_sizing_mode == "fixed_risk":
        portfolio_stats["risk_per_trade"] = float(fixed_risk_cash)
        portfolio_stats["n_trades_skipped_invalid_stop"] = float(
            sum(row.get("skip_reason") == "invalid_stop_distance" for row in skipped_rows)
        )
    elif position_sizing_mode == "fixed_risk_pct":
        portfolio_stats["risk_per_trade_pct"] = float(fixed_risk_pct)
        portfolio_stats["n_trades_skipped_invalid_stop"] = float(
            sum(row.get("skip_reason") == "invalid_stop_distance" for row in skipped_rows)
        )
    return {
        "portfolio_bars": portfolio_bars,
        "portfolio_stats": portfolio_stats,
        "symbol_stats": symbol_stats,
        "symbol_results": symbol_results,
    }


def run_portfolio_backtest(
    market_by_symbol: dict[str, object],
    params: Any,
    config: EngineConfig,
    strategy: StrategyDefinition | str = "ma_atr_breakout",
    show_progress: bool = False,
    progress_desc: str | None = None,
    collect_events: bool = True,
    portfolio_mode: str = "fixed_risk",
    cash_per_trade: float = 1.0,
    risk_per_trade: float = 0.01,
    risk_per_trade_pct: float = 0.01,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    strategy_def = get_strategy(strategy) if isinstance(strategy, str) else strategy
    if portfolio_mode == "fixed_cash":
        return run_integrated_cash_backtest(
            market_by_symbol=market_by_symbol,
            params=params,
            strategy=strategy_def,
            config=config,
            show_progress=show_progress,
            progress_desc=progress_desc,
            collect_events=collect_events,
            cash_per_trade=cash_per_trade,
            position_sizing_mode="fixed_cash",
        )
    if portfolio_mode == "fixed_risk":
        return run_integrated_cash_backtest(
            market_by_symbol=market_by_symbol,
            params=params,
            strategy=strategy_def,
            config=config,
            show_progress=show_progress,
            progress_desc=progress_desc,
            collect_events=collect_events,
            fixed_risk_cash=risk_per_trade,
            position_sizing_mode="fixed_risk",
        )
    if portfolio_mode == "fixed_risk_pct":
        return run_integrated_cash_backtest(
            market_by_symbol=market_by_symbol,
            params=params,
            strategy=strategy_def,
            config=config,
            show_progress=show_progress,
            progress_desc=progress_desc,
            collect_events=collect_events,
            fixed_risk_pct=risk_per_trade_pct,
            position_sizing_mode="fixed_risk_pct",
        )
    raise ValueError(f"Unsupported portfolio mode: {portfolio_mode}")


def _init_grid_worker(
    market_by_symbol: dict[str, object],
    config: EngineConfig,
    strategy_name: str,
    portfolio_mode: str,
    cash_per_trade: float,
    risk_per_trade: float,
    risk_per_trade_pct: float,
) -> None:
    global _GRID_WORKER_MARKET_BY_SYMBOL, _GRID_WORKER_CONFIG, _GRID_WORKER_STRATEGY_NAME
    global _GRID_WORKER_PORTFOLIO_MODE
    global _GRID_WORKER_CASH_PER_TRADE, _GRID_WORKER_RISK_PER_TRADE, _GRID_WORKER_RISK_PER_TRADE_PCT
    _GRID_WORKER_MARKET_BY_SYMBOL = market_by_symbol
    _GRID_WORKER_CONFIG = config
    _GRID_WORKER_STRATEGY_NAME = strategy_name
    _GRID_WORKER_PORTFOLIO_MODE = portfolio_mode
    _GRID_WORKER_CASH_PER_TRADE = float(cash_per_trade)
    _GRID_WORKER_RISK_PER_TRADE = float(risk_per_trade)
    _GRID_WORKER_RISK_PER_TRADE_PCT = float(risk_per_trade_pct)


def _create_shared_block(array: np.ndarray) -> tuple[SharedMemory, dict]:
    contiguous = np.ascontiguousarray(array)
    shm = SharedMemory(create=True, size=contiguous.nbytes)
    view = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
    view[:] = contiguous
    return shm, {"name": shm.name, "shape": contiguous.shape, "dtype": str(contiguous.dtype)}


def _build_shared_market_payload(market_by_symbol: dict[str, object]) -> tuple[dict[str, dict], dict[str, pd.DataFrame], list[SharedMemory]]:
    payload: dict[str, dict] = {}
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    shms: list[SharedMemory] = []

    for symbol, market in market_by_symbol.items():
        idx_shm, idx_meta = _create_shared_block(market.m1.index.asi8.astype("int64", copy=False))
        data_cols = ["bid_open", "bid_low", "bid_close", "ask_open", "ask_high", "ask_close"]
        data_matrix = np.column_stack(
            [market.m1[col].to_numpy(dtype="float64", copy=False) for col in data_cols]
        )
        data_shm, data_meta = _create_shared_block(data_matrix)
        shms.extend([idx_shm, data_shm])
        payload[symbol] = {
            "index": idx_meta,
            "data": data_meta,
            "cols": data_cols,
        }
        bars_by_symbol[symbol] = market.bars.copy()
    return payload, bars_by_symbol, shms


def _init_grid_worker_shared(
    market_shared: dict[str, dict],
    bars_by_symbol: dict[str, pd.DataFrame],
    config: EngineConfig,
    strategy_name: str,
    portfolio_mode: str,
    cash_per_trade: float,
    risk_per_trade: float,
    risk_per_trade_pct: float,
) -> None:
    global _GRID_WORKER_MARKET_BY_SYMBOL, _GRID_WORKER_MARKET_SHARED, _GRID_WORKER_CONFIG, _GRID_WORKER_SHARED_HANDLES
    global _GRID_WORKER_STRATEGY_NAME, _GRID_WORKER_PORTFOLIO_MODE
    global _GRID_WORKER_CASH_PER_TRADE
    global _GRID_WORKER_RISK_PER_TRADE, _GRID_WORKER_RISK_PER_TRADE_PCT
    _GRID_WORKER_MARKET_BY_SYMBOL = bars_by_symbol
    _GRID_WORKER_MARKET_SHARED = {}
    _GRID_WORKER_CONFIG = config
    _GRID_WORKER_SHARED_HANDLES = []
    _GRID_WORKER_STRATEGY_NAME = strategy_name
    _GRID_WORKER_PORTFOLIO_MODE = portfolio_mode
    _GRID_WORKER_CASH_PER_TRADE = float(cash_per_trade)
    _GRID_WORKER_RISK_PER_TRADE = float(risk_per_trade)
    _GRID_WORKER_RISK_PER_TRADE_PCT = float(risk_per_trade_pct)

    for symbol, meta in market_shared.items():
        idx_shm = SharedMemory(name=meta["index"]["name"])
        data_shm = SharedMemory(name=meta["data"]["name"])
        _GRID_WORKER_SHARED_HANDLES.extend([idx_shm, data_shm])
        idx = np.ndarray(meta["index"]["shape"], dtype=np.dtype(meta["index"]["dtype"]), buffer=idx_shm.buf)
        data = np.ndarray(meta["data"]["shape"], dtype=np.dtype(meta["data"]["dtype"]), buffer=data_shm.buf)
        _GRID_WORKER_MARKET_SHARED[symbol] = {
            "ts_ns": idx,
            "data": data,
        }


def _run_grid_task(task: dict[str, Any]) -> dict:
    strategy = get_strategy(_GRID_WORKER_STRATEGY_NAME)
    params = strategy.make_params(**task)
    if _GRID_WORKER_MARKET_BY_SYMBOL is None or _GRID_WORKER_CONFIG is None:
        raise RuntimeError("Grid worker not initialized")
    if _GRID_WORKER_MARKET_SHARED is None:
        result = run_portfolio_backtest(
            market_by_symbol=_GRID_WORKER_MARKET_BY_SYMBOL,
            params=params,
            strategy=strategy,
            config=_GRID_WORKER_CONFIG,
            show_progress=False,
            collect_events=False,
            portfolio_mode=_GRID_WORKER_PORTFOLIO_MODE,
            cash_per_trade=_GRID_WORKER_CASH_PER_TRADE,
            risk_per_trade=_GRID_WORKER_RISK_PER_TRADE,
            risk_per_trade_pct=_GRID_WORKER_RISK_PER_TRADE_PCT,
        )
    else:
        market_by_symbol = {}
        for symbol, bars in _GRID_WORKER_MARKET_BY_SYMBOL.items():
            shared = _GRID_WORKER_MARKET_SHARED[symbol]
            data = shared["data"]
            m1 = pd.DataFrame(
                data,
                columns=["bid_open", "bid_low", "bid_close", "ask_open", "ask_high", "ask_close"],
                index=pd.to_datetime(shared["ts_ns"], unit="ns", utc=True),
            )
            market_by_symbol[symbol] = type("SharedMarketSlice", (), {"m1": m1, "bars": bars})()
        result = run_portfolio_backtest(
            market_by_symbol=market_by_symbol,
            params=params,
            strategy=strategy,
            config=_GRID_WORKER_CONFIG,
            show_progress=False,
            collect_events=False,
            portfolio_mode=_GRID_WORKER_PORTFOLIO_MODE,
            cash_per_trade=_GRID_WORKER_CASH_PER_TRADE,
            risk_per_trade=_GRID_WORKER_RISK_PER_TRADE,
            risk_per_trade_pct=_GRID_WORKER_RISK_PER_TRADE_PCT,
        )
    row = {
        **_params_to_dict(params),
        **result["portfolio_stats"],
    }
    if len(result["symbol_results"]) == 1:
        symbol, symbol_result = next(iter(result["symbol_results"].items()))
        if str(symbol).upper() == "USDJPY":
            row[USDJPY_RANGE_RETURN_COLUMN] = _equity_return_between(symbol_result["bars"])
    return row


def _grid_product(strategy: StrategyDefinition, param_grid: dict[str, list]) -> list[dict[str, Any]]:
    valid_names = {field.name for field in fields(strategy.params_type)}
    extra = sorted(set(param_grid.keys()) - valid_names)
    if extra:
        raise ValueError(f"Grid keys include unknown strategy params: {extra}")
    keys = list(param_grid.keys())
    defaults: dict[str, Any] = {}
    for field in fields(strategy.params_type):
        if field.name in param_grid:
            continue
        if field.default is not MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            defaults[field.name] = field.default_factory()  # type: ignore[misc]
        else:
            raise ValueError(f"Missing required grid key with no default: {field.name}")
    combos: list[dict[str, Any]] = []
    for values in product(*(param_grid[key] for key in keys)):
        combo = dict(defaults)
        combo.update(dict(zip(keys, values, strict=True)))
        combos.append(combo)
    return combos


def _attach_grid_coordinates(results: pd.DataFrame, param_columns: list[str]) -> pd.DataFrame:
    out = results.copy()
    for col in param_columns:
        levels = sorted(out[col].drop_duplicates().tolist())
        mapping = {value: idx for idx, value in enumerate(levels)}
        out[f"{col}_ix"] = out[col].map(mapping).astype("int64")
    return out


def _normalize_series_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(0.0, index=series.index, dtype="float64")
    min_value = float(finite.min())
    max_value = float(finite.max())
    if np.isclose(min_value, max_value):
        return pd.Series(0.0, index=series.index, dtype="float64")
    out = (values - min_value) / (max_value - min_value)
    return out.fillna(0.0).astype("float64")


def run_cash_portfolio(
    symbol_results: dict[str, dict],
    initial_equity: float,
) -> pd.DataFrame:
    if not symbol_results:
        return pd.DataFrame(
            columns=[
                "bar_end",
                "equity",
                "bar_return",
                "n_active",
                "capital_invested",
                "cash_available",
                "margin_in_use",
                "maintenance_margin_required",
            ]
        )

    frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for symbol, result in symbol_results.items():
        bars = result["bars"].copy()
        if not bars.empty:
            alloc_col = "allocated_equity" if "allocated_equity" in bars.columns else "equity"
            margin_col = "margin_in_use" if "margin_in_use" in bars.columns else None
            maintenance_col = (
                "maintenance_margin_required" if "maintenance_margin_required" in bars.columns else None
            )
            cols = ["bar_end", alloc_col, "position_sign"]
            if margin_col is not None:
                cols.append(margin_col)
            if maintenance_col is not None:
                cols.append(maintenance_col)
            bars = bars[cols].rename(
                columns={
                    alloc_col: f"{symbol}__allocated_equity",
                    "position_sign": f"{symbol}__position_sign",
                    **({margin_col: f"{symbol}__margin_in_use"} if margin_col is not None else {}),
                    **(
                        {maintenance_col: f"{symbol}__maintenance_margin_required"}
                        if maintenance_col is not None
                        else {}
                    ),
                }
            )
            frames.append(bars)

        trades = result["trades"].copy()
        if not trades.empty:
            trades["symbol"] = symbol
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
            trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
            trade_frames.append(trades[["symbol", "entry_time", "exit_time", "entry_equity", "exit_equity"]].copy())

    if not frames:
        return pd.DataFrame(
            columns=[
                "bar_end",
                "equity",
                "bar_return",
                "n_active",
                "capital_invested",
                "cash_available",
                "margin_in_use",
                "maintenance_margin_required",
            ]
        )

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="bar_end", how="outer")

    merged["bar_end"] = pd.to_datetime(merged["bar_end"], utc=True)
    merged = merged.sort_values("bar_end").reset_index(drop=True)

    alloc_cols = [c for c in merged.columns if c.endswith("__allocated_equity")]
    pos_cols = [c for c in merged.columns if c.endswith("__position_sign")]
    margin_cols = [c for c in merged.columns if c.endswith("__margin_in_use")]
    maintenance_cols = [c for c in merged.columns if c.endswith("__maintenance_margin_required")]
    for alloc_col, pos_col in zip(alloc_cols, pos_cols):
        merged.loc[merged[pos_col].notna() & (merged[pos_col] == 0.0), alloc_col] = 0.0
    for margin_col, pos_col in zip(margin_cols, pos_cols):
        merged.loc[merged[pos_col].notna() & (merged[pos_col] == 0.0), margin_col] = 0.0
    for maintenance_col, pos_col in zip(maintenance_cols, pos_cols):
        merged.loc[merged[pos_col].notna() & (merged[pos_col] == 0.0), maintenance_col] = 0.0
    merged[alloc_cols] = merged[alloc_cols].ffill().fillna(0.0)
    merged[pos_cols] = merged[pos_cols].ffill().fillna(0.0)
    if margin_cols:
        merged[margin_cols] = merged[margin_cols].ffill().fillna(0.0)
    if maintenance_cols:
        merged[maintenance_cols] = merged[maintenance_cols].ffill().fillna(0.0)

    out = pd.DataFrame({"bar_end": merged["bar_end"]})
    out["open_equity"] = merged[alloc_cols].sum(axis=1)
    out["n_active"] = (merged[pos_cols] != 0).sum(axis=1).astype("int64")
    out["margin_in_use"] = merged[margin_cols].sum(axis=1) if margin_cols else 0.0
    out["maintenance_margin_required"] = merged[maintenance_cols].sum(axis=1) if maintenance_cols else 0.0

    if trade_frames:
        all_trades = pd.concat(trade_frames, ignore_index=True)
        entry_ns = pd.DatetimeIndex(all_trades["entry_time"]).asi8
        entry_order = np.argsort(entry_ns)
        entry_ns_sorted = entry_ns[entry_order]
        entry_equity_sorted = all_trades["entry_equity"].to_numpy(dtype="float64", copy=False)[entry_order]
        entry_equity_cum = np.cumsum(entry_equity_sorted)
        exit_ns = pd.DatetimeIndex(all_trades["exit_time"]).asi8
        exit_order = np.argsort(exit_ns)
        exit_ns_sorted = exit_ns[exit_order]
        exit_equity_sorted = all_trades["exit_equity"].to_numpy(dtype="float64", copy=False)[exit_order]
        exit_equity_cum = np.cumsum(exit_equity_sorted)
        bar_ns = pd.DatetimeIndex(out["bar_end"]).asi8
        n_entered = np.searchsorted(entry_ns_sorted, bar_ns, side="right")
        n_exited = np.searchsorted(exit_ns_sorted, bar_ns, side="right")
        entry_equity_sum = np.where(n_entered > 0, entry_equity_cum[n_entered - 1], 0.0)
        exit_equity_sum = np.where(n_exited > 0, exit_equity_cum[n_exited - 1], 0.0)
        out["cash_available"] = float(initial_equity) - entry_equity_sum + exit_equity_sum
    else:
        out["cash_available"] = float(initial_equity)

    out["equity"] = out["cash_available"] + out["open_equity"]
    out["bar_return"] = out["equity"].pct_change().fillna(0.0)
    out["capital_invested"] = np.where(out["equity"] != 0, out["margin_in_use"] / out["equity"], np.nan)
    return out[
        [
            "bar_end",
            "equity",
            "bar_return",
            "n_active",
            "capital_invested",
            "cash_available",
            "margin_in_use",
            "maintenance_margin_required",
        ]
    ]


def add_neighbor_medians(results: pd.DataFrame, param_columns: list[str], radius: int = 1) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    out = _attach_grid_coordinates(results, param_columns=param_columns)
    ann_neighbors: list[float] = []
    calmar_neighbors: list[float] = []
    coord_cols = [f"{col}_ix" for col in param_columns]

    for row_idx, row in enumerate(tqdm(
        out.itertuples(index=False),
        total=len(out),
        desc="Scoring parameter neighborhoods",
        leave=False,
    )):
        distance = pd.Series(0, index=out.index, dtype="int64")
        for col in coord_cols:
            distance = distance.add(out[col].sub(getattr(row, col)).abs(), fill_value=0).astype("int64")
        neigh = out.loc[distance <= radius]
        if neigh.empty:
            neigh = out.iloc[[row_idx]]
        ann_neighbors.append(float(neigh["annualized_return"].median()))
        calmar_neighbors.append(float(neigh["calmar"].median()))

    out["neighbor_median_annualized_return"] = ann_neighbors
    out["neighbor_median_calmar"] = calmar_neighbors
    return out.drop(columns=[f"{col}_ix" for col in param_columns])


def score_grid_search_results(results: pd.DataFrame, opt_symbol: str | None = None) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    out = results.copy()
    out["norm_neighbor_median_calmar"] = _normalize_series_minmax(out["neighbor_median_calmar"])
    out["norm_neighbor_median_annualized_return"] = _normalize_series_minmax(out["neighbor_median_annualized_return"])
    out["norm_annualized_return"] = _normalize_series_minmax(out["annualized_return"])
    out["weighted_score"] = (
        0.8 * out["norm_neighbor_median_calmar"]
        + 0.2 * out["norm_neighbor_median_annualized_return"]
        + 0.0 * out["norm_annualized_return"]
    )
    out[USDJPY_RANGE_PENALTY_COLUMN] = 0.0
    out[TRADE_COUNT_PENALTY_COLUMN] = 0.0
    trade_counts = pd.to_numeric(out.get("n_trades_total"), errors="coerce")
    valid_trade_counts = trade_counts.dropna()
    if not valid_trade_counts.empty:
        penalized_count = int(np.ceil(len(valid_trade_counts) * TRADE_COUNT_BOTTOM_FRACTION))
        if penalized_count > 0:
            penalized_index = valid_trade_counts.nsmallest(penalized_count).index
            out.loc[penalized_index, TRADE_COUNT_PENALTY_COLUMN] = TRADE_COUNT_BOTTOM_PENALTY
    if str(opt_symbol or "").upper() == "USDJPY" and USDJPY_RANGE_RETURN_COLUMN in out.columns:
        profitable_range = pd.to_numeric(out[USDJPY_RANGE_RETURN_COLUMN], errors="coerce") > 0.0
        out.loc[profitable_range, USDJPY_RANGE_PENALTY_COLUMN] = USDJPY_RANGE_PENALTY_POINTS
    out["weighted_score"] = (
        out["weighted_score"] - out[USDJPY_RANGE_PENALTY_COLUMN] - out[TRADE_COUNT_PENALTY_COLUMN]
    )
    out = out.sort_values(
        ["weighted_score", "neighbor_median_calmar", "neighbor_median_annualized_return", "annualized_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return out


def run_grid_search(
    market_by_symbol: dict[str, object],
    strategy: StrategyDefinition | str,
    param_grid: dict[str, list],
    config: EngineConfig,
    neighbor_radius: int = 1,
    max_workers: int | None = None,
    portfolio_mode: str = "fixed_risk",
    cash_per_trade: float = 1.0,
    risk_per_trade: float = 0.01,
    risk_per_trade_pct: float = 0.01,
) -> pd.DataFrame:
    strategy_def = get_strategy(strategy) if isinstance(strategy, str) else strategy
    rows: list[dict] = []
    combos = _grid_product(strategy_def, param_grid)
    tasks = list(combos)
    progress = tqdm(total=len(tasks), desc="Evaluating parameter grid")
    shared_blocks: list[SharedMemory] = []
    try:
        if max_workers == 1:
            _init_grid_worker(
                market_by_symbol,
                config,
                strategy_def.name,
                portfolio_mode,
                cash_per_trade,
                risk_per_trade,
                risk_per_trade_pct,
            )
            for task in tasks:
                rows.append(_run_grid_task(task))
                progress.update(1)
        else:
            market_shared, bars_by_symbol, shared_blocks = _build_shared_market_payload(market_by_symbol)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_grid_worker_shared,
                initargs=(
                    market_shared,
                    bars_by_symbol,
                    config,
                    strategy_def.name,
                    portfolio_mode,
                    cash_per_trade,
                    risk_per_trade,
                    risk_per_trade_pct,
                ),
            ) as executor:
                futures = [executor.submit(_run_grid_task, task) for task in tasks]
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
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    out = add_neighbor_medians(raw, param_columns=list(param_grid.keys()), radius=neighbor_radius)
    return score_grid_search_results(out)
