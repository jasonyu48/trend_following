from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

PositionSide = Literal["flat", "long", "short"]


@dataclass(frozen=True)
class EngineConfig:
    timeframe: str = "4H"
    timeframe_by_symbol: dict[str, str] | None = None
    commission_bps: float = 0.0
    commission_bps_by_symbol: dict[str, float] | None = None
    slippage_bps: float = 0.0
    overnight_long_rate: float = 0.0
    overnight_short_rate: float = 0.0
    overnight_long_rate_by_symbol: dict[str, float] | None = None
    overnight_short_rate_by_symbol: dict[str, float] | None = None
    overnight_day_count: int = 360
    initial_equity: float = 1.0
    initial_margin_ratio: float = 0.03
    maintenance_margin_ratio: float = 0.025
    initial_margin_ratio_by_symbol: dict[str, float] | None = None
    maintenance_margin_ratio_by_symbol: dict[str, float] | None = None
    account_currency: str = "USD"
    symbol_specs: dict[str, dict[str, object]] | None = None
    fx_daily: Any = None
    allow_flip_same_minute: bool = False
    opposite_signal_action: str = "close_only"


@dataclass
class Trade:
    symbol: str
    side: PositionSide
    entry_time: int
    exit_time: int | None
    entry_price: float
    exit_price: float | None
    quantity: float
    entry_equity: float
    exit_equity: float | None
    gross_pnl: float | None
    net_pnl: float | None
    entry_reason: str
    exit_reason: str | None
    bars_held: int = 0
    overnight_pnl: float | None = None


def _bars_per_year(timeframe: str) -> int:
    tf = str(timeframe).strip().lower()
    try:
        offset = pd.tseries.frequencies.to_offset(tf)
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc
    delta = pd.Timedelta(offset)
    if delta <= pd.Timedelta(0):
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    trading_year = pd.Timedelta(days=252)
    return max(int(round(trading_year / delta)), 1)


def _timeframe_for_symbol(config: EngineConfig, symbol: str) -> str:
    if config.timeframe_by_symbol is None:
        return str(config.timeframe)
    return str(config.timeframe_by_symbol.get(str(symbol).upper(), config.timeframe))


def _slippage_rate(config: EngineConfig) -> float:
    return float(config.slippage_bps) / 10000.0


def _commission_rate(config: EngineConfig) -> float:
    return float(config.commission_bps) / 10000.0


def _commission_rate_for_symbol(config: EngineConfig, symbol: str) -> float:
    if config.commission_bps_by_symbol is None:
        return _commission_rate(config)
    value = config.commission_bps_by_symbol.get(str(symbol).upper(), config.commission_bps)
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"commission_bps must be finite and non-negative for symbol {symbol!r}, got {value!r}")
    return value / 10000.0


def _overnight_day_count(config: EngineConfig) -> int:
    day_count = int(config.overnight_day_count)
    if day_count <= 0:
        raise ValueError(f"overnight_day_count must be positive, got {day_count!r}")
    return day_count


def _ratio_for_symbol(
    default_ratio: float,
    ratio_by_symbol: dict[str, float] | None,
    symbol: str,
    label: str,
) -> float:
    if ratio_by_symbol is not None:
        ratio = ratio_by_symbol.get(str(symbol).upper(), default_ratio)
    else:
        ratio = default_ratio
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"{label} must be positive for symbol {symbol!r}, got {ratio!r}")
    return ratio


def _rate_for_symbol(
    default_rate: float,
    rate_by_symbol: dict[str, float] | None,
    symbol: str,
    label: str,
) -> float:
    if rate_by_symbol is not None:
        rate = rate_by_symbol.get(str(symbol).upper(), default_rate)
    else:
        rate = default_rate
    rate = float(rate)
    if not np.isfinite(rate):
        raise ValueError(f"{label} must be finite for symbol {symbol!r}, got {rate!r}")
    return rate


def _initial_margin_ratio_for_symbol(config: EngineConfig, symbol: str) -> float:
    return _ratio_for_symbol(
        default_ratio=config.initial_margin_ratio,
        ratio_by_symbol=config.initial_margin_ratio_by_symbol,
        symbol=symbol,
        label="Initial margin ratio",
    )


def _maintenance_margin_ratio_for_symbol(config: EngineConfig, symbol: str) -> float:
    return _ratio_for_symbol(
        default_ratio=config.maintenance_margin_ratio,
        ratio_by_symbol=config.maintenance_margin_ratio_by_symbol,
        symbol=symbol,
        label="Maintenance margin ratio",
    )


def _overnight_rate_for_side(config: EngineConfig, symbol: str, side: PositionSide) -> float:
    if side == "long":
        return _rate_for_symbol(
            default_rate=config.overnight_long_rate,
            rate_by_symbol=config.overnight_long_rate_by_symbol,
            symbol=symbol,
            label="Overnight long rate",
        )
    if side == "short":
        return _rate_for_symbol(
            default_rate=config.overnight_short_rate,
            rate_by_symbol=config.overnight_short_rate_by_symbol,
            symbol=symbol,
            label="Overnight short rate",
        )
    return 0.0


def _apply_adverse_slippage(price: float, action: Literal["buy", "sell"], config: EngineConfig) -> float:
    slip = _slippage_rate(config)
    if action == "buy":
        return float(price) * (1.0 + slip)
    return float(price) * (1.0 - slip)


def _mark_equity(
    side: PositionSide,
    quantity: float,
    entry_fill: float | None,
    entry_equity_after_cost: float | None,
    bid_close: float,
    ask_close: float,
    flat_equity: float,
    contract_multiplier: float = 1.0,
    pnl_to_account_rate: float = 1.0,
) -> float:
    if side == "flat" or quantity <= 0 or entry_fill is None or entry_equity_after_cost is None:
        return float(flat_equity)
    if side == "long":
        mark = (
            entry_equity_after_cost
            + quantity * float(contract_multiplier) * (float(bid_close) - float(entry_fill)) * float(pnl_to_account_rate)
        )
        return float(mark)
    if side == "short":
        mark = (
            entry_equity_after_cost
            + quantity * float(contract_multiplier) * (float(entry_fill) - float(ask_close)) * float(pnl_to_account_rate)
        )
        return float(mark)
    raise ValueError(f"Unsupported side: {side}")


def _entry_fill_from_trigger(
    ask_open: float,
    bid_open: float,
    trigger: float,
    side: Literal["long", "short"],
    config: EngineConfig,
) -> float:
    if side == "long":
        raw = float(ask_open) if float(ask_open) >= trigger else float(trigger)
        return _apply_adverse_slippage(raw, "buy", config)
    raw = float(bid_open) if float(bid_open) <= trigger else float(trigger)
    return _apply_adverse_slippage(raw, "sell", config)


def _stop_fill(
    bid_open: float,
    ask_open: float,
    stop_price: float,
    side: Literal["long", "short"],
    config: EngineConfig,
) -> float:
    if side == "long":
        raw = float(bid_open) if float(bid_open) <= stop_price else float(stop_price)
        return _apply_adverse_slippage(raw, "sell", config)
    raw = float(ask_open) if float(ask_open) >= stop_price else float(stop_price)
    return _apply_adverse_slippage(raw, "buy", config)
