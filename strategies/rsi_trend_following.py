from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from indicators import relative_strength_index
from strategies.base import StrategyDefinition


@dataclass(frozen=True)
class RsiTrendFollowingParams:
    rsi_len: int
    entry_lookback: int = 10
    stop_lookback: int = 10
    rsi_bias: float = 10.0
    trade_direction: str = "both"


def _direction_flags(direction: str) -> tuple[bool, bool]:
    normalized = str(direction).strip().lower()
    if normalized == "both":
        return True, True
    if normalized == "long_only":
        return True, False
    if normalized == "short_only":
        return False, True
    raise ValueError(f"Unsupported trade_direction: {direction!r}")


def compute_features(bars: pd.DataFrame, params: RsiTrendFollowingParams) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "rsi",
                "long_trigger",
                "short_trigger",
                "long_stop",
                "short_stop",
                "long_rsi_threshold",
                "short_rsi_threshold",
                "allow_long_entry",
                "allow_short_entry",
                "ready",
            ]
        )

    allow_long, allow_short = _direction_flags(params.trade_direction)
    out = pd.DataFrame(index=bars.index)
    out["rsi"] = relative_strength_index(bars["close"], length=params.rsi_len)
    out["long_trigger"] = bars["high"].rolling(params.entry_lookback, min_periods=params.entry_lookback).max()
    out["short_trigger"] = bars["low"].rolling(params.entry_lookback, min_periods=params.entry_lookback).min()
    out["long_stop"] = bars["low"].rolling(params.stop_lookback, min_periods=params.stop_lookback).min()
    out["short_stop"] = bars["high"].rolling(params.stop_lookback, min_periods=params.stop_lookback).max()
    out["long_rsi_threshold"] = 50.0 + float(params.rsi_bias)
    out["short_rsi_threshold"] = 50.0 - float(params.rsi_bias)
    out["allow_long_entry"] = allow_long & (out["rsi"] >= out["long_rsi_threshold"])
    out["allow_short_entry"] = allow_short & (out["rsi"] <= out["short_rsi_threshold"])
    out["ready"] = out[["rsi", "long_trigger", "short_trigger", "long_stop", "short_stop"]].notna().all(axis=1)
    return out


def build_signal_schedule(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    params: RsiTrendFollowingParams,
) -> pd.DataFrame:
    del params
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "signal_bar_end",
                "window_start",
                "window_end",
                "long_trigger",
                "short_trigger",
                "long_stop",
                "short_stop",
                "allow_long_entry",
                "allow_short_entry",
            ]
        )

    idx = bars.index
    rows: list[dict[str, object]] = []
    for i in range(len(idx) - 1):
        bar_end = pd.Timestamp(idx[i])
        next_bar_end = pd.Timestamp(idx[i + 1])
        feat = features.loc[bar_end]
        if not bool(feat.get("ready", False)):
            continue
        rows.append(
            {
                "signal_bar_end": bar_end,
                "window_start": bar_end,
                "window_end": next_bar_end,
                "long_trigger": float(feat["long_trigger"]),
                "short_trigger": float(feat["short_trigger"]),
                "long_stop": float(feat["long_stop"]),
                "short_stop": float(feat["short_stop"]),
                "allow_long_entry": bool(feat["allow_long_entry"]),
                "allow_short_entry": bool(feat["allow_short_entry"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["signal_bar_end"] = pd.to_datetime(out["signal_bar_end"], utc=True)
    out["window_start"] = pd.to_datetime(out["window_start"], utc=True)
    out["window_end"] = pd.to_datetime(out["window_end"], utc=True)
    return out.sort_values("window_start").reset_index(drop=True)


def default_grid() -> dict[str, list]:
    return {
        "rsi_len": [7, 10, 14, 20, 30, 40, 50, 60],
        "entry_lookback": [7, 10, 14, 20, 30, 40, 50, 60],
        "stop_lookback": [5, 7, 10, 14, 20, 30, 40, 50],
        "rsi_bias": [7.5, 10.0, 20.0, 30.0, 40.0],
    }


STRATEGY = StrategyDefinition(
    name="rsi_trend_following",
    params_type=RsiTrendFollowingParams,
    execution_style="trailing_stop",
    compute_features=compute_features,
    build_signal_schedule=build_signal_schedule,
    default_grid=default_grid,
)
