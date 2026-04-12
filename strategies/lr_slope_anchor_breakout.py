from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators import average_true_range, linear_regression_slope
from strategies.base import StrategyDefinition

_LR_SLOPE_CACHE: dict[tuple, pd.Series] = {}


@dataclass(frozen=True)
class LrSlopeAnchorBreakoutParams:
    slope_len: int
    slope_threshold: float
    slope_norm_mode: str = "atr"
    anchor_lookback: int = 5
    session_start_hour_utc: int = 7
    session_end_hour_utc: int = 22


def _cached_lr_slope(bars: pd.DataFrame, length: int) -> pd.Series:
    if bars.empty:
        return pd.Series(dtype="float64", index=bars.index)
    key = (
        id(bars),
        int(length),
        len(bars),
        int(pd.Timestamp(bars.index[0]).value),
        int(pd.Timestamp(bars.index[-1]).value),
        float(bars["close"].iloc[0]),
        float(bars["close"].iloc[-1]),
    )
    cached = _LR_SLOPE_CACHE.get(key)
    if cached is None:
        cached = linear_regression_slope(bars["close"], length=length)
        _LR_SLOPE_CACHE[key] = cached
    return cached


def _normalize_slope(
    bars: pd.DataFrame,
    raw_slope: pd.Series,
    params: LrSlopeAnchorBreakoutParams,
) -> tuple[pd.Series, pd.Series]:
    mode = params.slope_norm_mode.lower()
    if mode == "none":
        return raw_slope, pd.Series(np.nan, index=bars.index, dtype="float64")
    if mode == "close_pct":
        close = bars["close"].replace(0.0, np.nan)
        return raw_slope / close, pd.Series(np.nan, index=bars.index, dtype="float64")
    if mode == "atr":
        atr = average_true_range(bars, length=params.slope_len)
        atr = atr.replace(0.0, np.nan)
        return raw_slope / atr, atr
    raise ValueError(f"Unsupported slope_norm_mode: {params.slope_norm_mode}")


def compute_features(bars: pd.DataFrame, params: LrSlopeAnchorBreakoutParams) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "lr_slope",
                "slope_signal",
                "anchor_high",
                "anchor_low",
                "bull_cross",
                "bear_cross",
                "ready",
            ]
        )

    out = pd.DataFrame(index=bars.index)
    out["lr_slope"] = _cached_lr_slope(bars, length=params.slope_len)
    out["atr"] = np.nan
    out["slope_signal"], atr = _normalize_slope(bars, out["lr_slope"], params)
    if not atr.empty:
        out["atr"] = atr

    threshold = float(params.slope_threshold)
    curr_signal = out["slope_signal"]
    prev_signal = curr_signal.shift(1)
    out["bull_cross"] = (prev_signal <= threshold) & (curr_signal > threshold)
    out["bear_cross"] = (prev_signal >= -threshold) & (curr_signal < -threshold)

    anchor_high_src = bars["close"].rolling(params.anchor_lookback, min_periods=params.anchor_lookback).max()
    anchor_low_src = bars["close"].rolling(params.anchor_lookback, min_periods=params.anchor_lookback).min()
    out["anchor_high"] = anchor_high_src.where(out["bull_cross"]).ffill()
    out["anchor_low"] = anchor_low_src.where(out["bear_cross"]).ffill()
    out["ready"] = curr_signal.notna() & out["anchor_high"].notna() & out["anchor_low"].notna()
    return out


def build_signal_schedule(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    params: LrSlopeAnchorBreakoutParams,
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
            ]
        )

    idx = bars.index
    rows: list[dict] = []
    for i in range(len(idx) - 1):
        bar_end = pd.Timestamp(idx[i])
        next_bar_end = pd.Timestamp(idx[i + 1])
        feat = features.loc[bar_end]
        if not bool(feat.get("ready", False)):
            continue
        long_trigger = float(feat["anchor_high"])
        short_trigger = float(feat["anchor_low"])
        if not np.isfinite(long_trigger) or not np.isfinite(short_trigger):
            continue
        rows.append(
            {
                "signal_bar_end": bar_end,
                "window_start": bar_end,
                "window_end": next_bar_end,
                "long_trigger": long_trigger,
                "short_trigger": short_trigger,
                "long_stop": np.nan,
                "short_stop": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["signal_bar_end"] = pd.to_datetime(out["signal_bar_end"], utc=True)
    out["window_start"] = pd.to_datetime(out["window_start"], utc=True)
    out["window_end"] = pd.to_datetime(out["window_end"], utc=True)
    return out.sort_values("window_start").reset_index(drop=True)


def is_entry_allowed(ts: pd.Timestamp, params: LrSlopeAnchorBreakoutParams) -> bool:
    hour = int(pd.Timestamp(ts).tz_convert("UTC").hour)
    start = int(params.session_start_hour_utc)
    end = int(params.session_end_hour_utc)
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def default_grid() -> dict[str, list]:
    return {
        "slope_len": [6, 12, 24, 36, 48, 72],
        "slope_threshold": [0.0, 0.05, 0.1, 0.2, 0.3],
        "anchor_lookback": [3, 5, 8, 12, 20, 40],
        "session_start_hour_utc": [0],
        "session_end_hour_utc": [0],
    }


STRATEGY = StrategyDefinition(
    name="lr_slope_anchor_breakout",
    params_type=LrSlopeAnchorBreakoutParams,
    execution_style="opposite_breakout",
    compute_features=compute_features,
    build_signal_schedule=build_signal_schedule,
    default_grid=default_grid,
    is_entry_allowed=is_entry_allowed,
)
