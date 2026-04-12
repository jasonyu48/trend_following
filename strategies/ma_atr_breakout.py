from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from indicators import average_true_range, moving_average
from strategies.base import StrategyDefinition


@dataclass(frozen=True)
class MaAtrBreakoutParams:
    ma_len: int
    atr_len: int
    atr_mult: float
    stop_lookback: int
    ma_kind: str = "ema"


def compute_features(bars: pd.DataFrame, params: MaAtrBreakoutParams) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "ma",
                "atr",
                "upper",
                "lower",
                "trail_stop_long",
                "trail_stop_short",
                "ready",
            ]
        )

    out = pd.DataFrame(index=bars.index)
    out["ma"] = moving_average(bars["close"], length=params.ma_len, kind=params.ma_kind)
    out["atr"] = average_true_range(bars, length=params.atr_len)
    out["upper"] = out["ma"] + float(params.atr_mult) * out["atr"]
    out["lower"] = out["ma"] - float(params.atr_mult) * out["atr"]
    out["trail_stop_long"] = bars["low"].rolling(params.stop_lookback, min_periods=params.stop_lookback).min()
    out["trail_stop_short"] = bars["high"].rolling(params.stop_lookback, min_periods=params.stop_lookback).max()
    out["ready"] = (
        out[["ma", "atr", "upper", "lower", "trail_stop_long", "trail_stop_short"]]
        .notna()
        .all(axis=1)
    )
    return out


def build_signal_schedule(bars: pd.DataFrame, features: pd.DataFrame, params: MaAtrBreakoutParams) -> pd.DataFrame:
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
        rows.append(
            {
                "signal_bar_end": bar_end,
                "window_start": bar_end,
                "window_end": next_bar_end,
                "long_trigger": float(feat["upper"]),
                "short_trigger": float(feat["lower"]),
                "long_stop": float(feat["trail_stop_long"]),
                "short_stop": float(feat["trail_stop_short"]),
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
        "ma_len": [1, 3, 5, 7, 10, 14],
        "atr_len": [10, 20, 30, 40, 50, 60],
        "atr_mult": [1.5, 2.0, 2.5, 3.0, 3.5],
        "stop_lookback": [5, 7, 10, 14, 20, 30, 40, 50, 60],
    }


STRATEGY = StrategyDefinition(
    name="ma_atr_breakout",
    params_type=MaAtrBreakoutParams,
    execution_style="trailing_stop",
    compute_features=compute_features,
    build_signal_schedule=build_signal_schedule,
    default_grid=default_grid,
)
