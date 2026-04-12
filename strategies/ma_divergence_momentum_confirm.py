from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators import moving_average
from strategies.base import StrategyDefinition


@dataclass(frozen=True)
class MaDivergenceMomentumConfirmParams:
    n: int
    divergence_lookback: int
    stop_lookback: int


def compute_features(
    bars: pd.DataFrame,
    params: MaDivergenceMomentumConfirmParams,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "fast_ma",
                "mid_ma",
                "slow_ma",
                "fast_mid_spread",
                "mid_slow_spread",
                "center_price",
                "displacement",
                "structure_dir",
                "up_push",
                "down_push",
                "momentum_diff",
                "trail_stop_long",
                "trail_stop_short",
                "bull_trend",
                "bear_trend",
                "bull_divergence",
                "bear_divergence",
                "bull_momentum",
                "bear_momentum",
                "long_signal",
                "short_signal",
                "ready",
            ]
        )

    n = int(params.n)
    out = pd.DataFrame(index=bars.index)
    out["fast_ma"] = moving_average(bars["close"], length=n, kind="sma")
    out["mid_ma"] = moving_average(bars["close"], length=2 * n, kind="sma")
    out["slow_ma"] = moving_average(bars["close"], length=4 * n, kind="sma")

    out["fast_mid_spread"] = out["fast_ma"] - out["mid_ma"]
    out["mid_slow_spread"] = out["mid_ma"] - out["slow_ma"]

    out["center_price"] = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    high_shift = bars["high"].shift(1)
    low_shift = bars["low"].shift(1)
    out["displacement"] = np.maximum((bars["high"] - high_shift).abs(), (bars["low"] - low_shift).abs())
    out["structure_dir"] = np.sign(out["center_price"].diff())

    out["up_push"] = np.where(out["structure_dir"] > 0, out["displacement"], 0.0)
    out["down_push"] = np.where(out["structure_dir"] < 0, out["displacement"], 0.0)
    out["momentum_diff"] = (
        pd.Series(out["up_push"], index=out.index).rolling(n, min_periods=n).sum()
        - pd.Series(out["down_push"], index=out.index).rolling(n, min_periods=n).sum()
    )

    out["trail_stop_long"] = bars["low"].rolling(params.stop_lookback, min_periods=params.stop_lookback).min()
    out["trail_stop_short"] = bars["high"].rolling(params.stop_lookback, min_periods=params.stop_lookback).max()

    out["bull_trend"] = (
        (bars["close"] > out["fast_ma"])
        & (out["fast_ma"] > out["mid_ma"])
        & (out["mid_ma"] > out["slow_ma"])
    )
    out["bear_trend"] = (
        (bars["close"] < out["fast_ma"])
        & (out["fast_ma"] < out["mid_ma"])
        & (out["mid_ma"] < out["slow_ma"])
    )

    fast_mid_prev = out["fast_mid_spread"].shift(params.divergence_lookback)
    mid_slow_prev = out["mid_slow_spread"].shift(params.divergence_lookback)
    out["bull_divergence"] = (
        (out["fast_mid_spread"] > fast_mid_prev)
        & (out["mid_slow_spread"] > mid_slow_prev)
    )
    out["bear_divergence"] = (
        (out["fast_mid_spread"] < fast_mid_prev)
        & (out["mid_slow_spread"] < mid_slow_prev)
    )

    momentum_prev = out["momentum_diff"].shift(1)
    out["bull_momentum"] = (out["momentum_diff"] > 0.0) & (out["momentum_diff"] > momentum_prev)
    out["bear_momentum"] = (out["momentum_diff"] < 0.0) & (out["momentum_diff"] < momentum_prev)

    out["long_signal"] = out["bull_trend"] & out["bull_divergence"] & out["bull_momentum"]
    out["short_signal"] = out["bear_trend"] & out["bear_divergence"] & out["bear_momentum"]

    out["ready"] = (
        out[
            [
                "fast_ma",
                "mid_ma",
                "slow_ma",
                "displacement",
                "momentum_diff",
                "trail_stop_long",
                "trail_stop_short",
            ]
        ]
        .notna()
        .all(axis=1)
        & fast_mid_prev.notna()
        & mid_slow_prev.notna()
        & momentum_prev.notna()
    )
    return out


def build_signal_schedule(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    params: MaDivergenceMomentumConfirmParams,
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

        next_open = float(bars["open"].iloc[i + 1])
        if not np.isfinite(next_open):
            continue

        if bool(feat["long_signal"]):
            rows.append(
                {
                    "signal_bar_end": bar_end,
                    "window_start": bar_end,
                    "window_end": next_bar_end,
                    "long_trigger": next_open,
                    "short_trigger": np.nan,
                    "long_stop": float(feat["trail_stop_long"]),
                    "short_stop": np.nan,
                }
            )
        elif bool(feat["short_signal"]):
            rows.append(
                {
                    "signal_bar_end": bar_end,
                    "window_start": bar_end,
                    "window_end": next_bar_end,
                    "long_trigger": np.nan,
                    "short_trigger": next_open,
                    "long_stop": np.nan,
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
        "n": [6, 12, 18, 24, 36, 48],
        "divergence_lookback": [1, 2, 4, 8],
        "stop_lookback": [6, 12, 18, 24, 36],
    }


STRATEGY = StrategyDefinition(
    name="ma_divergence_momentum_confirm",
    params_type=MaDivergenceMomentumConfirmParams,
    execution_style="trailing_stop",
    compute_features=compute_features,
    build_signal_schedule=build_signal_schedule,
    default_grid=default_grid,
)
