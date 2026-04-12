from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    ma_len: int
    atr_len: int
    atr_mult: float
    stop_lookback: int
    ma_kind: str = "ema"


def build_signal_schedule(bars: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
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
