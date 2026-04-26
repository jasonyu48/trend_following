from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from indicators import average_true_range
from strategies.base import StrategyDefinition


@dataclass(frozen=True)
class SupertrendParams:
    atr_len: int
    atr_mult: float


def compute_features(bars: pd.DataFrame, params: SupertrendParams) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["mid", "atr", "upper", "lower", "ready"])

    out = pd.DataFrame(index=bars.index)
    out["mid"] = (bars["high"] + bars["low"]) / 2.0
    out["atr"] = average_true_range(bars, length=params.atr_len)
    out["upper"] = out["mid"] + float(params.atr_mult) * out["atr"]
    out["lower"] = out["mid"] - float(params.atr_mult) * out["atr"]
    out["ready"] = out[["mid", "atr", "upper", "lower"]].notna().all(axis=1)
    return out


def build_signal_schedule(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    params: SupertrendParams,
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
                "long_exit_band",
                "short_exit_band",
            ]
        )

    idx = bars.index
    rows: list[dict[str, float | pd.Timestamp]] = []
    for i in range(len(idx) - 1):
        bar_end = pd.Timestamp(idx[i])
        next_bar_end = pd.Timestamp(idx[i + 1])
        feat = features.loc[bar_end]
        if not bool(feat.get("ready", False)):
            continue
        lower = float(feat["lower"])
        upper = float(feat["upper"])
        rows.append(
            {
                "signal_bar_end": bar_end,
                "window_start": bar_end,
                "window_end": next_bar_end,
                "long_trigger": upper,
                "short_trigger": lower,
                "long_stop": lower,
                "short_stop": upper,
                "long_exit_band": lower,
                "short_exit_band": upper,
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
        "atr_len": [3, 5, 7, 10, 14, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "atr_mult": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    }


STRATEGY = StrategyDefinition(
    name="supertrend",
    params_type=SupertrendParams,
    execution_style="trailing_stop",
    compute_features=compute_features,
    build_signal_schedule=build_signal_schedule,
    default_grid=default_grid,
)
