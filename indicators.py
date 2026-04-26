from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    x = bars["high"] - bars["low"]
    y = (bars["high"] - prev_close).abs()
    z = (bars["low"] - prev_close).abs()
    return pd.concat([x, y, z], axis=1).max(axis=1)


def moving_average(series: pd.Series, length: int, kind: str = "ema") -> pd.Series:
    kind = kind.lower()
    if kind == "ema":
        return series.ewm(span=length, adjust=False, min_periods=length).mean()
    if kind == "sma":
        return series.rolling(length, min_periods=length).mean()
    raise ValueError(f"Unsupported ma kind: {kind}")


def trend_ma_centerline(bars: pd.DataFrame, length: int, kind: str = "ema") -> pd.Series:
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return (pd.to_numeric(bars["high"], errors="coerce") + pd.to_numeric(bars["low"], errors="coerce")) / 2.0
    return moving_average(bars["close"], length=length, kind=kind)


def average_true_range(bars: pd.DataFrame, length: int) -> pd.Series:
    tr = true_range(bars)
    return tr.ewm(alpha=1.0 / float(length), adjust=False, min_periods=length).mean()


def relative_strength_index(series: pd.Series, length: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    delta = values.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / float(length), adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1.0 / float(length), adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down
    return 100.0 - (100.0 / (1.0 + rs))


def linear_regression_slope(series: pd.Series, length: int) -> pd.Series:
    values = series.to_numpy(dtype="float64", copy=False)
    out = np.full(len(values), np.nan, dtype="float64")
    if length <= 0:
        raise ValueError("length must be positive")
    if len(values) < length:
        return pd.Series(out, index=series.index, dtype="float64")

    x = np.arange(length, dtype="float64")
    sum_x = float(x.sum())
    sum_x2 = float(np.square(x).sum())
    denom = float(length * sum_x2 - sum_x * sum_x)
    if denom == 0.0:
        return pd.Series(out, index=series.index, dtype="float64")

    valid = np.isfinite(values).astype("float64")
    safe_values = np.where(np.isfinite(values), values, 0.0)
    sum_y = np.convolve(safe_values, np.ones(length, dtype="float64"), mode="valid")
    sum_xy = np.convolve(safe_values, x[::-1], mode="valid")
    valid_count = np.convolve(valid, np.ones(length, dtype="float64"), mode="valid")

    numer = float(length) * sum_xy - sum_x * sum_y
    slope_valid = numer / denom
    slope_valid = np.where(valid_count == float(length), slope_valid, np.nan)
    out[length - 1 :] = slope_valid
    return pd.Series(out, index=series.index, dtype="float64")


def compute_trend_features(
    bars: pd.DataFrame,
    ma_len: int,
    atr_len: int,
    atr_mult: float,
    stop_lookback: int,
    ma_kind: str = "ema",
) -> pd.DataFrame:
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
    out["ma"] = trend_ma_centerline(bars, length=ma_len, kind=ma_kind)
    out["atr"] = average_true_range(bars, length=atr_len)
    out["upper"] = out["ma"] + float(atr_mult) * out["atr"]
    out["lower"] = out["ma"] - float(atr_mult) * out["atr"]
    out["trail_stop_long"] = bars["low"].rolling(stop_lookback, min_periods=stop_lookback).min()
    out["trail_stop_short"] = bars["high"].rolling(stop_lookback, min_periods=stop_lookback).max()
    out["ready"] = (
        out[["ma", "atr", "upper", "lower", "trail_stop_long", "trail_stop_short"]]
        .notna()
        .all(axis=1)
    )
    return out


def bar_performance_stats(
    equity: pd.Series,
    bars_per_year: int,
    timestamps: pd.Series | pd.Index | None = None,
) -> dict[str, float]:
    return _bar_performance_stats_with_timestamps(equity, timestamps=timestamps, bars_per_year=bars_per_year)


def _max_recovery_time_days(equity: pd.Series, timestamps: pd.Series | pd.Index | None = None) -> float:
    eq = pd.to_numeric(equity, errors="coerce").astype("float64")
    if timestamps is None:
        return np.nan
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if len(eq) != len(ts):
        raise ValueError("equity and timestamps must have the same length")
    series = pd.Series(eq.to_numpy(copy=False), index=ts, dtype="float64").dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        return np.nan

    peak_value = float("-inf")
    peak_time: pd.Timestamp | None = None
    drawdown_start_time: pd.Timestamp | None = None
    in_drawdown = False
    max_recovery = pd.Timedelta(0)

    for ts_value, value in series.items():
        if value >= peak_value:
            if in_drawdown and drawdown_start_time is not None:
                max_recovery = max(max_recovery, ts_value - drawdown_start_time)
                in_drawdown = False
                drawdown_start_time = None
            peak_value = float(value)
            peak_time = ts_value
            continue
        if not in_drawdown and peak_time is not None:
            in_drawdown = True
            drawdown_start_time = peak_time

    if in_drawdown and drawdown_start_time is not None:
        max_recovery = max(max_recovery, series.index[-1] - drawdown_start_time)
    return float(max_recovery / pd.Timedelta(days=1))


def _bar_performance_stats_with_timestamps(
    equity: pd.Series,
    timestamps: pd.Series | pd.Index | None,
    bars_per_year: int,
) -> dict[str, float]:
    raw_eq = pd.to_numeric(equity, errors="coerce").astype("float64")
    if timestamps is not None:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        if len(raw_eq) != len(ts):
            raise ValueError("equity and timestamps must have the same length")
        valid = raw_eq.notna().to_numpy()
        eq = raw_eq.loc[raw_eq.notna()].astype("float64")
        stat_timestamps: pd.DatetimeIndex | None = ts[valid]
    else:
        eq = raw_eq.dropna().astype("float64")
        stat_timestamps = None
    if eq.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "max_recovery_time": np.nan,
        }
    ret = eq.pct_change().fillna(0.0)
    years = max(len(ret) / float(bars_per_year), 1.0 / float(bars_per_year))
    start_equity = float(eq.iloc[0])
    end_equity = float(eq.iloc[-1])
    total_return = float(end_equity / start_equity - 1.0) if start_equity != 0 else np.nan
    if start_equity > 0 and end_equity > 0 and np.isfinite(start_equity) and np.isfinite(end_equity):
        ann_return = float((end_equity / start_equity) ** (1.0 / years) - 1.0)
    elif start_equity > 0 and end_equity == 0:
        ann_return = -1.0
    else:
        ann_return = np.nan
    ann_vol = float(ret.std(ddof=1) * np.sqrt(bars_per_year)) if len(ret) > 1 else np.nan
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(bars_per_year)) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min()) if not dd.empty else np.nan
    calmar = float(ann_return / abs(max_dd)) if np.isfinite(ann_return) and np.isfinite(max_dd) and max_dd < 0 else np.nan
    max_recovery_time = _max_recovery_time_days(eq, timestamps=stat_timestamps)
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "max_recovery_time": max_recovery_time,
    }


def resampled_bar_performance_stats(
    equity: pd.Series,
    timestamps: pd.Series | pd.Index,
    resample_freq: str = "1D",
    bars_per_year: int = 252,
) -> dict[str, float]:
    eq = pd.to_numeric(equity, errors="coerce").astype("float64")
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if len(eq) != len(ts):
        raise ValueError("equity and timestamps must have the same length")
    if len(eq) == 0:
        return bar_performance_stats(pd.Series(dtype="float64"), bars_per_year=bars_per_year)
    series = pd.Series(eq.to_numpy(copy=False), index=ts, dtype="float64")
    series = series.dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        return bar_performance_stats(pd.Series(dtype="float64"), bars_per_year=bars_per_year)
    resampled = series.resample(resample_freq).last().ffill().dropna()
    return _bar_performance_stats_with_timestamps(
        resampled,
        timestamps=resampled.index,
        bars_per_year=bars_per_year,
    )


def trade_performance_stats(trades: pd.DataFrame, pnl_col: str = "net_pnl") -> dict[str, float]:
    if trades.empty or pnl_col not in trades.columns:
        return {
            "win_rate": np.nan,
            "payoff_ratio": np.nan,
            "profit_factor": np.nan,
        }

    pnl = pd.to_numeric(trades[pnl_col], errors="coerce").dropna().astype("float64")
    if pnl.empty:
        return {
            "win_rate": np.nan,
            "payoff_ratio": np.nan,
            "profit_factor": np.nan,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float((pnl > 0).mean())
    if not wins.empty and not losses.empty:
        payoff_ratio = float(wins.mean() / abs(losses.mean()))
        profit_factor = float(wins.sum() / abs(losses.sum()))
    else:
        payoff_ratio = np.nan
        profit_factor = np.nan
    return {
        "win_rate": win_rate,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
    }
