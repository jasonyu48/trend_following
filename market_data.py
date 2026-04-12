from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


YEAR_FILE_RE = re.compile(r"(?P<symbol>[A-Z0-9]+)_M1_(?P<side>bid|ask)_(?P<year>\d{4})\.parquet$")


@dataclass(frozen=True)
class MarketDataSlice:
    symbol: str
    m1: pd.DataFrame
    bars: pd.DataFrame


def _normalize_pandas_freq(timeframe: str) -> str:
    return str(timeframe).strip().lower()


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx.sort_values()
    return out


def _list_side_files(symbol_dir: Path, symbol: str, side: str) -> list[Path]:
    paths: list[tuple[int, Path]] = []
    for path in symbol_dir.glob(f"{symbol}_M1_{side}_*.parquet"):
        m = YEAR_FILE_RE.match(path.name)
        if not m:
            continue
        paths.append((int(m.group("year")), path))
    return [path for _, path in sorted(paths)]


def _load_side(symbol_dir: Path, symbol: str, side: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in _list_side_files(symbol_dir, symbol, side):
        part = pd.read_parquet(path)
        part = _ensure_utc_index(part)
        cols = ["open", "high", "low", "close", "volume"]
        missing = set(cols) - set(part.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        part = part[cols].rename(columns={c: f"{side}_{c}" for c in cols})
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"No {side} parquet files found for {symbol} in {symbol_dir}")
    non_empty_parts = [part for part in parts if not part.empty]
    if not non_empty_parts:
        return pd.DataFrame(columns=parts[0].columns)
    out = non_empty_parts[0].sort_index() if len(non_empty_parts) == 1 else pd.concat(non_empty_parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_symbol_m1_bid_ask(
    data_dir: str | Path,
    symbol: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    symbol = symbol.upper()
    symbol_dir = Path(data_dir) / symbol
    bid = _load_side(symbol_dir, symbol, "bid")
    ask = _load_side(symbol_dir, symbol, "ask")
    df = bid.join(ask, how="inner").sort_index()
    if df.empty:
        raise ValueError(f"No overlapping bid/ask rows for {symbol}")

    if start is not None:
        ts = pd.Timestamp(start)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        df = df.loc[df.index >= ts]
    if end is not None:
        ts = pd.Timestamp(end)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        df = df.loc[df.index < ts]

    for field in ("open", "high", "low", "close"):
        df[f"mid_{field}"] = (df[f"bid_{field}"] + df[f"ask_{field}"]) / 2.0
    df["spread_open"] = df["ask_open"] - df["bid_open"]
    df["spread_close"] = df["ask_close"] - df["bid_close"]
    return df.sort_index()


def resample_mid_bars(m1: pd.DataFrame, timeframe: str = "4H") -> pd.DataFrame:
    src = _ensure_utc_index(m1)
    if src.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    freq = _normalize_pandas_freq(timeframe)

    bars = pd.DataFrame(
        {
            "open": src["mid_open"].resample(freq, label="right", closed="right").first(),
            "high": src["mid_high"].resample(freq, label="right", closed="right").max(),
            "low": src["mid_low"].resample(freq, label="right", closed="right").min(),
            "close": src["mid_close"].resample(freq, label="right", closed="right").last(),
            "volume": (
                (src["bid_volume"].fillna(0.0) + src["ask_volume"].fillna(0.0)) / 2.0
            ).resample(freq, label="right", closed="right").sum(),
        }
    )
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    return bars.sort_index()


def load_symbol_market_data(
    data_dir: str | Path,
    symbol: str,
    timeframe: str = "4H",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> MarketDataSlice:
    m1 = load_symbol_m1_bid_ask(data_dir=data_dir, symbol=symbol, start=start, end=end)
    bars = resample_mid_bars(m1, timeframe=timeframe)
    return MarketDataSlice(symbol=symbol.upper(), m1=m1, bars=bars)
