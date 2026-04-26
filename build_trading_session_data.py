from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from data_paths import DEFAULT_DATA_DIR, RAW_DATA_DIR
from symbol_universe import DEFAULT_SYMBOLS
from trading_sessions import filter_frame_to_trading_sessions

FILTER_SYMBOLS = list(DEFAULT_SYMBOLS)
QUOTE_COLS = ["bid_open", "bid_low", "bid_close", "ask_open", "ask_high", "ask_close"]
STALE_MINUTES_BY_SYMBOL: dict[str, int] = {
    "GBPJPY": 60,
    "USDJPY": 60,
    "USDCHF": 60,
    "USDCNH": 60,
    "GBPUSD": 60,
    "CADJPY": 60,
    "XAUUSD": 30,
    "XAGUSD": 30,
    "BTCUSD": 180,
    "ETHUSD": 180,
    "HK50": 30,
    "US30": 30,
    "SP500": 30,
    "GER40": 30,
    "TESLA": 30,
    "NVDA": 30,
    "AAPL": 30,
}
SYMBOL_MIN_START_UTC: dict[str, pd.Timestamp] = {
    "JP225": pd.Timestamp("2015-01-01 00:00:00+00:00"),
    "US30": pd.Timestamp("2012-07-16 13:30:00+00:00"),
}
ETH_BAD_QUOTE_SPREAD_BPS = 1500.0
ETH_BAD_QUOTE_MOVE_PCT = 0.03
ETH_BAD_QUOTE_OTHER_SIDE_PCT = 0.003


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter raw parquet data to per-symbol legal trading sessions and save a cleaned copy."
    )
    p.add_argument("--source-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--symbols", nargs="+", default=FILTER_SYMBOLS)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _list_symbol_side_files(symbol_dir: Path, symbol: str, side: str) -> dict[str, Path]:
    paths = sorted(symbol_dir.glob(f"{symbol}_M1_{side}_*.parquet"))
    out: dict[str, Path] = {}
    for path in paths:
        year = path.stem.rsplit("_", 1)[-1]
        out[year] = path
    return out


def _stale_quote_mask(frame: pd.DataFrame, symbol: str) -> pd.Series:
    threshold = STALE_MINUTES_BY_SYMBOL.get(str(symbol).upper())
    if threshold is None or frame.empty:
        return pd.Series(False, index=frame.index)
    ts = pd.DatetimeIndex(frame.index)
    same_quote = frame[QUOTE_COLS].eq(frame[QUOTE_COLS].shift()).all(axis=1)
    consecutive_minute = ts.to_series().diff().eq(pd.Timedelta(minutes=1)).to_numpy()
    same_run = same_quote.to_numpy() & consecutive_minute
    mask = pd.Series(False, index=frame.index)
    run_start: int | None = None
    for i, flag in enumerate(same_run):
        if flag and run_start is None:
            run_start = i - 1
        elif not flag and run_start is not None:
            run_length = i - run_start
            if run_length >= threshold:
                mask.iloc[run_start:i] = True
            run_start = None
    if run_start is not None:
        run_length = len(frame) - run_start
        if run_length >= threshold:
            mask.iloc[run_start:] = True
    return mask


def _bad_quote_mask(frame: pd.DataFrame, symbol: str) -> pd.Series:
    symbol = str(symbol).upper()
    if frame.empty or symbol != "ETHUSD":
        return pd.Series(False, index=frame.index)
    mid_close = (frame["bid_close"] + frame["ask_close"]) / 2.0
    spread_bps = (frame["ask_close"] - frame["bid_close"]) / mid_close * 1e4
    bid_ret = (frame["bid_close"] / frame["bid_close"].shift(1) - 1.0).abs()
    ask_ret = (frame["ask_close"] / frame["ask_close"].shift(1) - 1.0).abs()
    one_sided_spike = (
        (ask_ret >= ETH_BAD_QUOTE_MOVE_PCT) & (bid_ret <= ETH_BAD_QUOTE_OTHER_SIDE_PCT)
    ) | (
        (bid_ret >= ETH_BAD_QUOTE_MOVE_PCT) & (ask_ret <= ETH_BAD_QUOTE_OTHER_SIDE_PCT)
    )
    invalid_cross = (
        (frame["ask_open"] <= frame["bid_open"])
        | (frame["ask_high"] <= frame["bid_low"])
        | (frame["ask_close"] <= frame["bid_close"])
    )
    extreme_spread = spread_bps >= ETH_BAD_QUOTE_SPREAD_BPS
    return (one_sided_spike | invalid_cross | extreme_spread).fillna(False)


def _filter_symbol_year(
    *,
    source_dir: Path,
    output_dir: Path,
    symbol: str,
    year: str,
    bid_path: Path,
    ask_path: Path,
) -> list[dict[str, Any]]:
    source_symbol_dir = source_dir / symbol
    output_symbol_dir = output_dir / symbol
    output_symbol_dir.mkdir(parents=True, exist_ok=True)
    del source_symbol_dir
    bid = pd.read_parquet(bid_path)
    ask = pd.read_parquet(ask_path)
    bid.index = pd.to_datetime(bid.index, utc=True)
    ask.index = pd.to_datetime(ask.index, utc=True)

    joined = bid.add_prefix("bid_").join(ask.add_prefix("ask_"), how="inner").sort_index()
    rows_before = int(len(joined))

    filtered = filter_frame_to_trading_sessions(joined, symbol)
    rows_after_session = int(len(filtered))

    stale_mask = _stale_quote_mask(filtered, symbol)
    rows_removed_stale = int(stale_mask.sum())
    filtered = filtered.loc[~stale_mask].copy()

    bad_quote_mask = _bad_quote_mask(filtered, symbol)
    rows_removed_bad_quote = int(bad_quote_mask.sum())
    filtered = filtered.loc[~bad_quote_mask].copy()

    min_start = SYMBOL_MIN_START_UTC.get(str(symbol).upper())
    rows_removed_before_start = 0
    if min_start is not None:
        start_mask = filtered.index < min_start
        rows_removed_before_start = int(start_mask.sum())
        filtered = filtered.loc[~start_mask].copy()

    bid_filtered = filtered[[col for col in filtered.columns if col.startswith("bid_")]].rename(columns=lambda c: c[4:])
    ask_filtered = filtered[[col for col in filtered.columns if col.startswith("ask_")]].rename(columns=lambda c: c[4:])

    bid_target_path = output_symbol_dir / bid_path.name
    ask_target_path = output_symbol_dir / ask_path.name
    bid_temp_path = bid_target_path.with_suffix(bid_target_path.suffix + ".tmp")
    ask_temp_path = ask_target_path.with_suffix(ask_target_path.suffix + ".tmp")
    bid_filtered.to_parquet(bid_temp_path)
    ask_filtered.to_parquet(ask_temp_path)
    bid_temp_path.replace(bid_target_path)
    ask_temp_path.replace(ask_target_path)

    rows_after = int(len(filtered))
    rows_removed_session = rows_before - rows_after_session
    base_summary = {
        "symbol": symbol,
        "year": year,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "rows_removed_session": rows_removed_session,
        "rows_removed_stale": rows_removed_stale,
        "rows_removed_bad_quote": rows_removed_bad_quote,
        "rows_removed_before_start": rows_removed_before_start,
    }
    return [
        {
            **base_summary,
            "side": "bid",
            "file": bid_path.name,
        },
        {
            **base_summary,
            "side": "ask",
            "file": ask_path.name,
        },
    ]


def main() -> int:
    args = _parse_args()
    args.symbols = [str(symbol).upper() for symbol in args.symbols]
    if args.output_dir.exists() and not args.overwrite:
        existing = any(args.output_dir.iterdir())
        if existing:
            raise ValueError(
                f"Output directory already exists and is not empty: {args.output_dir}. "
                "Pass --overwrite to rebuild it."
            )
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    for symbol in args.symbols:
        symbol_dir = args.source_dir / symbol
        bid_files = _list_symbol_side_files(symbol_dir, symbol, "bid")
        ask_files = _list_symbol_side_files(symbol_dir, symbol, "ask")
        years = sorted(set(bid_files) & set(ask_files))
        for year in years:
            all_summaries.extend(
                _filter_symbol_year(
                    source_dir=args.source_dir,
                    output_dir=args.output_dir,
                    symbol=symbol,
                    year=year,
                    bid_path=bid_files[year],
                    ask_path=ask_files[year],
                )
            )

    summary = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "symbols": args.symbols,
        "files": all_summaries,
        "rows_before_total": int(sum(item["rows_before"] for item in all_summaries)),
        "rows_after_total": int(sum(item["rows_after"] for item in all_summaries)),
        "rows_removed_total": int(sum(item["rows_removed"] for item in all_summaries)),
    }
    (args.output_dir / "_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
