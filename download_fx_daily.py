from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKER_CANDIDATES = {
    "EURUSD": ["EURUSD=X"],
    "GBPUSD": ["GBPUSD=X"],
    "USDJPY": ["USDJPY=X"],
    "USDCHF": ["USDCHF=X"],
    "USDCNH": ["USDCNH=X", "USDCNY=X"],
    "USDHKD": ["USDHKD=X"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily FX conversion rates for backtests."
    )
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/fx_daily"),
    )
    parser.add_argument(
        "--filename",
        default="fx_daily_2012_2025",
        help="Base filename without extension.",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        default=True,
        help="Forward-fill missing business days with the latest available value.",
    )
    return parser.parse_args()


def _extract_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    adj = raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Adj Close"]
    if isinstance(adj, pd.DataFrame):
        return adj[ticker].dropna().astype(float)
    return adj.dropna().astype(float)


def _download_first_available(
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[pd.Series, str | None]:
    for ticker in tickers:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            continue
        try:
            series = _extract_series(raw, ticker).rename(ticker)
        except (KeyError, ValueError):
            continue
        if not series.empty:
            return series, ticker
    return pd.Series(dtype="float64"), None


def _add_inverse_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(df.columns):
        if len(col) != 6 or not col.isalpha():
            continue
        inverse_col = f"{col[3:]}{col[:3]}"
        if inverse_col in out.columns:
            continue
        out[inverse_col] = 1.0 / out[col]
    return out


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.Series] = []
    meta: dict[str, object] = {
        "start": args.start,
        "end_exclusive": args.end,
        "source": "yfinance",
        "series": {},
    }

    for name, candidates in DEFAULT_TICKER_CANDIDATES.items():
        series, used_ticker = _download_first_available(candidates, start=args.start, end=args.end)
        series = series.rename(name)
        meta["series"][name] = {
            "candidates": candidates,
            "ticker": used_ticker,
            "rows": int(series.shape[0]),
            "first_date": None if series.empty else str(series.index.min().date()),
            "last_date": None if series.empty else str(series.index.max().date()),
        }
        if name == "USDCNH" and used_ticker == "USDCNY=X":
            meta["series"][name]["proxy_note"] = "USDCNY used as proxy for USDCNH"
        frames.append(series)

    df = pd.concat(frames, axis=1).sort_index()
    if args.fill_missing:
        business_days = pd.date_range(start=args.start, end=pd.Timestamp(args.end) - pd.Timedelta(days=1), freq="B")
        df = df.reindex(business_days).ffill()
    df = _add_inverse_columns(df)
    df.index.name = "date"

    csv_path = args.out_dir / f"{args.filename}.csv"
    meta_path = args.out_dir / f"{args.filename}.meta.json"

    df.to_csv(csv_path, float_format="%.10f")

    meta["output_csv"] = str(csv_path)
    meta["columns"] = list(df.columns)
    meta["rows"] = int(df.shape[0])
    meta["first_date"] = None if df.empty else str(df.index.min().date())
    meta["last_date"] = None if df.empty else str(df.index.max().date())
    meta["fill_missing"] = bool(args.fill_missing)
    meta["missing_after_fill"] = {col: int(df[col].isna().sum()) for col in df.columns}

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
