"""
Download Dukascopy M1 OHLC (UTC) via dukascopy-python — one HTTP stream per calendar year.

Install (in your venv): pip install dukascopy-python tqdm
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import dukascopy_python as dk
import numpy as np
import pandas as pd
from dukascopy_python import instruments
from tqdm import tqdm


def _quiet_custom_logger(debug: bool = False) -> logging.Logger:
    del debug
    lg = logging.getLogger("DUKASCRIPT")
    lg.setLevel(logging.WARNING)
    return lg


dk._get_custom_logger = _quiet_custom_logger


def _quiet_third_party_loggers() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)

# User-facing keys -> Dukascopy freeserv instrument codes (see dukascopy_python.instruments)
SYMBOL_TO_INSTRUMENT: dict[str, str] = {
    "GBPJPY": instruments.INSTRUMENT_FX_CROSSES_GBP_JPY,
    "USDJPY": instruments.INSTRUMENT_FX_MAJORS_USD_JPY,
    "USDCHF": instruments.INSTRUMENT_FX_MAJORS_USD_CHF,
    "USDCNH": instruments.INSTRUMENT_FX_CROSSES_USD_CNH,
    "XAUUSD": instruments.INSTRUMENT_FX_METALS_XAU_USD,
    "XAGUSD": instruments.INSTRUMENT_FX_METALS_XAG_USD,
    "HK50": instruments.INSTRUMENT_IDX_ASIA_E_H_KONG,
    "JP225": instruments.INSTRUMENT_IDX_ASIA_E_N225JAP,
}

DEFAULT_SYMBOLS: list[str] = [
    "GBPJPY",
    "USDJPY",
    "USDCHF",
    "USDCNH",
    "XAUUSD",
    "XAGUSD",
    "HK50",
    "JP225",
]

OFFER_SIDES = {"bid": dk.OFFER_SIDE_BID, "ask": dk.OFFER_SIDE_ASK}


def sides_to_download(side_arg: str) -> list[tuple[str, str]]:
    """Return [(label, offer_side_code), ...] for API fetch."""
    if side_arg == "both":
        return [("bid", OFFER_SIDES["bid"]), ("ask", OFFER_SIDES["ask"])]
    return [(side_arg, OFFER_SIDES[side_arg])]


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def parse_utc_datetime(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def year_chunks(start: datetime, end: datetime) -> list[tuple[int, datetime, datetime]]:
    """Calendar years overlapping [start, end): (year_label, chunk_start, chunk_end)."""
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    out: list[tuple[int, datetime, datetime]] = []
    y = start.year
    while _utc(y, 1, 1) < end:
        ys = _utc(y, 1, 1)
        ye = _utc(y + 1, 1, 1)
        cs = max(start, ys)
        ce = min(end, ye)
        if cs < ce:
            out.append((y, cs, ce))
        y += 1
    return out


def resolve_range(
    years: float | None,
    start: datetime | None,
    end: datetime | None,
) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    if end is None:
        end = now.replace(second=0, microsecond=0)
    if start is None:
        if years is None:
            raise ValueError("Provide --start or --years.")
        start = end - timedelta(days=365.25 * years)
    return start, end


def output_path(output_dir: Path, symbol: str, side: str, year: int, fmt: str) -> Path:
    ext = "parquet" if fmt == "parquet" else "csv"
    return output_dir / f"{symbol}_M1_{side}_{year}.{ext}"


def load_existing(path: Path, fmt: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        if fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    except Exception:
        return None


def save_df(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=True)
    elif fmt == "csv":
        df.to_csv(path, index_label="timestamp")
    else:
        raise ValueError(f"Unknown format: {fmt}")


@dataclass
class AuditResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_in_range: int = 0
    span_days: float = 0.0
    max_gap_minutes: float = 0.0
    gaps_over_10d: int = 0
    weekend_like_gaps: int = 0  # 3d < gap <= 10d


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = out.copy()
    out.index = idx
    return out


def _clip_range(df: pd.DataFrame, lo: datetime, hi: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    d = _ensure_utc_index(df)
    lo = lo.astimezone(timezone.utc)
    hi = hi.astimezone(timezone.utc)
    return d.loc[(d.index >= lo) & (d.index < hi)]


def audit_m1_ohlc(
    df: pd.DataFrame,
    range_start: datetime,
    range_end: datetime,
    *,
    strict_trailing_edge: bool = False,
) -> AuditResult:
    r = AuditResult()
    rs = range_start.astimezone(timezone.utc)
    re = range_end.astimezone(timezone.utc)
    r.span_days = (re - rs).total_seconds() / 86400.0

    if df.empty:
        r.errors.append("empty dataframe")
        return r

    need = {"open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        r.errors.append(f"missing columns: {sorted(missing)}")
        return r

    d = _clip_range(df, rs, re)
    r.rows_in_range = len(d)
    if r.rows_in_range == 0:
        r.errors.append(f"no rows in [{rs.isoformat()}, {re.isoformat()})")
        return r

    if not d.index.is_monotonic_increasing:
        r.errors.append("index is not monotonic increasing")

    dup = int(d.index.duplicated().sum())
    if dup:
        r.errors.append(f"duplicate timestamps: {dup}")

    o, h, lo_, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    if o.isna().any() or h.isna().any() or lo_.isna().any() or c.isna().any():
        r.errors.append("NaN in OHLC")
    if v.isna().any():
        r.warnings.append("NaN in volume (some rows)")

    if np.isinf(o.values).any() or np.isinf(h.values).any():
        r.errors.append("inf in OHLC")

    bad_price = ((o <= 0) | (h <= 0) | (lo_ <= 0) | (c <= 0)).any()
    if bad_price:
        r.errors.append("non-positive OHLC")

    if not ((h >= lo_) & (h >= o) & (h >= c) & (lo_ <= o) & (lo_ <= c)).all():
        r.errors.append("OHLC inconsistent (high/low vs open/close)")

    # Gap analysis on consecutive bars (M1: expect 1 min; market closures create larger gaps)
    delta_min = d.index.to_series().diff().dt.total_seconds().div(60.0)
    delta_min = delta_min.iloc[1:]
    if len(delta_min):
        r.max_gap_minutes = float(delta_min.max())
        over_10d = delta_min > (10 * 24 * 60)
        r.gaps_over_10d = int(over_10d.sum())
        weekend_band = (delta_min > (3 * 24 * 60)) & (delta_min <= (10 * 24 * 60))
        r.weekend_like_gaps = int(weekend_band.sum())
        if r.gaps_over_10d:
            r.warnings.append(
                f"gaps > 10 days: {r.gaps_over_10d} (max gap {r.max_gap_minutes:.0f} min)"
            )

    min_rows = max(50, int(120 * r.span_days))
    if r.rows_in_range < min_rows * 0.4:
        r.warnings.append(
            f"low bar count for span: {r.rows_in_range} rows over {r.span_days:.1f} d "
            f"(heuristic floor ~{min_rows})"
        )

    first_ts = d.index.min()
    last_ts = d.index.max()
    if first_ts > rs + timedelta(days=14):
        r.warnings.append(
            f"first bar late vs range start: {first_ts.isoformat()} > {rs.isoformat()} + 14d"
        )
    # Allow multi-day year-end / holiday closure (48h was too tight for XAU etc.).
    trailing_slack = timedelta(days=5)
    if last_ts < re - trailing_slack:
        msg = (
            f"last bar ends before range end: {last_ts.isoformat()} vs {re.isoformat()}"
        )
        if strict_trailing_edge:
            r.errors.append(msg)
        else:
            r.warnings.append(msg)

    return r


def _chunk_is_historical(chunk_end: datetime, slack: timedelta = timedelta(hours=3)) -> bool:
    return chunk_end.astimezone(timezone.utc) <= datetime.now(timezone.utc) - slack


def year_file_complete(
    path: Path,
    fmt: str,
    chunk_start: datetime,
    chunk_end: datetime,
) -> tuple[bool, AuditResult]:
    df = load_existing(path, fmt)
    if df is None or df.empty:
        return False, AuditResult(errors=["missing or unreadable file"])

    df = _ensure_utc_index(df)

    strict_tail = _chunk_is_historical(chunk_end)
    ar = audit_m1_ohlc(
        df, chunk_start, chunk_end, strict_trailing_edge=strict_tail
    )
    if ar.errors:
        return False, ar
    return True, ar


def symbol_has_pending_work(
    su: str,
    sym_dir: Path,
    chunks_by_year: dict[int, tuple[datetime, datetime]],
    side_jobs: list[tuple[str, str]],
    fmt: str,
) -> bool:
    """Any (year, side) missing or failing validation -> needs download."""
    for y, (cs, ce) in chunks_by_year.items():
        for side_label, _ in side_jobs:
            path = output_path(sym_dir, su, side_label, y, fmt)
            ok, _ = year_file_complete(path, fmt, cs, ce)
            if not ok:
                return True
    return False


def fetch_year(
    instrument: str,
    chunk_start: datetime,
    chunk_end: datetime,
    offer_side: str,
    max_retries: int,
) -> pd.DataFrame:
    chunk_start = chunk_start.astimezone(timezone.utc)
    chunk_end = chunk_end.astimezone(timezone.utc)
    if chunk_end <= chunk_start:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = dk.fetch(
        instrument,
        dk.INTERVAL_MIN_1,
        offer_side,
        chunk_start,
        chunk_end,
        max_retries=max_retries,
    )
    if out is None or out.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out = out[(out.index >= chunk_start) & (out.index < chunk_end)]
    return out


def print_audit(symbol: str, side_label: str, year: int, ar: AuditResult) -> None:
    tqdm.write(
        f"  [{symbol} {side_label} {year}] rows={ar.rows_in_range:,} span={ar.span_days:.1f}d "
        f"max_gap_min={ar.max_gap_minutes:.0f} weekendish_gaps={ar.weekend_like_gaps}"
    )
    for e in ar.errors:
        tqdm.write(f"  ERROR: {e}")
    for w in ar.warnings:
        tqdm.write(f"  WARN: {w}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Dukascopy M1 OHLC (UTC) by calendar year (one HTTP stream per year)."
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols (default: {' '.join(DEFAULT_SYMBOLS)})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root output directory; each symbol is saved under <dir>/<SYMBOL>/ (default: ./data)",
    )
    p.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output file format (default: parquet)",
    )
    p.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="Length of history when --start is omitted (default: 10)",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Range start (ISO-8601, UTC if no offset)",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Range end: keep bars with timestamp < this instant (ISO-8601 UTC)",
    )
    p.add_argument(
        "--side",
        choices=("bid", "ask", "both"),
        default="both",
        help="Bid, ask, or both (default: both — two files per symbol/year)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=15,
        help="HTTP retry budget per year request (default: 15)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a year file exists and validates",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip symbols that already have all (year, side) files passing validation (ignored with --force)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any post-download audit reports errors",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _quiet_third_party_loggers()
    args = parse_args(argv)
    start: datetime | None = parse_utc_datetime(args.start) if args.start else None
    end: datetime | None = parse_utc_datetime(args.end) if args.end else None
    start, end = resolve_range(None if start is not None else args.years, start, end)
    if end <= start:
        print("error: end must be after start", file=sys.stderr)
        return 2

    side_jobs = sides_to_download(args.side)
    unknown = [s for s in args.symbols if s.upper() not in SYMBOL_TO_INSTRUMENT]
    if unknown:
        print(f"error: unknown symbols: {unknown}", file=sys.stderr)
        print(f"known: {sorted(SYMBOL_TO_INSTRUMENT)}", file=sys.stderr)
        return 2

    chunks_by_year: dict[int, tuple[datetime, datetime]] = {}
    for y, cs, ce in year_chunks(start, end):
        chunks_by_year[y] = (cs, ce)

    symbols_to_process = [s.upper() for s in args.symbols]
    if args.only_missing and not args.force:
        done: list[str] = []
        pending: list[str] = []
        for su in symbols_to_process:
            sym_dir = args.output_dir / su
            if symbol_has_pending_work(su, sym_dir, chunks_by_year, side_jobs, args.format):
                pending.append(su)
            else:
                done.append(su)
        for su in done:
            tqdm.write(f"skip symbol (complete): {su}")
        symbols_to_process = pending
        if not symbols_to_process:
            tqdm.write("all listed symbols already complete for this date range and sides.")
            return 0
    elif args.only_missing and args.force:
        tqdm.write("note: --only-missing ignored when using --force")

    audit_failed = False
    for sym in tqdm(symbols_to_process, desc="symbols", unit="sym"):
        su = sym.upper()
        inst = SYMBOL_TO_INSTRUMENT[su]
        sym_dir = args.output_dir / su
        years = sorted(chunks_by_year.keys())
        for side_label, offer in side_jobs:
            for y in tqdm(
                years,
                desc=f"{su} {side_label}",
                unit="yr",
                leave=False,
            ):
                cs, ce = chunks_by_year[y]
                path = output_path(sym_dir, su, side_label, y, args.format)

                if not args.force:
                    ok, ar_existing = year_file_complete(path, args.format, cs, ce)
                    if ok:
                        tqdm.write(f"{su} {side_label} {y}: skip (ok) -> {path}")
                        print_audit(su, side_label, y, ar_existing)
                        continue
                    if path.is_file():
                        tqdm.write(
                            f"{su} {side_label} {y}: re-download (existing file failed checks)"
                        )
                        for e in ar_existing.errors:
                            tqdm.write(f"    was: ERROR {e}")
                        for w in ar_existing.warnings:
                            tqdm.write(f"    was: WARN {w}")

                df = fetch_year(inst, cs, ce, offer, args.max_retries)
                save_df(df, path, args.format)
                tqdm.write(f"{su} {side_label} {y}: {len(df):,} rows -> {path}")

                ar = audit_m1_ohlc(
                    df,
                    cs,
                    ce,
                    strict_trailing_edge=_chunk_is_historical(ce),
                )
                print_audit(su, side_label, y, ar)
                if ar.errors:
                    audit_failed = True

    if args.strict and audit_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
