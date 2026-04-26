from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from data_paths import DEFAULT_DATA_DIR


YEAR_FILE_RE = re.compile(r"(?P<symbol>[A-Z0-9]+)_M1_(?P<side>bid|ask)_(?P<year>\d{4})\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan yearly M1 parquet files for empty files and missing years."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text output.")
    return parser.parse_args()


def detect_year_range(symbol_rows: list[dict[str, object]], start_year: int | None, end_year: int | None) -> tuple[int, int] | None:
    years = sorted({int(row["year"]) for row in symbol_rows})
    if not years and (start_year is None or end_year is None):
        return None
    lo = start_year if start_year is not None else years[0]
    hi = end_year if end_year is not None else years[-1]
    return lo, hi


def main() -> int:
    args = parse_args()
    records: dict[str, list[dict[str, object]]] = {}

    for path in sorted(args.data_dir.glob("*/*.parquet")):
        match = YEAR_FILE_RE.match(path.name)
        if not match:
            continue
        symbol = match.group("symbol").upper()
        side = match.group("side")
        year = int(match.group("year"))
        df = pd.read_parquet(path)
        records.setdefault(symbol, []).append(
            {
                "symbol": symbol,
                "side": side,
                "year": year,
                "path": str(path),
                "rows": int(len(df)),
                "is_empty": bool(df.empty),
            }
        )

    summary: dict[str, object] = {}
    for symbol, rows in sorted(records.items()):
        bounds = detect_year_range(rows, args.start_year, args.end_year)
        sides = {"bid": set(), "ask": set()}
        empty_files: list[str] = []
        for row in rows:
            sides[str(row["side"])].add(int(row["year"]))
            if bool(row["is_empty"]):
                empty_files.append(str(Path(str(row["path"])).name))

        if bounds is None:
            expected_years: list[int] = []
        else:
            expected_years = list(range(bounds[0], bounds[1] + 1))

        missing_by_side = {
            side: [year for year in expected_years if year not in sorted(side_years)]
            for side, side_years in sides.items()
        }

        summary[symbol] = {
            "year_range": None if not bounds else {"start": bounds[0], "end": bounds[1]},
            "empty_files": empty_files,
            "missing_years_by_side": missing_by_side,
            "n_files": len(rows),
        }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    for symbol, info in summary.items():
        print(f"[{symbol}]")
        print(f"  year_range: {info['year_range']}")
        print(f"  empty_files: {len(info['empty_files'])}")
        for name in info["empty_files"]:
            print(f"    - {name}")
        print(f"  missing_years_bid: {info['missing_years_by_side']['bid']}")
        print(f"  missing_years_ask: {info['missing_years_by_side']['ask']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
