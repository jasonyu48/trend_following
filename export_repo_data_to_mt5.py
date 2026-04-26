from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_paths import DEFAULT_DATA_DIR
from market_data import load_symbol_m1_bid_ask
from symbol_universe import DEFAULT_SYMBOLS
MT5_SYMBOL_ALIASES = {
    "JP225": "JPN225",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export repo Dukascopy parquet data and symbol_specs.json for MT5 custom symbol import.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--symbol-specs", type=Path, default=Path("symbol_specs.json"))
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", type=str, default="20120101")
    p.add_argument("--end", type=str, default="20250101")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--custom-suffix", type=str, default="_DUKA")
    p.add_argument("--custom-prefix", type=str, default=None, help="Deprecated. Prefer --custom-suffix to keep FX base/quote parsing intact in MT5.")
    return p.parse_args()


def load_symbol_specs(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(symbol).upper(): dict(spec)
        for symbol, spec in payload.items()
        if str(symbol) != "_meta" and isinstance(spec, dict)
    }


def digits_from_tick_size(tick_size: float) -> int:
    text = f"{tick_size:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def mt5_custom_symbol_name(symbol: str, *, prefix: str | None = None, suffix: str = "_DUKA") -> str:
    base = MT5_SYMBOL_ALIASES.get(symbol.upper(), symbol.upper())
    if prefix:
        return f"{prefix}{base}"
    return f"{base}{suffix}"


def instrument_calc_mode(instrument_type: str) -> str:
    text = str(instrument_type).strip().lower()
    if text == "fx_cfd":
        return "forex"
    if text in {"metal_cfd", "index_cfd"}:
        return "cfd"
    return "cfd"


def export_symbol_bars(
    *,
    output_path: Path,
    m1: pd.DataFrame,
    tick_size: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(index=m1.index.copy())
    frame["Date"] = frame.index.tz_convert("UTC").strftime("%Y.%m.%d")
    frame["Time"] = frame.index.tz_convert("UTC").strftime("%H:%M:%S")
    frame["Open"] = pd.to_numeric(m1["bid_open"], errors="coerce")
    frame["High"] = pd.to_numeric(m1["bid_high"], errors="coerce")
    frame["Low"] = pd.to_numeric(m1["bid_low"], errors="coerce")
    frame["Close"] = pd.to_numeric(m1["bid_close"], errors="coerce")
    frame["TickVolume"] = pd.to_numeric(m1["bid_volume"], errors="coerce").fillna(0.0).round().astype("int64")
    spread_points = ((pd.to_numeric(m1["ask_open"], errors="coerce") - pd.to_numeric(m1["bid_open"], errors="coerce")) / float(tick_size))
    frame["Spread"] = spread_points.round().fillna(0.0).astype("int64")
    frame["RealVolume"] = 0
    frame.to_csv(output_path, index=False, lineterminator="\n")


def main() -> int:
    args = parse_args()
    symbols = [str(symbol).upper() for symbol in args.symbols]
    symbol_specs = load_symbol_specs(args.symbol_specs)

    output_dir = args.output_dir
    bars_dir = output_dir / "bars"
    output_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)

    spec_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        if symbol not in symbol_specs:
            raise KeyError(f"Missing symbol spec for {symbol!r} in {args.symbol_specs}")
        spec = symbol_specs[symbol]
        tick_size = float(spec["tick_size"])
        m1 = load_symbol_m1_bid_ask(
            data_dir=args.data_dir,
            symbol=symbol,
            start=args.start,
            end=args.end,
        )
        custom_symbol = mt5_custom_symbol_name(
            symbol,
            prefix=args.custom_prefix,
            suffix=args.custom_suffix,
        )
        bars_filename = f"{custom_symbol}_M1.csv"
        export_symbol_bars(
            output_path=bars_dir / bars_filename,
            m1=m1,
            tick_size=tick_size,
        )

        row = {
            "custom_symbol": custom_symbol,
            "source_symbol": symbol,
            "bars_csv": f"bars/{bars_filename}",
            "instrument_type": str(spec.get("instrument_type", "cfd")),
            "calc_mode": instrument_calc_mode(spec.get("instrument_type", "cfd")),
            "digits": int(digits_from_tick_size(tick_size)),
            "tick_size": tick_size,
            "contract_size": float(spec["contract_multiplier"]),
            "volume_min": float(spec["min_lot"]),
            "volume_step": float(spec["lot_step"]),
            "volume_max": 1000000.0,
            "currency_base": str(spec.get("base_currency", "")),
            "currency_profit": str(spec.get("pnl_currency", spec.get("quote_currency", "USD"))),
            "currency_margin": str(spec.get("margin_currency", spec.get("account_currency", "USD"))),
            "initial_margin_ratio": float(spec.get("initial_margin_ratio", 0.0)),
            "maintenance_margin_ratio": float(spec.get("maintenance_margin_ratio", 0.0)),
            "description": f"Dukascopy custom symbol imported from repo data for {symbol}",
        }
        spec_rows.append(row)
        manifest_rows.append(
            {
                "custom_symbol": custom_symbol,
                "source_symbol": symbol,
                "rows_exported": int(len(m1)),
                "start_utc": str(m1.index[0]) if not m1.empty else None,
                "end_utc": str(m1.index[-1]) if not m1.empty else None,
            }
        )

    spec_csv_path = output_dir / "symbols.csv"
    with spec_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(spec_rows[0].keys()))
        writer.writeheader()
        writer.writerows(spec_rows)

    manifest = {
        "output_dir": str(output_dir),
        "data_dir": str(args.data_dir),
        "symbol_specs_path": str(args.symbol_specs),
        "symbols": manifest_rows,
        "notes": [
            "Bars are exported from repo bid-side M1 parquet with spread points derived from ask_open-bid_open.",
            "Custom symbol names may differ from source symbols where MT5-friendly aliases are preferred.",
            "By default, names use a suffix like GBPJPY_DUKA so MT5 can still infer FX base/profit currencies from the leading symbol code.",
            "Use the paired MQL5 importer script to create/update custom symbols and import bars.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"symbols_csv": str(spec_csv_path), "manifest": str(output_dir / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
