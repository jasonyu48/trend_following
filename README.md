# Trend Following Research Repo

Systematic trend-following research and workflow scripts for multi-symbol backtests, timeframe selection, parameter search, and sensitivity analysis.

## Main entry points
- `run_cta_workflow.py`: end-to-end portfolio workflow.
- `run_backtest.py`: single run/backtest utilities and output saving.
- `run_single_symbol_param_sweep.py`: single-symbol sweep or heatmap runs.
- `run_supertrend_portfolio_dev.py`: timeframe-grid x supertrend-grid portfolio development workflow.
- `build_trading_session_data.py`: rebuild filtered market data from raw parquet files.

Note: the full grid parameter search in `run_cta_workflow.py` does not work well. Please First use the `heatmap` mode in `run_single_symbol_param_sweep.py` to find parameters, and then use `skip-param-search` in `run_cta_workflow.py`.

## Data layout
- Raw M1 data lives in `data/`.
- The default backtest dataset is `data_trading_sessions/`.
- Default paths are defined in `data_paths.py`.

Most scripts use `data_trading_sessions` by default, not `data`.

## Data processing
Filtered data in `data_trading_sessions` is built from `data/` with several layers of cleanup:

1. Trading-session filtering  
   Each symbol keeps only minutes that match its configured UTC trading windows in `trading_sessions.py`.
   The session rules are reusable code, not one-off cleanup scripts:
   - DST-aware rules for `FX`, `metals`, `US indices`, `US stocks`, and `GER40`
   - historical staged rules for `BTCUSD`, `ETHUSD`, and `HK50`

2. Stale-quote filtering  
   `build_trading_session_data.py` joins bid/ask first, then removes long runs where the full quote is unchanged:
   - `GBPJPY`, `USDJPY`, `USDCHF`, `USDCNH`: remove runs `>= 60` minutes
   - `XAUUSD`, `XAGUSD`, `HK50`, `US30`, `SP500`, `GER40`, `TESLA`, `NVDA`, `AAPL`: remove runs `>= 30` minutes
   - `BTCUSD`, `ETHUSD`: remove runs `>= 180` minutes
   - `JP225`: no generic stale threshold is used here; see the symbol-specific start below

3. Bad-quote filtering  
   `ETHUSD` gets an extra joined bid/ask filter in `build_trading_session_data.py`:
   - remove obviously crossed quotes
   - remove minutes with extreme spread
   - remove one-sided quote spikes where one side jumps sharply and the other side does not
   Both `bid` and `ask` are removed for the affected minute.

4. Symbol-specific start trimming  
   `JP225` is trimmed to start at `2015-01-01 00:00:00+00:00` in the filtered dataset to avoid an earlier period with materially worse data quality.
   `US30` is trimmed to start at `2012-07-16 13:30:00+00:00`.

The `JP225` and `US30` starts are currently defined in `build_trading_session_data.py` via `SYMBOL_MIN_START_UTC`.

## Reproducible rebuild flow
If you want to reproduce the dataset from scratch, use this order:

1. Download raw M1 data into `data/`
2. Download daily FX conversion data into `data/fx_daily/`
3. Rebuild the filtered dataset into `data_trading_sessions/`
4. Run sanity checks on the filtered output before backtesting

PowerShell example:

Note: replace `C:\Workspace\quantamental\.venv\Scripts\python.exe` with your own environment.

```powershell
& "C:\Workspace\quantamental\.venv\Scripts\python.exe" .\download_dukascopy_m1.py `
  --symbols GBPJPY USDJPY USDCHF USDCNH GBPUSD CADJPY XAUUSD XAGUSD BTCUSD ETHUSD HK50 JP225 US30 SP500 GER40 TESLA NVDA AAPL `
  --start 2012-01-01T00:00:00Z `
  --end 2025-01-01T00:00:00Z `
  --side both `
  --format parquet `
  --output-dir data

& "C:\Workspace\quantamental\.venv\Scripts\python.exe" .\download_fx_daily.py `
  --start 2012-01-01 `
  --end 2026-01-01 `
  --out-dir data\fx_daily `
  --filename fx_daily_2012_2025

& "C:\Workspace\quantamental\.venv\Scripts\python.exe" .\build_trading_session_data.py --overwrite
```

Why this matters:
- this repo previously hit parquet compatibility issues when files were written with a different `pyarrow` version than the one used to read them
- rebuilding with the same Python environment used for backtests avoids that mismatch
- the filtering rules live in repo code, so rebuilding is the intended way to keep data consistent after rule changes

## Current session rules
Session rules are configured in `trading_sessions.py` and are minute-based UTC windows. Some are DST-aware and some are historical/staged.

- `GBPJPY`, `USDJPY`, `USDCHF`, `GBPUSD`, `CADJPY`, `USDCNH`:
  - US DST: Sun `21:00-24:00`, Mon-Thu `00:00-24:00`, Fri `00:00-21:00`
  - US winter: Sun `22:00-24:00`, Mon-Thu `00:00-24:00`, Fri `00:00-22:00`

- `XAUUSD`, `XAGUSD`:
  - US DST: Sun `22:00-24:00`, Mon-Thu `00:00-21:00` and `22:00-24:00`, Fri `00:00-21:00`
  - US winter: Sun `23:00-24:00`, Mon-Thu `00:00-22:00` and `23:00-24:00`, Fri `00:00-22:00`

- `US30`, `SP500`:
  - US DST: Sun `22:00-24:00`, Mon-Thu `00:00-20:15` and `22:00-24:00`, Fri `00:00-20:15`
  - US winter: Sun `23:00-24:00`, Mon-Thu `00:00-21:15` and `23:00-24:00`, Fri `00:00-21:15`

- `GER40`:
  - EU DST: Sun `22:00-24:00`, Mon-Thu `00:00-20:15` and `22:00-24:00`, Fri `00:00-20:15`
  - EU winter: Sun `23:00-24:00`, Mon-Thu `00:00-21:00` and `23:00-24:00`, Fri `00:00-21:00`

- `AAPL`, `NVDA`, `TESLA`:
  - US DST: Mon-Fri `13:30-20:00`
  - US winter: Mon-Fri `14:30-21:00`

- `HK50`:
  - `2013-2014`: Mon-Fri `01:15-04:00`, `05:00-08:15`
  - `2015-2017`: Mon-Fri `01:20-04:00`, `05:00-08:15`, `09:00-15:45`
  - `2018+`: Mon-Fri `01:15-04:00`, `05:00-08:30`, `09:15-17:00`

- `JP225`:
  - Mon-Thu: `00:00-22:00`, `23:00-24:00`
  - Fri: `00:00-22:00`
  - Sun: `23:00-24:00`

- `BTCUSD`, `ETHUSD`:
  - before `2019-01-01`: Sun `21:00-24:00`, Mon-Thu `00:00-24:00`, Fri `00:00-21:00`, Sat closed
  - `2019+`: `24/7`

These rules are intended to be closer to actual Dukascopy coverage than the original broad filters, but they are still repo-level approximations rather than broker-certified specifications. Holiday closures are not modeled separately.

## Important JP225 note
`JP225` had two separate data-quality issues in the older history:
- off-session placeholder data
- long stale quote runs inside plausible trading hours

Because of that, the repo currently treats `JP225` more conservatively than other symbols:
- tighter session filtering
- filtered-dataset start forced to `2015-01-01 UTC`

If you move that start earlier, you should re-check stale runs before trusting backtest results.

## Workflow defaults and behavior
`run_cta_workflow.py` defaults to:
- data dir: `data_trading_sessions`
- strategy: `supertrend`
- opt symbol: `XAUUSD`
- opt timeframe: `2H`
- selection period: `20120101` to `20200101`
- portfolio period: `20120101` to `20250101`
- timeframe selection mode: `atr50_hl_sum_ratio_median_match`

The workflow can:
- search parameters on the opt symbol
- assign a timeframe per symbol
- run a final portfolio backtest
- write charts, bars, trades, summaries, and validation outputs under `results/...`

## Parameter sweep script
`run_single_symbol_param_sweep.py` supports:
- `sweep` mode: vary one parameter at a time while others stay fixed
- `heatmap` mode: currently used for full-grid `supertrend` heatmaps

Outputs are written under `results/` and the default output directory encodes strategy, mode, and timeframe.

## Supertrend portfolio development script
`run_supertrend_portfolio_dev.py` defaults to:
- strategy: `supertrend`
- start/end: `20120101` to `20200101`
- selection window: `20120101` to `20200101`
- opt symbol: `XAUUSD`
- symbols: `XAUUSD GBPJPY`
- portfolio mode: `fixed_risk`
- opposite-signal action: `close_and_reverse`
- timeframe candidates: `1H 2H 3H 4H 6H 8H 12H`
- other-symbol timeframe matching: `atr50_hl_sum_ratio_median_match`

For each opt timeframe, it:
- matches the other symbols once on the selection window
- runs the full `supertrend` grid over the full portfolio window
- writes one `heatmap_results.jsonl` plus one `summary.json` under `results/.../<timeframe>/`
- writes top-level combined heatmaps for `max_recovery_time`, `total_return`, and `calmar`

Example:

```powershell
.\.venv\Scripts\python.exe .\run_supertrend_portfolio_dev.py --symbols XAUUSD GBPJPY
```

## Easy-to-miss engineering details
- The default dataset is filtered data, not raw data.
- `build_trading_session_data.py` writes bid/ask files atomically via temp files.
- Stale filtering is done on joined bid/ask quotes, not per side independently.
- `ETHUSD` bad-minute filtering is also done on joined bid/ask quotes, not on a single side.
- `JP225` and `US30` start trimming happen during data generation, not in backtest logic.
- Many workflow scripts depend on `DEFAULT_DATA_DIR`; if you rebuild data elsewhere, pass `--data-dir` explicitly.
- Some older result folders may reflect earlier filtering rules or earlier datasets. Do not compare runs blindly unless they used the same filtered data build.
- `download_fx_daily.py` is reusable and should be rerun when you add symbols with new PnL currencies or need fresh FX conversions.
- `trading_sessions.py` is reusable repo infrastructure, not a scratch analysis file. If session assumptions change, rebuild the filtered dataset instead of patching results by hand.

## Sanity checks
After rebuilding data, useful quick checks are:

```powershell
.\.venv\Scripts\python.exe -c "from market_data import load_symbol_m1_bid_ask; jp=load_symbol_m1_bid_ask('data_trading_sessions','JP225'); print(jp.index.min(), jp.index.max(), len(jp))"
```

```powershell
.\.venv\Scripts\python.exe -c "from market_data import load_symbol_m1_bid_ask; us30=load_symbol_m1_bid_ask('data_trading_sessions','US30'); print(us30.index.min(), us30.index.max(), len(us30))"
```

```powershell
.\.venv\Scripts\python.exe -c "from market_data import load_symbol_m1_bid_ask; eth=load_symbol_m1_bid_ask('data_trading_sessions','ETHUSD'); print(eth.index.min(), eth.index.max(), len(eth))"
```

You should see:
- `JP225` start after its configured filtered start
- `US30` start after `2012-07-16 13:30:00+00:00`
- `ETHUSD` load from `data_trading_sessions` without the known bad quote minutes
