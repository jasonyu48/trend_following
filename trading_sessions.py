from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd


MinuteWindow = tuple[int, int]
DstRegion = str


@dataclass(frozen=True)
class SessionWindows:
    mon_thu_windows_utc: tuple[MinuteWindow, ...]
    friday_windows_utc: tuple[MinuteWindow, ...]
    saturday_windows_utc: tuple[MinuteWindow, ...]
    sunday_windows_utc: tuple[MinuteWindow, ...]

    def by_day(self) -> dict[int, tuple[MinuteWindow, ...]]:
        return {
            0: self.mon_thu_windows_utc,
            1: self.mon_thu_windows_utc,
            2: self.mon_thu_windows_utc,
            3: self.mon_thu_windows_utc,
            4: self.friday_windows_utc,
            5: self.saturday_windows_utc,
            6: self.sunday_windows_utc,
        }


@dataclass(frozen=True)
class SymbolSessionRule:
    summer: SessionWindows
    winter: SessionWindows
    dst_region: DstRegion


@dataclass(frozen=True)
class HistoricalSessionStage:
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp | None
    rule: SymbolSessionRule


FULL_DAY_WINDOW: MinuteWindow = (0, 24 * 60)
NO_DST: DstRegion = "none"
US_DST: DstRegion = "us"
EU_DST: DstRegion = "eu"

FX_SUMMER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=(FULL_DAY_WINDOW,),
    friday_windows_utc=((0, 21 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=((21 * 60, 24 * 60),),
)
FX_WINTER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=(FULL_DAY_WINDOW,),
    friday_windows_utc=((0, 22 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=((22 * 60, 24 * 60),),
)
FX_LIKE_RULE = SymbolSessionRule(
    summer=FX_SUMMER_WINDOWS,
    winter=FX_WINTER_WINDOWS,
    dst_region=US_DST,
)
METALS_SUMMER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((0, 21 * 60), (22 * 60, 24 * 60)),
    friday_windows_utc=((0, 21 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=((22 * 60, 24 * 60),),
)
METALS_WINTER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((0, 22 * 60), (23 * 60, 24 * 60)),
    friday_windows_utc=((0, 22 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=((23 * 60, 24 * 60),),
)
METALS_RULE = SymbolSessionRule(
    summer=METALS_SUMMER_WINDOWS,
    winter=METALS_WINTER_WINDOWS,
    dst_region=US_DST,
)
USDCNH_RULE = SymbolSessionRule(
    summer=FX_SUMMER_WINDOWS,
    winter=FX_WINTER_WINDOWS,
    dst_region=US_DST,
)
HK50_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((75, 240), (300, 510), (555, 1140)),
    friday_windows_utc=((75, 240), (300, 510), (555, 1140)),
    saturday_windows_utc=(),
    sunday_windows_utc=(),
)
HK50_RULE = SymbolSessionRule(
    summer=HK50_WINDOWS,
    winter=HK50_WINDOWS,
    dst_region=NO_DST,
)
JP225_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((0, 22 * 60), (23 * 60, 24 * 60)),
    friday_windows_utc=((0, 22 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=((23 * 60, 24 * 60),),
)
JP225_RULE = SymbolSessionRule(
    summer=JP225_WINDOWS,
    winter=JP225_WINDOWS,
    dst_region=NO_DST,
)
US_INDEX_SUMMER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((0, 20 * 60 + 15), (22 * 60, 24 * 60)),
    friday_windows_utc=((0, 20 * 60 + 15),),
    saturday_windows_utc=(),
    sunday_windows_utc=((22 * 60, 24 * 60),),
)
US_INDEX_WINTER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((0, 21 * 60 + 15), (23 * 60, 24 * 60)),
    friday_windows_utc=((0, 21 * 60 + 15),),
    saturday_windows_utc=(),
    sunday_windows_utc=((23 * 60, 24 * 60),),
)
US_INDEX_RULE = SymbolSessionRule(
    summer=US_INDEX_SUMMER_WINDOWS,
    winter=US_INDEX_WINTER_WINDOWS,
    dst_region=US_DST,
)
GER40_SUMMER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((0, 20 * 60 + 15), (22 * 60, 24 * 60)),
    friday_windows_utc=((0, 20 * 60 + 15),),
    saturday_windows_utc=(),
    sunday_windows_utc=((22 * 60, 24 * 60),),
)
GER40_WINTER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((0, 21 * 60), (23 * 60, 24 * 60)),
    friday_windows_utc=((0, 21 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=((23 * 60, 24 * 60),),
)
GER40_RULE = SymbolSessionRule(
    summer=GER40_SUMMER_WINDOWS,
    winter=GER40_WINTER_WINDOWS,
    dst_region=EU_DST,
)
US_STOCK_SUMMER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((13 * 60 + 30, 20 * 60),),
    friday_windows_utc=((13 * 60 + 30, 20 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=(),
)
US_STOCK_WINTER_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((14 * 60 + 30, 21 * 60),),
    friday_windows_utc=((14 * 60 + 30, 21 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=(),
)
US_STOCK_RULE = SymbolSessionRule(
    summer=US_STOCK_SUMMER_WINDOWS,
    winter=US_STOCK_WINTER_WINDOWS,
    dst_region=US_DST,
)
CRYPTO_WINDOWS = SessionWindows(
    mon_thu_windows_utc=(FULL_DAY_WINDOW,),
    friday_windows_utc=(FULL_DAY_WINDOW,),
    saturday_windows_utc=(FULL_DAY_WINDOW,),
    sunday_windows_utc=(FULL_DAY_WINDOW,),
)
CRYPTO_EARLY_WINDOWS = SessionWindows(
    mon_thu_windows_utc=(FULL_DAY_WINDOW,),
    friday_windows_utc=((0, 21 * 60),),
    saturday_windows_utc=(),
    sunday_windows_utc=((21 * 60, 24 * 60),),
)
CRYPTO_RULE = SymbolSessionRule(
    summer=CRYPTO_WINDOWS,
    winter=CRYPTO_WINDOWS,
    dst_region=NO_DST,
)
CRYPTO_EARLY_RULE = SymbolSessionRule(
    summer=CRYPTO_EARLY_WINDOWS,
    winter=CRYPTO_EARLY_WINDOWS,
    dst_region=NO_DST,
)
HK50_2013_2014_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((75, 240), (300, 495)),
    friday_windows_utc=((75, 240), (300, 495)),
    saturday_windows_utc=(),
    sunday_windows_utc=(),
)
HK50_2015_2017_WINDOWS = SessionWindows(
    mon_thu_windows_utc=((80, 240), (300, 495), (540, 945)),
    friday_windows_utc=((80, 240), (300, 495), (540, 945)),
    saturday_windows_utc=(),
    sunday_windows_utc=(),
)
HK50_2013_2014_RULE = SymbolSessionRule(
    summer=HK50_2013_2014_WINDOWS,
    winter=HK50_2013_2014_WINDOWS,
    dst_region=NO_DST,
)
HK50_2015_2017_RULE = SymbolSessionRule(
    summer=HK50_2015_2017_WINDOWS,
    winter=HK50_2015_2017_WINDOWS,
    dst_region=NO_DST,
)


SYMBOL_SESSION_RULES: dict[str, SymbolSessionRule] = {
    "GBPJPY": FX_LIKE_RULE,
    "USDJPY": FX_LIKE_RULE,
    "USDCHF": FX_LIKE_RULE,
    "USDCNH": USDCNH_RULE,
    "GBPUSD": FX_LIKE_RULE,
    "CADJPY": FX_LIKE_RULE,
    "XAUUSD": METALS_RULE,
    "XAGUSD": METALS_RULE,
    "BTCUSD": CRYPTO_RULE,
    "ETHUSD": CRYPTO_RULE,
    "JP225": JP225_RULE,
    "US30": US_INDEX_RULE,
    "SP500": US_INDEX_RULE,
    "GER40": GER40_RULE,
    "TESLA": US_STOCK_RULE,
    "TSLA": US_STOCK_RULE,
    "NVDA": US_STOCK_RULE,
    "AAPL": US_STOCK_RULE,
    "HK50": HK50_RULE,
}
HISTORICAL_SESSION_RULES: dict[str, tuple[HistoricalSessionStage, ...]] = {
    "BTCUSD": (
        HistoricalSessionStage(
            start_utc=pd.Timestamp.min.tz_localize("UTC"),
            end_utc=pd.Timestamp("2019-01-01 00:00:00+00:00"),
            rule=CRYPTO_EARLY_RULE,
        ),
        HistoricalSessionStage(
            start_utc=pd.Timestamp("2019-01-01 00:00:00+00:00"),
            end_utc=None,
            rule=CRYPTO_RULE,
        ),
    ),
    "ETHUSD": (
        HistoricalSessionStage(
            start_utc=pd.Timestamp.min.tz_localize("UTC"),
            end_utc=pd.Timestamp("2019-01-01 00:00:00+00:00"),
            rule=CRYPTO_EARLY_RULE,
        ),
        HistoricalSessionStage(
            start_utc=pd.Timestamp("2019-01-01 00:00:00+00:00"),
            end_utc=None,
            rule=CRYPTO_RULE,
        ),
    ),
    "HK50": (
        HistoricalSessionStage(
            start_utc=pd.Timestamp.min.tz_localize("UTC"),
            end_utc=pd.Timestamp("2015-01-01 00:00:00+00:00"),
            rule=HK50_2013_2014_RULE,
        ),
        HistoricalSessionStage(
            start_utc=pd.Timestamp("2015-01-01 00:00:00+00:00"),
            end_utc=pd.Timestamp("2018-01-01 00:00:00+00:00"),
            rule=HK50_2015_2017_RULE,
        ),
        HistoricalSessionStage(
            start_utc=pd.Timestamp("2018-01-01 00:00:00+00:00"),
            end_utc=None,
            rule=HK50_RULE,
        ),
    ),
}


def _minute_window_mask(
    idx: pd.DatetimeIndex,
    weekdays: pd.Series,
    windows_by_day: dict[int, tuple[MinuteWindow, ...]],
) -> pd.Series:
    minute_of_day = pd.Series(idx.hour * 60 + idx.minute, index=idx)
    mask = pd.Series(False, index=idx)
    for weekday, windows in windows_by_day.items():
        day_mask = weekdays == weekday
        for start_minute, end_minute in windows:
            mask |= day_mask & minute_of_day.ge(start_minute) & minute_of_day.lt(end_minute)
    return mask


def _nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    delta_days = (weekday - first.dayofweek) % 7
    day = 1 + delta_days + 7 * (occurrence - 1)
    return pd.Timestamp(year=year, month=month, day=day, tz="UTC")


def _last_weekday_of_month(year: int, month: int, weekday: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
    delta_days = (last.dayofweek - weekday) % 7
    return pd.Timestamp(year=year, month=month, day=int(last.day - delta_days), tz="UTC")


@lru_cache(maxsize=None)
def _dst_window_utc(region: DstRegion, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    if region == US_DST:
        start = _nth_weekday_of_month(year, 3, weekday=6, occurrence=2) + pd.Timedelta(hours=7)
        end = _nth_weekday_of_month(year, 11, weekday=6, occurrence=1) + pd.Timedelta(hours=6)
        return start, end
    if region == EU_DST:
        start = _last_weekday_of_month(year, 3, weekday=6) + pd.Timedelta(hours=1)
        end = _last_weekday_of_month(year, 10, weekday=6) + pd.Timedelta(hours=1)
        return start, end
    raise ValueError(f"Unsupported DST region: {region}")


def _dst_mask(idx: pd.DatetimeIndex, region: DstRegion) -> pd.Series:
    if region == NO_DST:
        return pd.Series(False, index=idx)
    idx_utc = idx if str(idx.tz) == "UTC" else idx.tz_convert("UTC")
    mask = pd.Series(False, index=idx_utc)
    years = sorted(set(int(year) for year in idx_utc.year))
    for year in years:
        start, end = _dst_window_utc(region, year)
        mask |= (idx_utc >= start) & (idx_utc < end)
    return mask


def _session_mask_for_rule(idx: pd.DatetimeIndex, rule: SymbolSessionRule) -> pd.Series:
    weekdays = pd.Series(idx.dayofweek, index=idx)
    if rule.dst_region == NO_DST:
        return _minute_window_mask(idx, weekdays, rule.winter.by_day())
    is_dst = _dst_mask(idx, rule.dst_region)
    summer_mask = _minute_window_mask(idx, weekdays, rule.summer.by_day())
    winter_mask = _minute_window_mask(idx, weekdays, rule.winter.by_day())
    return (summer_mask & is_dst) | (winter_mask & ~is_dst)


def trading_session_mask(index: pd.DatetimeIndex, symbol: str) -> pd.Series:
    symbol = str(symbol).upper()
    idx = index if index.tz is not None else index.tz_localize("UTC")
    idx = idx.tz_convert("UTC")
    stages = HISTORICAL_SESSION_RULES.get(symbol)
    if stages:
        mask = pd.Series(False, index=idx)
        for stage in stages:
            stage_idx_mask = idx >= stage.start_utc
            if stage.end_utc is not None:
                stage_idx_mask &= idx < stage.end_utc
            if not bool(stage_idx_mask.any()):
                continue
            stage_index = idx[stage_idx_mask]
            mask.loc[stage_index] = _session_mask_for_rule(stage_index, stage.rule).to_numpy()
        return mask
    rule = SYMBOL_SESSION_RULES[symbol]
    return _session_mask_for_rule(idx, rule)


def filter_frame_to_trading_sessions(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = trading_session_mask(pd.DatetimeIndex(frame.index), symbol)
    return frame.loc[mask.to_numpy()].copy()
