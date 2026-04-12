from __future__ import annotations

from strategies.base import StrategyDefinition
from strategies.lr_slope_anchor_breakout import STRATEGY as LR_SLOPE_ANCHOR_BREAKOUT
from strategies.ma_divergence_momentum_confirm import STRATEGY as MA_DIVERGENCE_MOMENTUM_CONFIRM
from strategies.ma_atr_breakout import STRATEGY as MA_ATR_BREAKOUT
from strategies.rsi_trend_following import STRATEGY as RSI_TREND_FOLLOWING


STRATEGIES: dict[str, StrategyDefinition] = {
    MA_ATR_BREAKOUT.name: MA_ATR_BREAKOUT,
    MA_DIVERGENCE_MOMENTUM_CONFIRM.name: MA_DIVERGENCE_MOMENTUM_CONFIRM,
    LR_SLOPE_ANCHOR_BREAKOUT.name: LR_SLOPE_ANCHOR_BREAKOUT,
    RSI_TREND_FOLLOWING.name: RSI_TREND_FOLLOWING,
}


def get_strategy(name: str) -> StrategyDefinition:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc
