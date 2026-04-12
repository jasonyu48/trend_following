from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable

import pandas as pd


class StrategyDefinition:
    def __init__(
        self,
        *,
        name: str,
        params_type: type,
        execution_style: str,
        compute_features: Callable[[pd.DataFrame, Any], pd.DataFrame],
        build_signal_schedule: Callable[[pd.DataFrame, pd.DataFrame, Any], pd.DataFrame],
        default_grid: Callable[[], dict[str, list]],
        is_entry_allowed: Callable[[pd.Timestamp, Any], bool] | None = None,
    ) -> None:
        self.name = name
        self.params_type = params_type
        self.execution_style = execution_style
        self.compute_features = compute_features
        self.build_signal_schedule = build_signal_schedule
        self.default_grid = default_grid
        self.is_entry_allowed = is_entry_allowed or (lambda ts, params: True)

    @property
    def param_names(self) -> list[str]:
        return [field.name for field in fields(self.params_type)]

    def make_params(self, **kwargs: Any) -> Any:
        return self.params_type(**kwargs)
