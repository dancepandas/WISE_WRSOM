"""全线通水时长目标函数。"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..protocols.routing import RoutingResult
from .registry import register


@register("water_duration")
class WaterDurationObjective:
    """全线通水时长目标：最大化所有河段都有水的天数。"""

    @property
    def name(self) -> str:
        return "全线通水时长"

    @property
    def direction(self) -> Literal["max", "min"]:
        return "max"

    def compute(
        self,
        routing_result: RoutingResult,
        population: np.ndarray,
        total_water: float,
    ) -> np.ndarray:
        return routing_result.water_duration

    def apply_penalty(
        self,
        objective_value: np.ndarray,
        penalty: np.ndarray,
    ) -> np.ndarray:
        scale = np.maximum(np.abs(objective_value).max(), 1.0)
        return objective_value - scale * penalty
