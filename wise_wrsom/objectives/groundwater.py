"""地下水回补目标函数。"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..protocols.routing import RoutingResult
from .registry import register


@register("groundwater_recharge")
class GroundwaterRechargeObjective:
    """地下水回补目标：最大化通过渗漏损失补给地下水的水量。"""

    @property
    def name(self) -> str:
        return "地下水回补水量"

    @property
    def direction(self) -> Literal["max", "min"]:
        return "max"

    def compute(
        self,
        routing_result: RoutingResult,
        population: np.ndarray,
        total_water: float,
    ) -> np.ndarray:
        return routing_result.infiltration

    def apply_penalty(
        self,
        objective_value: np.ndarray,
        penalty: np.ndarray,
    ) -> np.ndarray:
        scale = np.maximum(np.abs(objective_value).max(), 1.0)
        return objective_value - scale * penalty
