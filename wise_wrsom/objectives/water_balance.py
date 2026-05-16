"""水量平衡目标函数。"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..protocols.routing import RoutingResult
from .registry import register


@register("water_balance")
class WaterBalanceObjective:
    """水量平衡目标：最小化调度总水量与目标水量的偏差。

    direction="min"，作为独立目标参与多目标优化，
    确保水量约束不会被其他目标的权重向量掩盖。
    """

    @property
    def name(self) -> str:
        return "水量平衡"

    @property
    def direction(self) -> Literal["max", "min"]:
        return "min"

    def compute(
        self,
        routing_result: RoutingResult,
        population: np.ndarray,
        total_water: float,
    ) -> np.ndarray:
        downstream_flow = np.array(routing_result.downstream_flows[0])
        actual_water = np.sum(downstream_flow, axis=1) * 24 * 60 * 60
        relative_violation = (actual_water - total_water) / total_water
        tolerance = 0.10
        excess = np.maximum(np.abs(relative_violation) - tolerance, 0.0)
        return np.square(excess)

    def apply_penalty(
        self,
        objective_value: np.ndarray,
        penalty: np.ndarray,
    ) -> np.ndarray:
        return objective_value