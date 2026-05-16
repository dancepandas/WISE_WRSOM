"""出境水量控制目标函数。"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..protocols.routing import RoutingResult
from .registry import register


@register("outflow_control")
class OutflowControlObjective:
    """出境水量控制目标：最小化出境水量。

    direction="min"，compute 返回原始出流量值。
    TOPSIS 归一化时需将 min 目标取反以统一为最大化问题。
    """

    @property
    def name(self) -> str:
        return "出境水量控制"

    @property
    def direction(self) -> Literal["max", "min"]:
        return "min"

    def compute(
        self,
        routing_result: RoutingResult,
        population: np.ndarray,
        total_water: float,
    ) -> np.ndarray:
        return routing_result.outflow

    def apply_penalty(
        self,
        objective_value: np.ndarray,
        penalty: np.ndarray,
    ) -> np.ndarray:
        scale = np.maximum(np.abs(objective_value).max(), 1.0)
        return objective_value + scale * penalty
