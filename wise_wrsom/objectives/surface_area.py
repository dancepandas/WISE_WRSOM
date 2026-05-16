"""水面面积目标函数。"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..protocols.routing import RoutingResult
from .registry import register


@register("surface_area")
class SurfaceAreaObjective:
    """水面面积目标：最大化河道水面面积，改善生态环境。"""

    @property
    def name(self) -> str:
        return "水面面积"

    @property
    def direction(self) -> Literal["max", "min"]:
        return "max"

    def compute(
        self,
        routing_result: RoutingResult,
        population: np.ndarray,
        total_water: float,
    ) -> np.ndarray:
        return routing_result.surface_area

    def apply_penalty(
        self,
        objective_value: np.ndarray,
        penalty: np.ndarray,
        scale: float | None = None,
    ) -> np.ndarray:
        scale = np.maximum(np.abs(objective_value).max(), 1.0)
        return objective_value - scale * penalty
