"""目标函数共用的罚函数。"""
from __future__ import annotations

import numpy as np

from ..protocols.routing import RoutingResult


def compute_penalty(
    routing_result: RoutingResult,
    population: np.ndarray,
    total_water: float,
) -> np.ndarray:
    """计算水量平衡约束罚函数值。

    使用分段线性+二次罚函数：
    - 偏差在 tolerance 内：无惩罚
    - 偏差超出 tolerance：二次增长惩罚

    返回无量纲罚函数值，由各目标函数按自身量级缩放。

    population 单位为 m³/s，shape=(n_population, n_days)。
    总水量 = sum(q_i) * 24 * 60 * 60 (m³)，与 total_water (m³) 比较。
    """
    downstream_flow = np.array(routing_result.downstream_flows[0])
    actual_water = np.sum(downstream_flow, axis=1) * 24 * 60 * 60
    relative_violation = (actual_water - total_water) / total_water
    tolerance = 0.10
    excess = np.maximum(np.abs(relative_violation) - tolerance, 0.0)
    return np.square(excess) * 500.0
