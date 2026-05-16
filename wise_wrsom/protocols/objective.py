"""调度目标协议定义。

任何调度目标只需实现 ObjectiveFunction 协议即可接入优化系统。
水量平衡约束罚函数由协议层统一处理，按各目标量级自动缩放。
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .routing import RoutingResult


@runtime_checkable
class ObjectiveFunction(Protocol):
    """调度目标函数协议。

    实现此协议的类可用于定义优化目标，例如：
    - WaterDurationObjective（全线通水时长）
    - GroundwaterRechargeObjective（地下水回补）
    - SurfaceAreaObjective（水面面积）
    - OutflowControlObjective（出境水量控制）
    - 自定义目标...

    新增目标只需：
    1. 实现此协议
    2. 通过 @register("name") 装饰器注册
    3. 在配置文件中启用

    水量平衡约束罚函数由协议层统一管理：
    - compute() 只计算纯目标值，不含罚函数
    - apply_penalty() 按目标量级缩放罚函数值
    """

    @property
    def name(self) -> str:
        """目标名称。"""
        ...

    @property
    def direction(self) -> Literal["max", "min"]:
        """优化方向：'max' 最大化，'min' 最小化。"""
        ...

    def compute(
        self,
        routing_result: RoutingResult,
        population: np.ndarray,
        total_water: float,
    ) -> np.ndarray:
        """计算纯目标函数值（不含水量平衡罚函数）。

        Args:
            routing_result: 河道流量计算结果
            population: 调度方案矩阵，shape=(n_population, n_days)
            total_water: 总调度水量（m³）

        Returns:
            np.ndarray: 纯目标函数值，shape=(n_population,)
        """
        ...

    def apply_penalty(
        self,
        objective_value: np.ndarray,
        penalty: np.ndarray,
    ) -> np.ndarray:
        """按目标量级缩放罚函数值并合并。

        加法罚函数：objective_value - scale * penalty
        penalty 为无量纲的水量平衡相对偏差平方值，
        scale 取目标值绝对最大值，确保罚函数与目标值同量级。
        偏差越大，惩罚越重，且不会出现乘法罚函数中大偏差时目标归零的问题。

        对于 min 方向的目标，penalty 应为加法（让值更大，取负后更小）。

        Args:
            objective_value: 纯目标函数值，shape=(n_population,)
            penalty: 无量纲罚函数值，shape=(n_population,)

        Returns:
            np.ndarray: 含罚函数的目标值，shape=(n_population,)
        """
        ...
