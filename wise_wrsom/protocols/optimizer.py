"""优化算法协议定义。

任何优化算法（SMPSO、NSGA-III、MOEA/D 等）只需实现 Optimizer 协议即可接入系统。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .objective import ObjectiveFunction
from .routing import RoutingModel, RiverParams


@dataclass
class OptimizationResult:
    """优化结果。

    Attributes:
        pareto_objectives: Pareto 最优解的目标值矩阵，shape=(n_solutions, n_objectives)
        pareto_population: Pareto 最优解的决策变量矩阵，shape=(n_solutions, n_days)
    """
    pareto_objectives: np.ndarray
    pareto_population: np.ndarray


@runtime_checkable
class Optimizer(Protocol):
    """优化算法协议。

    实现此协议的类可用于替换优化算法模块，例如：
    - SMPSOOptimizer（改进的 SMPSO）
    - NSGA3Optimizer（NSGA-III）
    - MOEADOptimizer（MOEA/D）
    - 自定义优化器...

    只需保证 optimize() 方法的输入输出符合约定即可。
    """

    def optimize(
        self,
        objectives: list[ObjectiveFunction],
        routing_model: RoutingModel,
        river_params: RiverParams,
        total_water: float,
        population_size: int,
        scheduling_days: int,
        max_iterations: int,
        flow_min: float,
        flow_max: float,
    ) -> OptimizationResult:
        """执行多目标优化。

        Args:
            objectives: 目标函数列表
            routing_model: 河道流量计算模型
            river_params: 河道参数
            total_water: 总调度水量（m³）
            population_size: 种群大小
            scheduling_days: 调度天数
            max_iterations: 最大迭代次数
            flow_min: 最小流量边界（m³/s）
            flow_max: 最大流量边界（m³/s）

        Returns:
            OptimizationResult: Pareto 最优解集
        """
        ...
