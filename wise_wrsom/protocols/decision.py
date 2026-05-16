"""决策方法协议定义。

任何决策方法（TOPSIS、VIKOR、PROMETHEE 等）只需实现 DecisionModel 协议即可接入系统。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DecisionModel(Protocol):
    """多准则决策方法协议。

    实现此协议的类可用于替换决策方法模块，例如：
    - TOPSISDecision（TOPSIS，当前默认）
    - VIKORDecision
    - PROMETHEEDecision
    - 自定义决策方法...

    只需保证 rank() 方法的输入输出符合约定即可。
    """

    def rank(
        self,
        objective_values: np.ndarray,
        subjective_weights: list[float] | None = None,
        directions: list[str] | None = None,
    ) -> np.ndarray:
        """对 Pareto 最优解进行排序。

        Args:
            objective_values: 目标值矩阵，shape=(n_solutions, n_objectives)
            subjective_weights: 决策者主观权重，None 则使用纯客观权重
            directions: 各目标优化方向列表，"max" 或 "min"

        Returns:
            np.ndarray: 各方案的贴近度系数，shape=(n_solutions,)，值越大越优
        """
        ...
