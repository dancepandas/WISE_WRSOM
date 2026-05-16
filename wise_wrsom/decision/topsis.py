"""TOPSIS 决策排序方法。"""
from __future__ import annotations

import numpy as np

from ..protocols.decision import DecisionModel


class TOPSISDecision(DecisionModel):
    """基于 TOPSIS 的多属性决策排序。

    支持 max/min 方向目标：min 方向目标在确定正负理想解时反转。
    """

    def rank(
        self,
        objective_values: np.ndarray,
        subjective_weights: list[float] | None = None,
        directions: list[str] | None = None,
    ) -> np.ndarray:
        """对 Pareto 方案进行 TOPSIS 排序。

        Args:
            objective_values: shape=(n_solutions, n_objectives)，原始目标值。
            subjective_weights: 主观权重，长度等于目标数。
            directions: 每个目标的方向 "max" 或 "min"，长度等于目标数。

        Returns:
            贴近度系数数组，值越大越优。
        """
        n_obj = objective_values.shape[1]
        if directions is None:
            directions = ["max"] * n_obj
        if subjective_weights is None:
            subjective_weights = [1.0 / n_obj] * n_obj

        weights = np.array(subjective_weights)

        # 标准化（向量归一化）
        norm = np.sqrt(np.sum(objective_values ** 2, axis=0))
        norm = np.where(norm == 0, 1.0, norm)
        normalized = objective_values / norm

        # 加权标准化矩阵
        weighted = normalized * weights

        # 确定正负理想解（考虑方向）
        ideal_best = np.zeros(n_obj)
        ideal_worst = np.zeros(n_obj)
        for j in range(n_obj):
            if directions[j] == "max":
                ideal_best[j] = weighted[:, j].max()
                ideal_worst[j] = weighted[:, j].min()
            else:  # min
                ideal_best[j] = weighted[:, j].min()
                ideal_worst[j] = weighted[:, j].max()

        # 计算距离
        dist_best = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
        dist_worst = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))

        # 贴近度系数
        denominator = dist_best + dist_worst
        denominator = np.where(denominator == 0, 1.0, denominator)
        closeness = dist_worst / denominator

        return closeness
