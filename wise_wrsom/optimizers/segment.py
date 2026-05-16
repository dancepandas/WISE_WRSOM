"""分段常数编码模块。

将调度方案表示为 (day_splits, flow_rates) 紧凑参数，
遗传算子直接在分段空间操作，天然保持阶梯函数结构。
"""
from __future__ import annotations

import numpy as np


class SegmentSolution:
    """单个分段调度方案。

    用 day_splits（各段天数）和 flow_rates（各段流量）表示一个阶梯型调度方案。
    调用 expand() 可展开为完整的逐日流量向量。
    """

    def __init__(
        self,
        day_splits: np.ndarray,
        flow_rates: np.ndarray,
        scheduling_days: int,
        total_water: float,
        flow_min: float,
        flow_max: float,
    ):
        self.day_splits = np.asarray(day_splits, dtype=int)
        self.flow_rates = np.asarray(flow_rates, dtype=float)
        self.scheduling_days = scheduling_days
        self.total_water = total_water
        self.flow_min = flow_min
        self.flow_max = flow_max

    @property
    def n_segments(self) -> int:
        return len(self.day_splits)

    def expand(self) -> np.ndarray:
        """展开为逐日流量向量。"""
        return np.concatenate([
            np.full(d, q) for d, q in zip(self.day_splits, self.flow_rates)
        ])

    def normalize(self) -> None:
        """缩放 flow_rates 使水量接近目标，允许自然偏差。

        等比缩放到目标水量附近（±5%随机偏差），clip 后迭代修正
        直到偏差在 ±10% 内。保留自然偏差让罚函数和水量平衡目标区分。
        """
        water_per_day = self.flow_rates * self.day_splits
        current_water = water_per_day.sum() * 24 * 60 * 60
        if current_water <= 0:
            return
        target = self.total_water
        # 缩放到目标附近，允许±5%随机偏差
        target_with_tolerance = target * (1.0 + np.random.uniform(-0.05, 0.05))
        scale = target_with_tolerance / current_water
        self.flow_rates *= scale
        self.flow_rates = np.clip(self.flow_rates, self.flow_min, self.flow_max)

        # 修正 clip 导致的水量偏差：迭代重分配，偏差在 ±10% 内停止
        for _ in range(100):
            current = (self.flow_rates * self.day_splits).sum() * 24 * 60 * 60
            diff = target - current
            if abs(diff / target) < 0.10:
                break
            headroom_up = self.flow_max - self.flow_rates
            headroom_down = self.flow_rates - self.flow_min
            if diff > 0:
                weights = headroom_up * self.day_splits
                total_w = weights.sum()
                if total_w < 1e-10:
                    break
                self.flow_rates += diff / (24 * 60 * 60) * (weights / total_w)
            else:
                weights = headroom_down * self.day_splits
                total_w = weights.sum()
                if total_w < 1e-10:
                    break
                self.flow_rates -= abs(diff) / (24 * 60 * 60) * (weights / total_w)
            self.flow_rates = np.clip(self.flow_rates, self.flow_min, self.flow_max)

    def copy(self) -> SegmentSolution:
        return SegmentSolution(
            self.day_splits.copy(),
            self.flow_rates.copy(),
            self.scheduling_days,
            self.total_water,
            self.flow_min,
            self.flow_max,
        )

    @classmethod
    def from_expanded(
        cls,
        expanded: np.ndarray,
        scheduling_days: int,
        total_water: float,
        flow_min: float,
        flow_max: float,
    ) -> SegmentSolution:
        """从展开向量重建分段结构。"""
        changes = np.where(np.diff(expanded) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes, [len(expanded)]])
        day_splits = np.diff(boundaries)
        flow_rates = np.array([
            expanded[boundaries[i]] for i in range(len(day_splits))
        ])
        return cls(day_splits, flow_rates, scheduling_days, total_water, flow_min, flow_max)


class SegmentPopulation:
    """分段方案种群容器。"""

    def __init__(
        self,
        solutions: list[SegmentSolution],
        scheduling_days: int,
        total_water: float,
        flow_min: float,
        flow_max: float,
    ):
        self.solutions = solutions
        self.scheduling_days = scheduling_days
        self.total_water = total_water
        self.flow_min = flow_min
        self.flow_max = flow_max

    def __len__(self) -> int:
        return len(self.solutions)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            return SegmentPopulation(
                [self.solutions[i] for i in idx],
                self.scheduling_days,
                self.total_water,
                self.flow_min,
                self.flow_max,
            )
        return self.solutions[idx]

    def __setitem__(self, idx, value: SegmentSolution):
        self.solutions[idx] = value

    def expand(self) -> np.ndarray:
        """展开为 (n, scheduling_days) 数组。"""
        return np.array([sol.expand() for sol in self.solutions])


def segment_crossover(
    p1: SegmentSolution,
    p2: SegmentSolution,
    eta: float = 20.0,
) -> tuple[SegmentSolution, SegmentSolution]:
    """分段交叉算子。对匹配段做 SBX 混合 flow_rates，交叉后零和调整。"""
    n = min(p1.n_segments, p2.n_segments)

    c1 = p1.copy()
    c2 = p2.copy()

    for i in range(n):
        if np.random.random() < 0.5:
            u = np.random.random()
            beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 else (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            c1.flow_rates[i] = 0.5 * ((1 + beta) * p1.flow_rates[i] + (1 - beta) * p2.flow_rates[i])
            c2.flow_rates[i] = 0.5 * ((1 - beta) * p1.flow_rates[i] + (1 + beta) * p2.flow_rates[i])

    c1.normalize()
    c2.normalize()
    return c1, c2


def segment_mutation(
    solution: SegmentSolution,
    rate: float = 0.1,
    eta: float = 20.0,
) -> SegmentSolution:
    """分段变异算子。支持流量变异、段分裂、段合并。"""
    sol = solution.copy()
    r = np.random.random()

    if r < 0.70:
        _mutate_flow_rates(sol, rate, eta)
    elif r < 0.85:
        sol = _split_segment(sol, eta)
    else:
        sol = _merge_segments(sol)

    sol.normalize()
    return sol


def _mutate_flow_rates(sol: SegmentSolution, rate: float, eta: float) -> None:
    """零和变异：变异某些段的流量，同时反向调整其他段以保持总水量不变。"""
    mask = np.random.random(sol.n_segments) < rate
    if not np.any(mask):
        return
    u = np.random.random(sol.n_segments)
    delta_max = sol.flow_max - sol.flow_min
    delta = np.where(
        u < 0.5,
        (2 * u) ** (1.0 / (eta + 1)) - 1,
        1 - (2 * (1 - u)) ** (1.0 / (eta + 1)),
    )
    # 计算变异段的流量变化
    delta_q = delta * delta_max
    delta_q[~mask] = 0.0
    # 水量变化 = sum(delta_q * day_splits) * 86400
    water_change = np.sum(delta_q * sol.day_splits) * 24 * 60 * 60
    # 将水量变化反向分配到未变异段
    unmasked = ~mask
    if np.any(unmasked) and abs(water_change) > 1e-6:
        # 按天数加权分配补偿
        weights = sol.day_splits[unmasked]
        total_w = weights.sum()
        if total_w > 0:
            # 每段补偿的流量 = -water_change / (total_w * 86400) * (day_split_i / total_w)
            # 即补偿后 sum(compensation * day_splits_unmasked) * 86400 = -water_change
            per_unit = -water_change / (24 * 60 * 60 * total_w)
            sol.flow_rates[unmasked] += per_unit
    sol.flow_rates[mask] += delta_q[mask]
    sol.flow_rates = np.clip(sol.flow_rates, sol.flow_min, sol.flow_max)


def _split_segment(sol: SegmentSolution, eta: float) -> SegmentSolution:
    """将一个段分裂为两段。"""
    eligible = np.where(sol.day_splits >= 10)[0]
    if len(eligible) == 0:
        return sol

    idx = np.random.choice(eligible)
    d = sol.day_splits[idx]
    split_point = np.random.randint(3, d - 2)
    d1, d2 = split_point, d - split_point

    q = sol.flow_rates[idx]
    mutation_delta = (np.random.random() - 0.5) * 0.1 * (sol.flow_max - sol.flow_min)
    q1 = np.clip(q + mutation_delta, sol.flow_min, sol.flow_max)
    q2 = np.clip(q - mutation_delta, sol.flow_min, sol.flow_max)

    new_days = np.concatenate([sol.day_splits[:idx], [d1, d2], sol.day_splits[idx + 1:]])
    new_flows = np.concatenate([sol.flow_rates[:idx], [q1, q2], sol.flow_rates[idx + 1:]])

    return SegmentSolution(
        new_days, new_flows, sol.scheduling_days, sol.total_water, sol.flow_min, sol.flow_max
    )


def _merge_segments(sol: SegmentSolution) -> SegmentSolution:
    """合并两个相邻段。"""
    if sol.n_segments <= 2:
        return sol

    idx = np.random.randint(0, sol.n_segments - 1)
    d1, d2 = sol.day_splits[idx], sol.day_splits[idx + 1]
    q1, q2 = sol.flow_rates[idx], sol.flow_rates[idx + 1]

    merged_days = d1 + d2
    merged_flow = (q1 * d1 + q2 * d2) / merged_days

    new_days = np.concatenate([sol.day_splits[:idx], [merged_days], sol.day_splits[idx + 2:]])
    new_flows = np.concatenate([sol.flow_rates[:idx], [merged_flow], sol.flow_rates[idx + 2:]])

    return SegmentSolution(
        new_days, new_flows, sol.scheduling_days, sol.total_water, sol.flow_min, sol.flow_max
    )
