"""优化算法公共组件。

提供种群初始化、Pareto 归档、拥挤度、遗传算子等公共逻辑。
"""
from __future__ import annotations

import itertools
import json
from math import comb
from pathlib import Path

import numpy as np

from ..protocols.objective import ObjectiveFunction
from ..protocols.optimizer import OptimizationResult
from ..protocols.routing import RoutingModel, RiverParams, RoutingResult


class PopulationInitializer:
    """约束满足的种群初始化器。"""

    def __init__(
        self,
        total_water: float,
        scheduling_days: int,
        flow_min: float,
        flow_max: float,
    ):
        self.total_water = total_water
        self.scheduling_days = scheduling_days
        self.flow_min = flow_min
        self.flow_max = flow_max

    def initialize(self, population_size: int) -> tuple[np.ndarray, np.ndarray]:
        """生成满足约束的初始种群。"""
        while True:
            n_segments = np.random.randint(3, 6)
            v1 = np.random.dirichlet(
                [np.random.uniform(1.2, 2.2) for _ in range(n_segments)],
                population_size,
            )
            number_of_days = np.round(self.scheduling_days * v1).astype(int)
            number_of_days = np.where(number_of_days == 0, 5, number_of_days)

            # 修正天数总和
            value_day = np.sum(number_of_days, axis=1) - self.scheduling_days
            for i in range(population_size):
                if value_day[i] < 0:
                    min_idx = np.argmin(number_of_days[i])
                    number_of_days[i, min_idx] += -value_day[i]
                elif value_day[i] > 0:
                    max_idx = np.argmax(number_of_days[i])
                    number_of_days[i, max_idx] -= value_day[i]

            # 用浮点数分配水量，避免 int 截断累积误差
            w_split = self.total_water * v1

            q_np = w_split / (24 * 60 * 60 * number_of_days)
            population = self._build_population(q_np, number_of_days)

            if np.all(population <= self.flow_max):
                return population, number_of_days

    def _build_population(
        self, q_np: np.ndarray, number_of_days: np.ndarray
    ) -> np.ndarray:
        """从分段流量和天数构建完整调度方案。"""
        population = []
        for j in range(q_np.shape[0]):
            segments = [np.full(number_of_days[j, n], q_np[j, n]) for n in range(q_np.shape[1])]
            population.append(np.concatenate(segments))
        return np.array(population)

    def initialize_segments(self, population_size: int):
        """生成分段编码的初始种群。返回 SegmentPopulation。"""
        from .segment import SegmentPopulation, SegmentSolution

        while True:
            n_segments = np.random.randint(3, 6)
            v1 = np.random.dirichlet(
                [np.random.uniform(1.2, 2.2) for _ in range(n_segments)],
                population_size,
            )
            number_of_days = np.round(self.scheduling_days * v1).astype(int)
            number_of_days = np.where(number_of_days == 0, 5, number_of_days)

            # 修正天数总和
            day_diff = np.sum(number_of_days, axis=1) - self.scheduling_days
            for i in range(population_size):
                if day_diff[i] < 0:
                    min_idx = np.argmin(number_of_days[i])
                    number_of_days[i, min_idx] += -day_diff[i]
                elif day_diff[i] > 0:
                    max_idx = np.argmax(number_of_days[i])
                    number_of_days[i, max_idx] -= day_diff[i]

            # 用浮点数分配水量，避免 int 截断累积误差
            w_split = self.total_water * v1

            q_np = w_split / (24 * 60 * 60 * number_of_days)

            if np.all(q_np <= self.flow_max):
                solutions = [
                    SegmentSolution(
                        number_of_days[i], q_np[i],
                        self.scheduling_days, self.total_water,
                        self.flow_min, self.flow_max,
                    )
                    for i in range(population_size)
                ]
                seg_pop = SegmentPopulation(
                    solutions, self.scheduling_days,
                    self.total_water, self.flow_min, self.flow_max,
                )
                # 确保每个方案精确满足水量约束
                for sol in seg_pop.solutions:
                    sol.normalize()
                return seg_pop


class ParetoArchive:
    """Pareto 归档管理器。"""

    def __init__(self):
        self.objectives: list[list[float]] = []
        self.population: list[list[float]] = []

    @property
    def size(self) -> int:
        return len(self.objectives)

    def update(
        self, new_objectives: np.ndarray, new_population: np.ndarray
    ) -> None:
        """用新的解更新归档。"""
        if self.size == 0:
            for i in range(len(new_objectives)):
                self.objectives.append(list(new_objectives[i]))
                self.population.append(list(new_population[i]))
            return

        archive_np = np.array(self.objectives)
        existing_tuples = {tuple(row) for row in archive_np}

        for k in range(len(new_objectives)):
            b = new_objectives[k]
            b_tuple = tuple(b)

            if b_tuple in existing_tuples:
                continue

            dominated = np.any(
                np.all(archive_np >= b, axis=1) & np.any(archive_np > b, axis=1)
            )
            if dominated:
                continue

            dominates_mask = np.all(b >= archive_np, axis=1) & np.any(b > archive_np, axis=1)
            to_keep = ~dominates_mask
            archive_np = archive_np[to_keep]
            self.population = [p for p, keep in zip(self.population, to_keep) if keep]
            existing_tuples = {tuple(row) for row in archive_np}

            archive_np = np.vstack([archive_np, b]) if archive_np.size else np.array([b])
            self.population.append(list(new_population[k]))
            existing_tuples.add(b_tuple)

        self.objectives[:] = archive_np.tolist()

    def merge(self, other: ParetoArchive) -> None:
        if other.size == 0:
            return
        self.update(np.array(other.objectives), np.array(other.population))

    def get_result(self) -> OptimizationResult:
        return OptimizationResult(
            pareto_objectives=np.array(self.objectives),
            pareto_population=np.array(self.population),
        )

    def save(self, path: str) -> None:
        data = {
            "archive_total": self.objectives,
            "archive_population_total": self.population,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """计算拥挤度距离。"""
    infinity = 1e11
    n = len(objectives)
    if n <= 2:
        return np.full(n, infinity)

    crowd = np.zeros(n, dtype=float)
    for obj_idx in range(objectives.shape[1]):
        order = np.argsort(objectives[:, obj_idx])
        sorted_vals = objectives[order, obj_idx]
        denom = max(sorted_vals[-1] - sorted_vals[0], 1e-6)

        dist = np.zeros(n)
        dist[order[0]] = infinity
        dist[order[-1]] = infinity
        dist[order[1:-1]] = (sorted_vals[2:] - sorted_vals[:-2]) / denom

        finite_mask = ~np.isinf(dist)
        crowd[finite_mask] += dist[finite_mask]
        crowd[~finite_mask] = infinity

    return crowd


def polynomial_mutation(
    population: np.ndarray,
    rate: float,
    eta: float,
    flow_min: float,
    flow_max: float,
) -> np.ndarray:
    """多项式变异算子。"""
    result = population.copy()
    mask = np.random.random(population.shape) < rate
    u = np.random.random(population.shape)
    delta_max = flow_max - flow_min
    delta = np.where(
        u < 0.5,
        (2 * u) ** (1.0 / (eta + 1)) - 1,
        1 - (2 * (1 - u)) ** (1.0 / (eta + 1)),
    )
    result[mask] = population[mask] + delta[mask] * delta_max
    return np.clip(result, flow_min, flow_max)


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    eta: float,
    flow_min: float,
    flow_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉（SBX）。"""
    c1 = p1.copy()
    c2 = p2.copy()
    mask = np.random.random(len(p1)) < 0.5
    u = np.random.random(len(p1))

    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1)),
        (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1)),
    )

    c1[mask] = 0.5 * ((1 + beta[mask]) * p1[mask] + (1 - beta[mask]) * p2[mask])
    c2[mask] = 0.5 * ((1 - beta[mask]) * p1[mask] + (1 + beta[mask]) * p2[mask])

    return np.clip(c1, flow_min, flow_max), np.clip(c2, flow_min, flow_max)


def das_dennis_reference_points(n_objectives: int, n_divisions: int) -> np.ndarray:
    """Das-Dennis 均匀参考点生成。"""
    def _generate(objectives_left: int, divisions_left: int, current: list[int]):
        if objectives_left == 1:
            yield current + [divisions_left]
            return
        for i in range(divisions_left + 1):
            yield from _generate(objectives_left - 1, divisions_left - i, current + [i])

    points = []
    for combo in _generate(n_objectives, n_divisions, []):
        points.append([x / n_divisions for x in combo])
    return np.array(points)


def evaluate_objectives(
    objectives: list[ObjectiveFunction],
    routing_model: RoutingModel,
    river_params: RiverParams,
    population: np.ndarray,
    total_water: float,
) -> np.ndarray:
    """评估种群的目标函数值。

    先计算纯目标值，再统一计算水量平衡罚函数，
    最后通过各目标的 apply_penalty 按自身量级缩放罚函数值。

    min 方向的目标取负值，统一为最大化问题，便于 Pareto 支配比较。
    """
    routing_result = routing_model.compute(population, river_params)
    penalty = compute_water_balance_penalty(routing_result, total_water)
    obj_values = np.zeros((len(population), len(objectives)))
    for i, obj in enumerate(objectives):
        raw_value = obj.compute(routing_result, population, total_water)
        penalized = obj.apply_penalty(raw_value, penalty)
        if obj.direction == "min":
            penalized = -penalized
        obj_values[:, i] = penalized
    return obj_values


def compute_water_balance_penalty(
    routing_result: RoutingResult,
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
