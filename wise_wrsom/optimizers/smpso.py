"""改进的 SMPSO 优化算法。"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ..protocols.objective import ObjectiveFunction
from ..protocols.optimizer import OptimizationResult
from ..protocols.routing import RoutingModel, RiverParams
from .base import (
    ParetoArchive,
    PopulationInitializer,
    crowding_distance,
    evaluate_objectives,
)
from .registry import register
from .segment import SegmentPopulation, segment_mutation


@register("smpso")
class SMPSOOptimizer:
    """改进的 SMPSO 多目标粒子群优化算法。"""

    def __init__(
        self,
        velocity_max: float = 5.0,
        velocity_min: float = -5.0,
        mutation_rate: float = 0.1,
        mutation_eta: float = 20.0,
    ):
        self.velocity_max = velocity_max
        self.velocity_min = velocity_min
        self.mutation_rate = mutation_rate
        self.mutation_eta = mutation_eta

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
        initializer = PopulationInitializer(total_water, scheduling_days, flow_min, flow_max)
        archive = ParetoArchive()

        # 只初始化一次种群
        seg_pop = initializer.initialize_segments(population_size)
        velocity = self._create_velocity(seg_pop)
        obj_values = evaluate_objectives(
            objectives, routing_model, river_params, seg_pop.expand(), total_water
        )
        archive.update(obj_values, seg_pop.expand())

        w_start, w_end = 0.5, 0.1
        total_iters = max_iterations
        for inner_iter in tqdm(range(total_iters), desc="SMPSO"):
            w = w_start - (w_start - w_end) * inner_iter / max(total_iters - 1, 1)
            velocity = self._update_velocity(seg_pop, velocity, archive, w)
            self._update_population(seg_pop, velocity, flow_min, flow_max)
            for i in range(len(seg_pop)):
                seg_pop[i] = segment_mutation(seg_pop[i], self.mutation_rate, self.mutation_eta)
            obj_values = evaluate_objectives(
                objectives, routing_model, river_params, seg_pop.expand(), total_water
            )
            archive.update(obj_values, seg_pop.expand())

        archive = self._filter_pareto_front(archive)
        return archive.get_result()

    def _create_velocity(self, seg_pop: SegmentPopulation) -> list[np.ndarray]:
        """为每个粒子创建段级速度向量。"""
        velocity = []
        for sol in seg_pop.solutions:
            v = np.zeros(sol.n_segments)
            n_active = sol.n_segments - 1
            if n_active > 0:
                v[:n_active] = np.random.uniform(self.velocity_min, self.velocity_max, n_active)
                v[-1] = -np.dot(v[:n_active], sol.day_splits[:n_active]) / sol.day_splits[-1]
            velocity.append(np.clip(v, self.velocity_min, self.velocity_max))
        return velocity

    def _update_velocity(
        self, seg_pop: SegmentPopulation, velocity: list[np.ndarray],
        archive: ParetoArchive, w: float,
    ) -> list[np.ndarray]:
        if archive.size == 0:
            return velocity

        archive_pop = np.array(archive.population)
        crowd = crowding_distance(np.array(archive.objectives))
        leader_idx = self._roulette_select(crowd, len(seg_pop))

        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        c1 = np.random.uniform(1.5, 2.5)
        c2 = np.random.uniform(1.5, 2.5)
        phi = c1 + c2 if (c1 + c2) > 4 else 0
        chi = 2.0 / abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi)) if phi > 0 else 1.0
        delta = (self.velocity_max - self.velocity_min) / 2

        new_velocity = []
        for i in range(len(seg_pop)):
            sol = seg_pop[i]
            expanded = sol.expand()
            leader_expanded = archive_pop[leader_idx[i]]
            leader_delta = leader_expanded - expanded

            seg_delta = np.zeros(sol.n_segments)
            idx = 0
            for j in range(sol.n_segments):
                seg_delta[j] = np.mean(leader_delta[idx:idx + sol.day_splits[j]])
                idx += sol.day_splits[j]

            cognitive = np.random.uniform(0, 1, sol.n_segments) * seg_delta * 0.1
            v = (w * velocity[i] + c2 * r2 * seg_delta + c1 * r1 * cognitive) * chi
            v = np.clip(v, -delta, delta)
            # 零和约束：确保 sum(v * day_splits) = 0，保持总水量不变
            v_weighted_sum = np.sum(v * sol.day_splits)
            total_days = sol.day_splits.sum()
            v -= v_weighted_sum / total_days
            v = np.clip(v, self.velocity_min, self.velocity_max)
            # clip 后再次修正零和
            v_weighted_sum2 = np.sum(v * sol.day_splits)
            v -= v_weighted_sum2 / total_days
            new_velocity.append(v)

        return new_velocity

    @staticmethod
    def _update_population(
        seg_pop: SegmentPopulation,
        velocity: list[np.ndarray],
        flow_min: float,
        flow_max: float,
    ) -> None:
        """直接对段级 flow_rates 施加速度。"""
        for i, sol in enumerate(seg_pop.solutions):
            sol.flow_rates = np.clip(sol.flow_rates + velocity[i], flow_min, flow_max)
            sol.normalize()

    @staticmethod
    def _roulette_select(weights: np.ndarray, n_select: int) -> np.ndarray:
        if len(weights) == 0:
            return np.zeros(n_select, dtype=int)
        finite_mask = np.isfinite(weights)
        if not np.any(finite_mask):
            return np.random.randint(0, len(weights), n_select)
        w = weights.copy()
        w[~finite_mask] = w[finite_mask].max() * 2 if np.any(finite_mask) else 1.0
        w = w - w.min() + 1e-6
        return np.random.choice(len(weights), size=n_select, replace=True, p=w / w.sum())

    @staticmethod
    def _filter_pareto_front(archive: ParetoArchive) -> ParetoArchive:
        """严格非支配排序，过滤出真正的 Pareto 前沿。"""
        objectives = np.array(archive.objectives)
        n = len(objectives)
        if n <= 1:
            return archive

        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                    is_dominated[i] = True
                    break

        filtered = ParetoArchive()
        filtered.objectives = [archive.objectives[i] for i in range(n) if not is_dominated[i]]
        filtered.population = [archive.population[i] for i in range(n) if not is_dominated[i]]
        return filtered