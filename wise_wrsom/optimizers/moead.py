"""MOEA/D 优化算法。"""
from __future__ import annotations

from math import comb

import numpy as np
from tqdm import tqdm

from ..protocols.objective import ObjectiveFunction
from ..protocols.optimizer import OptimizationResult
from ..protocols.routing import RoutingModel, RiverParams
from .base import (
    ParetoArchive,
    PopulationInitializer,
    das_dennis_reference_points,
    evaluate_objectives,
)
from .registry import register
from .segment import segment_crossover, segment_mutation


@register("moead")
class MOEADOptimizer:
    """MOEA/D 多目标优化算法。"""

    def __init__(
        self,
        n_neighbors: int = 10,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        mutation_eta: float = 20.0,
    ):
        self.n_neighbors = n_neighbors
        self.crossover_rate = crossover_rate
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
        n_objectives = len(objectives)
        initializer = PopulationInitializer(total_water, scheduling_days, flow_min, flow_max)

        weight_vectors = self._generate_weight_vectors(n_objectives, population_size)
        neighbors = self._compute_neighbors(weight_vectors, self.n_neighbors)

        seg_pop = initializer.initialize_segments(population_size)
        obj_values = evaluate_objectives(
            objectives, routing_model, river_params, seg_pop.expand(), total_water
        )
        ideal_point = obj_values.min(axis=0)

        for gen in tqdm(range(max_iterations), desc="MOEA/D"):
            for i in range(population_size):
                neighbor_idx = neighbors[i]
                p1_idx, p2_idx = np.random.choice(neighbor_idx, 2, replace=False)

                child = self._create_child(
                    seg_pop[p1_idx], seg_pop[p2_idx], flow_min, flow_max
                )
                child_obj = evaluate_objectives(
                    objectives, routing_model, river_params,
                    child.expand().reshape(1, -1), total_water
                )[0]

                ideal_point = np.minimum(ideal_point, child_obj)

                for j in neighbor_idx:
                    if self._is_better(child_obj, obj_values[j], weight_vectors[j], ideal_point):
                        seg_pop[j] = child.copy()
                        obj_values[j] = child_obj

        archive = ParetoArchive()
        archive.update(obj_values, seg_pop.expand())
        # MOEA/D 种群中大部分解在不同权重方向上最优，
        # 5目标时几乎全部非支配。做一次严格过滤得到真正的 Pareto 前沿。
        archive = self._filter_pareto_front(archive)
        return archive.get_result()

    def _generate_weight_vectors(
        self, n_objectives: int, n_vectors: int
    ) -> np.ndarray:
        if n_objectives == 2:
            return np.array([[i / (n_vectors - 1), 1 - i / (n_vectors - 1)]
                             for i in range(n_vectors)])

        h = 1
        while comb(n_objectives + h - 1, h) < n_vectors:
            h += 1
        vectors = das_dennis_reference_points(n_objectives, h)

        while len(vectors) < n_vectors:
            v = np.random.dirichlet(np.ones(n_objectives))
            vectors = np.vstack([vectors, v.reshape(1, -1)])
        return vectors[:n_vectors]

    def _compute_neighbors(
        self, weight_vectors: np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        n = len(weight_vectors)
        n_neighbors = min(n_neighbors, n)
        diff = weight_vectors[:, None, :] - weight_vectors[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return np.argsort(distances, axis=1)[:, :n_neighbors]

    def _create_child(self, p1, p2, flow_min: float, flow_max: float):
        if np.random.random() < self.crossover_rate:
            c1, c2 = segment_crossover(p1, p2, 20.0)
            child = c1
        else:
            child = p1.copy()
        child = segment_mutation(child, self.mutation_rate, self.mutation_eta)
        child.normalize()
        return child

    @staticmethod
    def _is_better(
        new_obj: np.ndarray, old_obj: np.ndarray, weight: np.ndarray, ideal_point: np.ndarray
    ) -> bool:
        return np.max(weight * np.abs(new_obj - ideal_point)) < np.max(weight * np.abs(old_obj - ideal_point))

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
                # j 支配 i: j 在所有目标上 >= i，且至少一个 > i
                if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                    is_dominated[i] = True
                    break

        filtered = ParetoArchive()
        filtered.objectives = [archive.objectives[i] for i in range(n) if not is_dominated[i]]
        filtered.population = [archive.population[i] for i in range(n) if not is_dominated[i]]
        return filtered
