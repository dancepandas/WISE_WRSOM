"""NSGA-III 优化算法。"""
from __future__ import annotations

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
from .segment import SegmentPopulation, segment_crossover, segment_mutation


@register("nsga3")
class NSGA3Optimizer:
    """NSGA-III 多目标优化算法。"""

    def __init__(
        self,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        mutation_eta: float = 20.0,
        crossover_eta: float = 20.0,
        n_reference_divisions: int = 12,
    ):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_eta = mutation_eta
        self.crossover_eta = crossover_eta
        self.n_reference_divisions = n_reference_divisions

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
        reference_points = das_dennis_reference_points(n_objectives, self.n_reference_divisions)

        seg_pop = initializer.initialize_segments(population_size)
        obj_values = evaluate_objectives(
            objectives, routing_model, river_params, seg_pop.expand(), total_water
        )

        for gen in tqdm(range(max_iterations), desc="NSGA-III"):
            offspring = self._create_offspring(seg_pop)
            offspring_obj = evaluate_objectives(
                objectives, routing_model, river_params, offspring.expand(), total_water
            )

            combined_pop = self._combine(seg_pop, offspring)
            combined_obj = np.vstack([obj_values, offspring_obj])

            selected_idx = self._environmental_selection(
                combined_obj, population_size, reference_points
            )
            seg_pop = combined_pop[selected_idx]
            obj_values = combined_obj[selected_idx]

        archive = ParetoArchive()
        archive.update(obj_values, seg_pop.expand())
        archive = self._filter_pareto_front(archive)
        return archive.get_result()

    def _create_offspring(self, seg_pop: SegmentPopulation) -> SegmentPopulation:
        n = len(seg_pop)
        children = []

        for i in range(0, n - 1, 2):
            p1, p2 = seg_pop[i], seg_pop[i + 1]
            if np.random.random() < self.crossover_rate:
                c1, c2 = segment_crossover(p1, p2, self.crossover_eta)
            else:
                c1, c2 = p1.copy(), p2.copy()
            children.append(segment_mutation(c1, self.mutation_rate, self.mutation_eta))
            children.append(segment_mutation(c2, self.mutation_rate, self.mutation_eta))

        if n % 2 == 1:
            children.append(segment_mutation(seg_pop[n - 1].copy(), self.mutation_rate, self.mutation_eta))

        return SegmentPopulation(
            children, seg_pop.scheduling_days,
            seg_pop.total_water, seg_pop.flow_min, seg_pop.flow_max,
        )

    @staticmethod
    def _combine(pop1: SegmentPopulation, pop2: SegmentPopulation) -> SegmentPopulation:
        return SegmentPopulation(
            pop1.solutions + pop2.solutions,
            pop1.scheduling_days, pop1.total_water,
            pop1.flow_min, pop1.flow_max,
        )

    def _environmental_selection(
        self,
        objectives: np.ndarray,
        target_size: int,
        reference_points: np.ndarray,
    ) -> list[int]:
        fronts = self._fast_non_dominated_sort(objectives)
        selected = []
        remaining = target_size

        for front in fronts:
            if len(front) <= remaining:
                selected.extend(front)
                remaining -= len(front)
            else:
                selected.extend(
                    self._niche_count_selection(front, objectives, remaining, reference_points)
                )
                break

        return selected

    @staticmethod
    def _fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(objectives[i] >= objectives[j]) and np.any(objectives[i] > objectives[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front += 1

        return [f for f in fronts if f]

    @staticmethod
    def _niche_count_selection(
        front: list[int],
        objectives: np.ndarray,
        n_select: int,
        reference_points: np.ndarray,
    ) -> list[int]:
        front_obj = objectives[front]
        ideal = front_obj.min(axis=0)
        nadir = front_obj.max(axis=0)
        denom = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)
        normalized = (front_obj - ideal) / denom

        associations = []
        distances = []
        for sol in normalized:
            dists = np.linalg.norm(reference_points - sol.reshape(1, -1), axis=1)
            nearest = np.argmin(dists)
            associations.append(nearest)
            distances.append(dists[nearest])

        niche_counts = np.zeros(len(reference_points), dtype=int)
        for a in associations:
            niche_counts[a] += 1

        selected = []
        remaining = set(range(len(front)))

        while len(selected) < n_select and remaining:
            min_count_idx = np.argmin(niche_counts)
            candidates = [i for i in remaining if associations[i] == min_count_idx]
            if candidates:
                best = candidates[np.argmin([distances[i] for i in candidates])]
                selected.append(front[best])
                remaining.remove(best)
                niche_counts[min_count_idx] += 1
            else:
                niche_counts[min_count_idx] = len(front) + 1

        if len(selected) < n_select and remaining:
            selected.extend([front[i] for i in list(remaining)[:n_select - len(selected)]])

        return selected

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
