"""分段编码测试。"""
import numpy as np
import pytest

from wise_wrsom.optimizers.segment import (
    SegmentPopulation,
    SegmentSolution,
    segment_crossover,
    segment_mutation,
)


class TestSegmentSolution:
    def test_basic_creation(self):
        sol = SegmentSolution(
            day_splits=np.array([10, 20, 30]),
            flow_rates=np.array([5.0, 10.0, 15.0]),
            scheduling_days=60,
            total_water=1e8,
            flow_min=1.0,
            flow_max=50.0,
        )
        assert sol.n_segments == 3
        assert sol.scheduling_days == 60

    def test_normalize_satisfies_water_constraint(self):
        """normalize 后水量应在目标水量 ±10% 范围内。"""
        total_water = 1.5e8
        sol = SegmentSolution(
            day_splits=np.array([10, 20, 30]),
            flow_rates=np.array([5.0, 10.0, 15.0]),
            scheduling_days=60,
            total_water=total_water,
            flow_min=1.0,
            flow_max=50.0,
        )
        sol.normalize()
        actual = np.sum(sol.flow_rates * sol.day_splits) * 24 * 60 * 60
        assert abs(actual - total_water) / total_water < 0.10

    def test_normalize_with_clip(self):
        """clip 后归一化仍应满足水量约束（在可行范围内，±10%）。"""
        total_water = 5e7
        sol = SegmentSolution(
            day_splits=np.array([10, 20, 30]),
            flow_rates=np.array([30.0, 40.0, 50.0]),
            scheduling_days=60,
            total_water=total_water,
            flow_min=1.0,
            flow_max=100.0,
        )
        sol.normalize()
        actual = np.sum(sol.flow_rates * sol.day_splits) * 24 * 60 * 60
        assert abs(actual - total_water) / total_water < 0.10

    def test_expand(self):
        sol = SegmentSolution(
            day_splits=np.array([2, 3]),
            flow_rates=np.array([5.0, 10.0]),
            scheduling_days=5,
            total_water=1e8,
            flow_min=1.0,
            flow_max=50.0,
        )
        expanded = sol.expand()
        assert len(expanded) == 5
        np.testing.assert_array_equal(expanded[:2], 5.0)
        np.testing.assert_array_equal(expanded[2:], 10.0)

    def test_normalize_preserves_clip_bounds(self):
        """归一化后流量不应超出 [flow_min, flow_max]。"""
        sol = SegmentSolution(
            day_splits=np.array([15, 15, 30]),
            flow_rates=np.array([3.0, 8.0, 12.0]),
            scheduling_days=60,
            total_water=1.5e8,
            flow_min=1.0,
            flow_max=50.0,
        )
        sol.normalize()
        assert np.all(sol.flow_rates >= sol.flow_min)
        assert np.all(sol.flow_rates <= sol.flow_max)


class TestSegmentPopulation:
    def test_basic_creation(self):
        solutions = [
            SegmentSolution(
                day_splits=np.array([20, 20, 20]),
                flow_rates=np.array([5.0, 10.0, 15.0]),
                scheduling_days=60,
                total_water=1e8,
                flow_min=1.0,
                flow_max=50.0,
            ),
            SegmentSolution(
                day_splits=np.array([10, 30, 20]),
                flow_rates=np.array([8.0, 12.0, 6.0]),
                scheduling_days=60,
                total_water=1e8,
                flow_min=1.0,
                flow_max=50.0,
            ),
        ]
        pop = SegmentPopulation(solutions, scheduling_days=60, total_water=1e8, flow_min=1.0, flow_max=50.0)
        assert len(pop) == 2

    def test_expand(self):
        solutions = [
            SegmentSolution(
                day_splits=np.array([2, 3]),
                flow_rates=np.array([5.0, 10.0]),
                scheduling_days=5,
                total_water=1e8,
                flow_min=1.0,
                flow_max=50.0,
            ),
        ]
        pop = SegmentPopulation(solutions, scheduling_days=5, total_water=1e8, flow_min=1.0, flow_max=50.0)
        expanded = pop.expand()
        assert expanded.shape == (1, 5)


class TestSegmentMutation:
    def test_mutation_preserves_water(self):
        """变异后归一化应维持水量约束（±10%）。"""
        total_water = 1.5e8
        sol = SegmentSolution(
            day_splits=np.array([15, 15, 30]),
            flow_rates=np.array([5.0, 10.0, 15.0]),
            scheduling_days=60,
            total_water=total_water,
            flow_min=1.0,
            flow_max=50.0,
        )
        sol.normalize()
        mutated = segment_mutation(sol, rate=0.5, eta=20.0)
        actual = np.sum(mutated.flow_rates * mutated.day_splits) * 24 * 60 * 60
        assert abs(actual - total_water) / total_water < 0.10


class TestSegmentCrossover:
    def test_crossover_produces_valid_offspring(self):
        total_water = 1.5e8
        parent1 = SegmentSolution(
            day_splits=np.array([15, 15, 30]),
            flow_rates=np.array([5.0, 10.0, 15.0]),
            scheduling_days=60,
            total_water=total_water,
            flow_min=1.0,
            flow_max=50.0,
        )
        parent1.normalize()
        parent2 = SegmentSolution(
            day_splits=np.array([20, 10, 30]),
            flow_rates=np.array([8.0, 12.0, 6.0]),
            scheduling_days=60,
            total_water=total_water,
            flow_min=1.0,
            flow_max=50.0,
        )
        parent2.normalize()
        child1, child2 = segment_crossover(parent1, parent2)
        actual1 = np.sum(child1.flow_rates * child1.day_splits) * 24 * 60 * 60
        assert abs(actual1 - total_water) / total_water < 0.10