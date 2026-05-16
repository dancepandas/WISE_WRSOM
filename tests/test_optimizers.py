"""优化算法测试。"""
import numpy as np

from wise_wrsom.optimizers.base import (
    ParetoArchive,
    PopulationInitializer,
    crowding_distance,
    evaluate_objectives,
)
from wise_wrsom.optimizers.registry import create_optimizer, list_optimizers, get_optimizer


class TestRegistry:
    def test_list_optimizers(self):
        names = list_optimizers()
        assert "smpso" in names
        assert "nsga3" in names
        assert "moead" in names
        assert len(names) == 3

    def test_create_optimizer(self):
        opt = create_optimizer("smpso")
        assert opt.__class__.__name__ == "SMPSOOptimizer"

    def test_create_unknown(self):
        import pytest
        with pytest.raises(KeyError):
            create_optimizer("unknown")


class TestPopulationInitializer:
    def test_initialize_shape(self):
        init = PopulationInitializer(
            total_water=150_000_000,
            scheduling_days=80,
            flow_min=3.0,
            flow_max=100.0,
        )
        pop, days = init.initialize(20)
        assert pop.shape[0] == 20
        assert pop.shape[1] == 80
        assert days.shape[0] == 20

    def test_initialize_within_bounds(self):
        init = PopulationInitializer(150_000_000, 80, 3.0, 100.0)
        pop, _ = init.initialize(10)
        assert np.all(pop >= 3.0)
        assert np.all(pop <= 100.0)


class TestParetoArchive:
    def test_empty_archive(self):
        archive = ParetoArchive()
        assert archive.size == 0

    def test_update_single(self):
        archive = ParetoArchive()
        obj = np.array([[1.0, 2.0], [3.0, 1.0]])
        pop = np.array([[10.0, 20.0], [30.0, 40.0]])
        archive.update(obj, pop)
        assert archive.size == 2

    def test_update_dominance(self):
        archive = ParetoArchive()
        # 第一个解
        archive.update(np.array([[1.0, 2.0]]), np.array([[10.0]]))
        # 被支配的解（两个目标都更差）
        archive.update(np.array([[0.5, 1.5]]), np.array([[20.0]]))
        assert archive.size == 1  # 被支配的解不会被加入

    def test_merge(self):
        a1 = ParetoArchive()
        a1.update(np.array([[1.0, 2.0]]), np.array([[10.0]]))

        a2 = ParetoArchive()
        a2.update(np.array([[3.0, 1.0]]), np.array([[20.0]]))

        a1.merge(a2)
        assert a1.size == 2

    def test_get_result(self):
        archive = ParetoArchive()
        archive.update(
            np.array([[1.0, 2.0], [3.0, 1.0]]),
            np.array([[10.0, 20.0], [30.0, 40.0]]),
        )
        result = archive.get_result()
        assert result.pareto_objectives.shape == (2, 2)
        assert result.pareto_population.shape == (2, 2)


class TestCrowdingDistance:
    def test_two_solutions(self):
        obj = np.array([[1.0, 2.0], [3.0, 1.0]])
        dist = crowding_distance(obj)
        assert len(dist) == 2
        assert np.all(dist == 1e11)  # 两个解时返回大常数

    def test_many_solutions(self):
        np.random.seed(42)
        obj = np.random.rand(20, 3)
        dist = crowding_distance(obj)
        assert len(dist) == 20
        assert np.all(dist >= 0)

    def test_empty(self):
        obj = np.array([]).reshape(0, 2)
        dist = crowding_distance(obj)
        assert len(dist) == 0
