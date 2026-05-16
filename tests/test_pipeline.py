"""完整流程集成测试。"""
import json
import os
import tempfile

import numpy as np
import pytest

from wise_wrsom.config import ProjectConfig
from wise_wrsom.pipeline import Pipeline


@pytest.fixture
def test_config(tmp_path):
    """创建测试配置，使用少量迭代以加快测试速度。"""
    config = ProjectConfig()
    config.optimizer.population_size = 20
    config.optimizer.scheduling_days = 80
    config.optimizer.max_iterations_outer = 3
    config.optimizer.max_iterations_inner = 3
    config.optimizer.total_water = 150_000_000
    config.optimizer.flow_min = 3.0
    config.optimizer.flow_max = 100.0
    config.optimizer.algorithm = "nsga3"
    config.output.output_dir = str(tmp_path)
    config.output.pareto_file = "SM_model_result.json"
    config.output.ranking_file = "DO_g_result.json"
    config.__post_init__()
    return config


class TestPipelineIntegration:
    def test_full_pipeline_runs(self, test_config):
        """完整流程应能正常运行。"""
        pipeline = Pipeline(test_config)
        result = pipeline.run()
        assert result is not None
        assert len(result.optimization_result.pareto_objectives) > 0
        assert len(result.rankings) > 0

    def test_water_balance_in_results(self, test_config):
        """保存的调度方案总水量应在目标水量 ±10% 范围内。"""
        pipeline = Pipeline(test_config)
        result = pipeline.run()

        pareto_path = test_config.pareto_file_path
        assert os.path.exists(pareto_path)

        with open(pareto_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        population = np.array(data["archive_population_total"])
        total_water = test_config.optimizer.total_water

        for i in range(len(population)):
            actual = np.sum(population[i]) * 24 * 60 * 60
            rel_err = abs(actual - total_water) / total_water
            assert rel_err < 0.10, (
                f"方案{i}: 实际水量={actual:.0f}m³, "
                f"目标={total_water:.0f}m³, 偏差={rel_err:.2%}"
            )

    def test_ranking_file_has_complete_data(self, test_config):
        """排序结果文件应包含完整信息。"""
        pipeline = Pipeline(test_config)
        result = pipeline.run()

        ranking_path = test_config.ranking_file_path
        assert os.path.exists(ranking_path)

        with open(ranking_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "g" in data
        assert "ranked_indices" in data
        assert "objective_names" in data
        assert "objective_directions" in data
        assert "objective_values" in data
        assert len(data["g"]) == len(data["objective_values"])

    def test_pareto_file_has_metadata(self, test_config):
        """Pareto 结果文件应包含元信息。"""
        pipeline = Pipeline(test_config)
        result = pipeline.run()

        pareto_path = test_config.pareto_file_path
        with open(pareto_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "total_water" in data["metadata"]
        assert "objective_names" in data["metadata"]
        assert "objective_directions" in data["metadata"]

    def test_min_direction_objective(self, test_config):
        """出境水量控制(min方向)目标值应为正数（还原后）。"""
        pipeline = Pipeline(test_config)
        result = pipeline.run()

        pareto_path = test_config.pareto_file_path
        with open(pareto_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objectives = np.array(data["archive_total"])
        directions = data["metadata"]["objective_directions"]
        # 找到 min 方向的目标（排除水量平衡）
        for i, d in enumerate(directions):
            if d == "min" and data["metadata"]["objective_names"][i] != "水量平衡":
                assert np.all(objectives[:, i] > 0), f"min方向目标{i}还原后应为正数"


class TestWaterBalanceVerification:
    """专门验证水量平衡的测试。"""

    def test_population_initializer_water_balance(self):
        """初始化种群应满足水量约束（±10%范围内）。"""
        from wise_wrsom.optimizers.base import PopulationInitializer

        total_water = 150_000_000
        init = PopulationInitializer(total_water, 80, 3.0, 100.0)
        seg_pop = init.initialize_segments(10)

        for sol in seg_pop.solutions:
            actual = np.sum(sol.flow_rates * sol.day_splits) * 24 * 60 * 60
            rel_err = abs(actual - total_water) / total_water
            assert rel_err < 0.10, f"初始化方案水量偏差: {rel_err:.2%}"

    def test_normalize_after_mutation(self):
        """变异后归一化应恢复水量约束（±10%范围内）。"""
        from wise_wrsom.optimizers.segment import SegmentSolution, segment_mutation

        total_water = 150_000_000
        sol = SegmentSolution(
            day_splits=np.array([15, 15, 30]),
            flow_rates=np.array([5.0, 10.0, 15.0]),
            scheduling_days=60,
            total_water=total_water,
            flow_min=3.0,
            flow_max=100.0,
        )
        sol.normalize()

        for _ in range(20):
            mutated = segment_mutation(sol, rate=0.5, eta=20.0)
            actual = np.sum(mutated.flow_rates * mutated.day_splits) * 24 * 60 * 60
            rel_err = abs(actual - total_water) / total_water
            assert rel_err < 0.10, f"变异后水量偏差: {rel_err:.2%}"

    def test_penalty_computation(self):
        """罚函数应正确计算水量偏差。"""
        from wise_wrsom.optimizers.base import compute_water_balance_penalty
        from wise_wrsom.protocols.routing import RoutingResult

        total_water = 150_000_000
        # 构造一个精确满足约束的种群
        q_avg = total_water / (24 * 60 * 60 * 80)
        population = np.full((5, 80), q_avg)

        routing_result = RoutingResult(
            water_duration=np.array([80.0] * 5),
            outflow=np.array([total_water] * 5),
            surface_area=np.array([100.0] * 5),
            infiltration=np.array([1e6] * 5),
            downstream_flows=[population],
        )

        penalty = compute_water_balance_penalty(routing_result, total_water)
        assert np.all(penalty < 1e-6), f"精确满足约束时罚函数应接近0，实际: {penalty}"