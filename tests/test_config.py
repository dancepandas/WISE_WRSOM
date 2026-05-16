"""配置模块测试。"""
import os
import tempfile

import pytest

from wise_wrsom.config import (
    DecisionParams,
    MuskingumParams,
    OptimizerParams,
    OutputParams,
    ProjectConfig,
)


class TestMuskingumParams:
    def test_default_values(self):
        p = MuskingumParams()
        assert len(p.k) == 9
        assert len(p.x) == 9
        assert len(p.t) == 9
        assert len(p.cost_coefficients) == 3
        assert len(p.surface_coefficients) == 3

    def test_custom_values(self):
        p = MuskingumParams(k=[1, 2, 3], x=[0.1, 0.2, 0.3], t=[24, 24, 24])
        assert p.k == [1, 2, 3]


class TestOptimizerParams:
    def test_defaults(self):
        p = OptimizerParams()
        assert p.population_size == 100
        assert p.scheduling_days == 80
        assert p.algorithm == "auto"
        assert p.total_water == 150_000_000


class TestProjectConfig:
    def test_default_config(self):
        config = ProjectConfig()
        assert config.muskingum.k == [0, 0, 0, 0, 13, 16, 16, 80, 150]
        assert config.optimizer.population_size == 100
        assert config.decision.subjective_weights == [0.25, 0.25, 0.25, 0.25]

    def test_from_yaml(self, config_yaml):
        assert config_yaml.optimizer.algorithm == "auto"
        assert config_yaml.optimizer.population_size == 100
        assert config_yaml.muskingum.k == [0, 0, 0, 0, 13, 16, 16, 80, 150]

    def test_to_yaml_and_reload(self):
        config = ProjectConfig()
        config.optimizer.algorithm = "nsga3"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            tmp_path = f.name

        try:
            config.to_yaml(tmp_path)
            loaded = ProjectConfig.from_yaml(tmp_path)
            assert loaded.optimizer.algorithm == "nsga3"
        finally:
            os.unlink(tmp_path)

    def test_pareto_file_path(self, config):
        assert config.pareto_file_path.endswith("SM_model_result.json")

    def test_ranking_file_path(self, config):
        assert config.ranking_file_path.endswith("DO_g_result.json")
