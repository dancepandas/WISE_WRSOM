"""测试公共配置。"""
import os
import sys

import numpy as np
import pytest

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wise_wrsom.config import ProjectConfig
from wise_wrsom.protocols.routing import RiverParams, RoutingResult


@pytest.fixture
def config():
    """默认项目配置。"""
    return ProjectConfig()


@pytest.fixture
def config_yaml():
    """从 YAML 加载的配置。"""
    path = os.path.join(PROJECT_ROOT, "config.yaml")
    return ProjectConfig.from_yaml(path)


@pytest.fixture
def river_params(config):
    """河道参数。"""
    return RiverParams.from_muskingum(config.muskingum)


@pytest.fixture
def sample_population():
    """示例种群（5 个方案，80 天）。"""
    np.random.seed(42)
    pop = np.random.uniform(5, 50, size=(5, 80))
    # 修正水量约束
    q_total = 150_000_000 / (24 * 60 * 60)
    row_sums = pop.sum(axis=1, keepdims=True)
    pop = pop * (q_total / row_sums)
    return pop


@pytest.fixture
def sample_routing_result():
    """示例河道流量计算结果。"""
    np.random.seed(42)
    n = 10
    return RoutingResult(
        water_duration=np.random.randint(10, 80, n).astype(float),
        outflow=np.random.uniform(1e6, 1e8, n),
        surface_area=np.random.uniform(100, 5000, n),
        infiltration=np.random.uniform(1e5, 1e7, n),
        downstream_flows=[np.random.uniform(0, 100, (n, 80))],
    )


@pytest.fixture
def sample_objectives_matrix():
    """示例目标值矩阵。"""
    np.random.seed(42)
    return np.random.rand(20, 4)
