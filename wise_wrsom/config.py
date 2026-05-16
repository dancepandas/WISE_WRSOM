"""项目配置模块。

使用 dataclass 定义配置结构，支持从 YAML 文件加载。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MuskingumParams:
    """马斯京根模型参数。"""
    k: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 13, 16, 16, 80, 150])
    x: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0.2])
    t: list[float] = field(default_factory=lambda: [24, 24, 24, 24, 24, 24, 24, 24, 24])
    cost_coefficients: list[list[float]] = field(default_factory=lambda: [
        [2e-06, 4e-07, 7e-07, 1e-06, 5e-05, 9e-05, 5e-07, 0, -0.0256],
        [0.0051, 0.0257, 0.0607, 0.0262, -0.0022, -0.0044, -0.0002, -1e-04, 1.2622],
        [0.1383, 0.1532, 0.1517, 0.5247, 1.026, 0.2016, 0.3134, 0.1667, -4.191],
    ])
    surface_coefficients: list[list[float]] = field(default_factory=lambda: [
        [-0.0025, -0.0029, -0.0006, -0.0064, -0.0046, -0.003, -7e-05, -0.0012, -0.1226],
        [0.7429, 0.983, 0.2342, 1.6813, 1.0961, 0.9248, 0.02, 0.2796, 33.107],
        [48.48, 36.206, 39.956, 62.855, 68.038, 224.92, 85.033, 39.779, 770.2],
    ])
    special_lose_water: list[float] = field(
        default_factory=lambda: [0, 0, 0, 0, 0, 0, 6500000, 0, 850000]
    )
    initial_flow: list[float] = field(
        default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    tributary_data_path: str = ""


@dataclass
class OptimizerParams:
    """优化算法参数。"""
    population_size: int = 100
    scheduling_days: int = 80
    max_iterations_outer: int = 5
    max_iterations_inner: int = 5
    flow_min: float = 3.0
    flow_max: float = 100.0
    total_water: float = 150_000_000
    velocity_max: float = 5.0
    velocity_min: float = -5.0
    algorithm: str = "auto"  # "auto", "smpso", "nsga3", "moead"


@dataclass
class DecisionParams:
    """决策方法参数。"""
    subjective_weights: list[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.25, 0.25]
    )
    method: str = "topsis"


@dataclass
class OutputParams:
    """输出参数。"""
    output_dir: str = ""
    pareto_file: str = "SM_model_result.json"
    ranking_file: str = "DO_g_result.json"


@dataclass
class ProjectConfig:
    """项目总配置。"""
    muskingum: MuskingumParams = field(default_factory=MuskingumParams)
    optimizer: OptimizerParams = field(default_factory=OptimizerParams)
    decision: DecisionParams = field(default_factory=DecisionParams)
    output: OutputParams = field(default_factory=OutputParams)

    def __post_init__(self):
        if not self.muskingum.tributary_data_path:
            data_dir = os.path.join(self._get_project_root(), "data")
            self.muskingum.tributary_data_path = os.path.join(data_dir, "water_divide.xlsx")
        if not self.output.output_dir:
            self.output.output_dir = os.path.join(self._get_project_root(), "data")

    @staticmethod
    def _get_project_root() -> str:
        return str(Path(__file__).parent.parent)

    @property
    def pareto_file_path(self) -> str:
        return os.path.join(self.output.output_dir, self.output.pareto_file)

    @property
    def ranking_file_path(self) -> str:
        return os.path.join(self.output.output_dir, self.output.ranking_file)

    @classmethod
    def from_yaml(cls, path: str) -> ProjectConfig:
        """从 YAML 文件加载配置。"""
        import dataclasses

        def _convert(value):
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
            if isinstance(value, list):
                return [_convert(v) for v in value]
            return value

        def _merge(instance, overrides: dict):
            converted = {k: _convert(v) for k, v in overrides.items()}
            fields = {f.name for f in dataclasses.fields(instance)}
            return type(instance)(**{**{k: getattr(instance, k) for k in fields}, **converted})

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        if "muskingum" in data:
            config.muskingum = _merge(config.muskingum, data["muskingum"])
        if "optimizer" in data:
            config.optimizer = _merge(config.optimizer, data["optimizer"])
        if "decision" in data:
            config.decision = _merge(config.decision, data["decision"])
        if "output" in data:
            config.output = _merge(config.output, data["output"])

        config.__post_init__()
        return config

    def to_yaml(self, path: str) -> None:
        """导出配置到 YAML 文件。"""
        import dataclasses

        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, allow_unicode=True)
