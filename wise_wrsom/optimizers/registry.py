"""优化算法注册表。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import OptimizerParams
    from ..protocols.optimizer import Optimizer

_REGISTRY: dict[str, type[Optimizer]] = {}

# 每个算法从 OptimizerParams 中需要提取的字段名
_ALGORITHM_FIELDS: dict[str, list[str]] = {
    "smpso": ["velocity_max", "velocity_min", "mutation_rate", "mutation_eta"],
    "nsga3": ["crossover_rate", "mutation_rate", "mutation_eta", "crossover_eta", "n_reference_divisions"],
    "moead": ["n_neighbors", "crossover_rate", "mutation_rate", "mutation_eta"],
}


def register(name: str):
    """注册优化算法类。"""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_optimizer(name: str) -> type[Optimizer]:
    """获取已注册的优化算法类。"""
    if name not in _REGISTRY:
        raise KeyError(f"未注册的优化算法: {name}，可用: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_optimizers() -> list[str]:
    """列出所有已注册的优化算法名称。"""
    return list(_REGISTRY.keys())


def create_optimizer(name: str, **kwargs) -> Optimizer:
    """根据名称创建优化算法实例。"""
    return get_optimizer(name)(**kwargs)


def build_optimizer_kwargs(opt_params: OptimizerParams, algorithm: str) -> dict:
    """根据算法类型从 OptimizerParams 中提取对应的构造参数。"""
    fields = _ALGORITHM_FIELDS.get(algorithm)
    if fields is None:
        return {}
    return {f: getattr(opt_params, f) for f in fields}
