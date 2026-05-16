"""优化算法注册表。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..protocols.optimizer import Optimizer

_REGISTRY: dict[str, type[Optimizer]] = {}


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
