"""调度目标注册表。

通过装饰器 @register("name") 注册目标函数类，支持动态发现和创建。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..protocols.objective import ObjectiveFunction

_REGISTRY: dict[str, type[ObjectiveFunction]] = {}


def register(name: str):
    """注册目标函数类。

    Usage:
        @register("water_duration")
        class WaterDurationObjective:
            ...
    """
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_objective(name: str) -> type[ObjectiveFunction]:
    """获取已注册的目标函数类。"""
    if name not in _REGISTRY:
        raise KeyError(f"未注册的目标函数: {name}，可用: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_objectives() -> list[str]:
    """列出所有已注册的目标函数名称。"""
    return list(_REGISTRY.keys())


def create_objectives(names: list[str]) -> list[ObjectiveFunction]:
    """根据名称列表创建目标函数实例。"""
    return [get_objective(n)() for n in names]
