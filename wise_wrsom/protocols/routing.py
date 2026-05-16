"""河道流量计算协议定义。

任何河道流量计算模型（马斯京根、深度学习、其他机理模型）只需实现 RoutingModel 协议即可接入系统。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class RiverParams:
    """河道参数集合。"""
    k: list[float]  # 蓄量常数
    x: list[float]  # 楔形储量权重系数
    t: list[float]  # 时间步长（小时）
    cost_coefficients: list[list[float]]  # 损失流量经验方程系数
    surface_coefficients: list[list[float]]  # 水面面积经验方程系数
    special_lose_water: list[float]  # 河道砂石坑、湖泊等蓄水水量
    initial_flow: list[float]  # 河道初始流量
    tributary_data_path: str  # 分水口数据文件路径

    @classmethod
    def from_muskingum(cls, muskingum_params) -> RiverParams:
        """从 MuskingumParams 创建 RiverParams。"""
        return cls(
            k=muskingum_params.k,
            x=muskingum_params.x,
            t=muskingum_params.t,
            cost_coefficients=muskingum_params.cost_coefficients,
            surface_coefficients=muskingum_params.surface_coefficients,
            special_lose_water=muskingum_params.special_lose_water,
            initial_flow=muskingum_params.initial_flow,
            tributary_data_path=muskingum_params.tributary_data_path,
        )


@dataclass
class RoutingResult:
    """河道流量计算结果。

    Attributes:
        water_duration: 各方案全线通水时长（天），shape=(n_population,)
        outflow: 各方案出流量（m³），shape=(n_population,)
        surface_area: 各方案水面面积（ha），shape=(n_population,)
        infiltration: 各方案渗漏损失水量（m³），shape=(n_population,)
        downstream_flows: 下游流量过程列表，list[np.ndarray]，每个 shape=(n_population, n_days)
    """
    water_duration: np.ndarray
    outflow: np.ndarray
    surface_area: np.ndarray
    infiltration: np.ndarray
    downstream_flows: list[np.ndarray] = field(default_factory=list)


@runtime_checkable
class RoutingModel(Protocol):
    """河道流量计算协议。

    实现此协议的类可用于替换河道流量计算模块，例如：
    - MuskingumRouter（马斯京根法，当前默认）
    - 深度学习模型
    - 其他水文机理模型

    只需保证 compute() 方法的输入输出符合约定即可。
    """

    def compute(self, population: np.ndarray, river_params: RiverParams) -> RoutingResult:
        """计算河道流量过程。

        Args:
            population: 调度方案矩阵，shape=(n_population, n_days)，单位 m³/s
            river_params: 河道参数

        Returns:
            RoutingResult: 包含通水时长、出流量、水面面积、渗漏损失等结果
        """
        ...
