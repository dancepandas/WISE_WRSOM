"""通用工具函数。"""
from __future__ import annotations

import numpy as np


def summation(summation_list: list) -> np.ndarray:
    """计算各河段渗漏损失水量总和。"""
    arr = np.array(summation_list)
    per_segment = np.sum(arr, axis=2) * 24 * 60 * 60
    return np.round(np.sum(per_segment, axis=0), 2)


def summation_max(summation_list: list) -> tuple[np.ndarray, np.ndarray]:
    """计算各河段水面面积最大值。"""
    arr = np.array(summation_list).T
    per_segment = np.sum(arr, axis=2)
    return np.round(np.max(per_segment, axis=0), 2), per_segment


def summation_any_cross(
    downstream_flow_process_list: list,
    calculation_cross_rank: int = -1,
) -> np.ndarray:
    """计算指定断面的出流水量。"""
    a = np.array(downstream_flow_process_list)
    cross = a[calculation_cross_rank]
    return np.round(np.sum(cross, axis=1) * 24 * 60 * 60, 2)


def calculation_time(downstream_flow_process_list: list) -> list[int]:
    """计算全线通水时长。"""
    a = np.array(downstream_flow_process_list)
    all_positive = np.all(a > 0, axis=0)
    return np.sum(all_positive, axis=1).tolist()
