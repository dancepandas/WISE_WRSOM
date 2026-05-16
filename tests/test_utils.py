"""工具函数测试。"""
import numpy as np

from wise_wrsom.utils import (
    calculation_time,
    summation,
    summation_any_cross,
    summation_max,
)


class TestCalculationTime:
    def test_basic(self):
        # 3 个河段，2 个方案，3 天
        flows = [
            np.array([[10, 20, 30], [5, 5, 5]]),  # 河段 0
            np.array([[10, 0, 30], [5, 5, 5]]),   # 河段 1
            np.array([[10, 20, 30], [5, 5, 5]]),  # 河段 2
        ]
        time_list = calculation_time(flows)
        assert len(time_list) == 2
        assert time_list[0] == 2  # 第 2 天河段 1 为 0
        assert time_list[1] == 3


class TestSummationAnyCross:
    def test_basic(self):
        flows = [
            np.array([[10, 20], [5, 5]]),
            np.array([[30, 40], [10, 10]]),
        ]
        outflow = summation_any_cross(flows, calculation_cross_rank=-1)
        assert len(outflow) == 2
        # 方案 0: (30 + 40) * 86400
        assert np.isclose(outflow[0], (30 + 40) * 24 * 60 * 60)


class TestSummationMax:
    def test_basic(self):
        # 各河段水面面积列表
        area_list = [
            [[10, 20], [5, 5]],   # 河段 0
            [[30, 40], [10, 10]],  # 河段 1
        ]
        result, matrix = summation_max(area_list)
        assert len(result) == 2
