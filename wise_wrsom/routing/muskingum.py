"""马斯京根河道流量计算模型。"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..protocols.routing import RiverParams, RoutingResult
from ..utils import calculation_time, summation, summation_any_cross, summation_max


class MuskingumRouter:
    """马斯京根河道流量计算模型。"""

    def __init__(self):
        self._coefficients_cache: np.ndarray | None = None
        self._tributary_cache: tuple[str, np.ndarray] | None = None

    def compute(self, population: np.ndarray, river_params: RiverParams) -> RoutingResult:
        k = river_params.k
        x = river_params.x
        t = river_params.t
        cost_coefficients = np.array(river_params.cost_coefficients)
        surface_coefficients = np.array(river_params.surface_coefficients)
        special_lose_water = river_params.special_lose_water
        initial_flow = river_params.initial_flow

        tributary_flow_process_list = self._load_tributary_data(
            river_params.tributary_data_path, len(k), population.shape[1]
        )

        coeff_result = self._compute_coefficients(k, x, t)

        upstream_flow_process_cost_list = []
        water_surface_area_value_list = []
        downstream_flow_process_list = []

        upstream_flow_process = population
        downstream_flow_process_list.append(list(upstream_flow_process))

        for segment_idx in range(len(k)):
            or_q = initial_flow[segment_idx]
            tributary_flow_process = tributary_flow_process_list[segment_idx]

            downstream_flow_process = self._muskingum_segment(
                upstream_flow_process, coeff_result, segment_idx,
                cost_coefficients, special_lose_water,
                tributary_flow_process, or_q,
            )
            downstream_flow_process_list.append(list(downstream_flow_process))

            infiltration = self._evaluate_quadratic(
                upstream_flow_process, cost_coefficients, segment_idx, clamp_negative=True
            )
            upstream_flow_process_cost_list.append(list(infiltration))

            surface_area = self._evaluate_quadratic(
                upstream_flow_process, surface_coefficients, segment_idx, clamp_negative=False
            )
            water_surface_area_value_list.append(list(surface_area))

            upstream_flow_process = downstream_flow_process

        water_duration = np.array(calculation_time(downstream_flow_process_list))
        outflow = summation_any_cross(downstream_flow_process_list)
        surface_area_result = summation_max(water_surface_area_value_list)[0]
        infiltration_result = summation(upstream_flow_process_cost_list)

        return RoutingResult(
            water_duration=water_duration,
            outflow=outflow,
            surface_area=surface_area_result,
            infiltration=infiltration_result,
            downstream_flows=downstream_flow_process_list,
        )

    def _load_tributary_data(
        self, excel_path: str, n_segments: int, n_days: int
    ) -> np.ndarray:
        cache_key = (excel_path, n_segments, n_days)
        if self._tributary_cache is not None and self._tributary_cache[0] == cache_key:
            return self._tributary_cache[1]

        df = pd.read_excel(excel_path)
        qingbaikou = df[df.columns[0]].values.tolist()
        yong_yin = df[df.columns[1]].values.tolist()
        xiao_hong_men = df[df.columns[2]].values.tolist()
        daxing = df[df.columns[3]].values.tolist()
        nan_shui = df[df.columns[4]].values.tolist()

        tributary_list = list(np.zeros((n_segments, len(qingbaikou))))
        tributary_list[1] = np.array(qingbaikou)
        tributary_list[4] = np.array(yong_yin) * (-1)
        tributary_list[5] = np.array(xiao_hong_men)
        tributary_list[6] = np.array(nan_shui) + np.array(daxing) * (-1)
        result = np.array(tributary_list)

        self._tributary_cache = (cache_key, result)
        return result

    def _compute_coefficients(
        self, k: list[float], x: list[float], t: list[float]
    ) -> np.ndarray:
        if self._coefficients_cache is not None:
            return self._coefficients_cache

        k_arr = np.array(k)
        x_arr = np.array(x)
        t_arr = np.array(t)
        norm = k_arr - k_arr * x_arr + 0.5 * t_arr
        c0 = (0.5 * t_arr - k_arr * x_arr) / norm
        c1 = (0.5 * t_arr + k_arr * x_arr) / norm
        c2 = (k_arr - k_arr * x_arr - 0.5 * t_arr) / norm
        result = np.array([c0, c1, c2]).T

        target_first = np.array([1, 1, -1])
        for i in range(result.shape[0]):
            if np.allclose(result[i], target_first):
                result[i] = [1, 0, 0]

        self._coefficients_cache = result
        return result

    @staticmethod
    def _evaluate_quadratic(
        flow: np.ndarray,
        coefficients: np.ndarray,
        segment_idx: int,
        clamp_negative: bool = False,
    ) -> np.ndarray:
        """统一的二次经验方程计算（渗漏损失 / 水面面积）。"""
        coeff = coefficients.T
        r0, r1, r2 = coeff[segment_idx, 0], coeff[segment_idx, 1], coeff[segment_idx, 2]
        mask = flow > 0
        result = np.zeros_like(flow)
        result[mask] = r0 * np.square(flow[mask]) + r1 * flow[mask] + r2
        if clamp_negative:
            result[result < 0] = 0
        return result

    def _muskingum_segment(
        self,
        upstream_flow_process: np.ndarray,
        coeff_result: np.ndarray,
        segment_idx: int,
        cost_coefficients: np.ndarray,
        special_lose_water: list[float],
        tributary_flow_process: np.ndarray,
        or_q: float,
    ) -> np.ndarray:
        infiltration = self._evaluate_quadratic(
            upstream_flow_process, cost_coefficients, segment_idx, clamp_negative=True
        )
        upstream_after_loss = upstream_flow_process - infiltration

        # 向量化支流加法
        trib = tributary_flow_process[:upstream_after_loss.shape[1]]
        if np.any(trib != 0):
            upstream_calculation = np.maximum(upstream_after_loss + trib, 0)
        else:
            upstream_calculation = upstream_after_loss.copy()

        c0 = coeff_result[segment_idx, 0]
        c1 = coeff_result[segment_idx, 1]
        c2 = coeff_result[segment_idx, 2]

        upstream_cut_first = upstream_calculation[:, 1:]
        upstream_cut_last = upstream_calculation[:, :-1]

        downstream_flow_process = np.zeros_like(upstream_flow_process)
        downstream_flow_process[:, 0] = or_q
        downstream_flow_process_correct = np.zeros_like(upstream_flow_process)
        downstream_flow_process_correct[:, 0] = or_q

        w = special_lose_water[segment_idx]
        n_days = upstream_cut_first.shape[1]

        for i in range(upstream_cut_first.shape[0]):
            cumsum_prev = downstream_flow_process[i, 0] * 24 * 60 * 60
            for j in range(n_days):
                downstream_flow_process[i, j + 1] = (
                    c0 * upstream_cut_first[i, j]
                    + c1 * upstream_cut_last[i, j]
                    + c2 * downstream_flow_process[i, j]
                )
                if downstream_flow_process[i, j + 1] < 0:
                    downstream_flow_process[i, j + 1] = 0

                cumsum_now = cumsum_prev + downstream_flow_process[i, j + 1] * 24 * 60 * 60
                if cumsum_now <= w:
                    downstream_flow_process_correct[i, j + 1] = 0
                elif cumsum_now > w and cumsum_prev < w:
                    downstream_flow_process[i, j + 1] = (cumsum_now - w) / (24 * 60 * 60)
                    downstream_flow_process_correct[i, j + 1] = downstream_flow_process[i, j + 1]
                else:
                    downstream_flow_process_correct[i, j + 1] = downstream_flow_process[i, j + 1]
                cumsum_prev = cumsum_now

        return downstream_flow_process_correct
