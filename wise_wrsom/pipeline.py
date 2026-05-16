"""流水线编排模块。

编排优化 → 决策 → 保存 → 可视化的完整流程。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .decision.topsis import TOPSISDecision
from .objectives import create_objectives
from .optimizers import create_optimizer
from .protocols.decision import DecisionModel
from .protocols.objective import ObjectiveFunction
from .protocols.optimizer import OptimizationResult
from .protocols.routing import RoutingModel, RiverParams
from .routing.muskingum import MuskingumRouter


@dataclass
class PipelineResult:
    """流水线执行结果。"""
    optimization_result: OptimizationResult
    rankings: np.ndarray
    pareto_file: str
    ranking_file: str


class Pipeline:
    """水资源优化调度流水线。

    编排 优化算法 → 决策排序 → 结果保存 → 可视化 的完整流程。
    各组件通过协议接口注入，可灵活替换。
    """

    def __init__(
        self,
        config: ProjectConfig,
        routing_model: RoutingModel | None = None,
        objectives: list[ObjectiveFunction] | None = None,
        optimizer: object | None = None,
        decision_model: DecisionModel | None = None,
    ):
        self.config = config

        # 河道流量计算模型（默认马斯京根）
        self.routing_model = routing_model or MuskingumRouter()

        # 目标函数（默认全部 4 个）
        if objectives is not None:
            self.objectives = objectives
        else:
            self.objectives = create_objectives([
                "water_duration", "groundwater_recharge",
                "surface_area", "outflow_control", "water_balance",
            ])

        # 优化算法
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = self._create_default_optimizer()

        # 决策模型（默认 TOPSIS）
        self.decision_model = decision_model or TOPSISDecision()

        self.river_params = RiverParams.from_muskingum(config.muskingum)

    def _create_default_optimizer(self):
        """根据目标数量自动选择优化算法。"""
        algorithm = self.config.optimizer.algorithm
        if algorithm == "auto":
            n_obj = len(self.objectives)
            if n_obj <= 3:
                return create_optimizer("nsga3")
            else:
                return create_optimizer("moead")
        return create_optimizer(algorithm)

    def run(self) -> PipelineResult:
        """执行完整流水线。"""
        print("=" * 60)
        print("WISE-WRSOM 水资源优化调度系统")
        print("=" * 60)
        print(f"优化算法: {type(self.optimizer).__name__}")
        print(f"目标函数: {[obj.name for obj in self.objectives]}")
        print(f"目标方向: {[obj.direction for obj in self.objectives]}")
        print(f"决策方法: {type(self.decision_model).__name__}")
        print(f"种群大小: {self.config.optimizer.population_size}")
        print(f"调度天数: {self.config.optimizer.scheduling_days}")
        print(f"迭代次数: {self.config.optimizer.max_iterations_outer}")
        print(f"总调度水量: {self.config.optimizer.total_water} m³")
        print("=" * 60)

        # Step 1: 多目标优化
        print("\n[Step 1/4] 多目标优化...")
        opt_result = self.optimizer.optimize(
            objectives=self.objectives,
            routing_model=self.routing_model,
            river_params=self.river_params,
            total_water=self.config.optimizer.total_water,
            population_size=self.config.optimizer.population_size,
            scheduling_days=self.config.optimizer.scheduling_days,
            max_iterations=self.config.optimizer.max_iterations_outer,
            flow_min=self.config.optimizer.flow_min,
            flow_max=self.config.optimizer.flow_max,
        )
        print(f"Pareto 最优解数量: {len(opt_result.pareto_objectives)}")

        # 将 min 方向的目标值还原为原始值（优化时取负了）
        original_objectives = self._restore_objectives(opt_result.pareto_objectives)

        # 水量验证
        self._verify_water_balance(opt_result.pareto_population)

        # 保存优化结果
        self._save_optimization_result(opt_result, original_objectives)

        # Step 2: 决策排序（仅使用前4个原始目标，水量平衡是约束不参与排序）
        print("\n[Step 2/4] 决策排序...")
        decision_objectives = original_objectives[:, :4]
        decision_directions = [obj.direction for obj in self.objectives[:4]]
        n_decision_obj = len(decision_directions)
        decision_weights = self.config.decision.subjective_weights
        if len(decision_weights) < n_decision_obj:
            decision_weights = [1.0 / n_decision_obj] * n_decision_obj
        rankings = self.decision_model.rank(
            decision_objectives,
            subjective_weights=decision_weights[:n_decision_obj],
            directions=decision_directions,
        )
        print(f"排序完成，最高分: {rankings.max():.4f}")

        # 保存排序结果
        self._save_rankings(rankings, original_objectives, opt_result.pareto_population)

        # Step 3: 输出结果
        print("\n[Step 3/4] 结果输出...")
        print(f"Pareto 结果: {self.config.pareto_file_path}")
        print(f"排序结果: {self.config.ranking_file_path}")

        # Step 4: 可视化
        print("\n[Step 4/4] 可视化...")
        from .visualization.plots import plot_pareto_front, plot_schedule_comparison

        output_dir = self.config.output.output_dir
        objective_names = [obj.name for obj in self.objectives]
        objective_directions = [obj.direction for obj in self.objectives]

        pareto_plot = plot_pareto_front(
            self.config.pareto_file_path,
            objective_names=objective_names,
            objective_directions=objective_directions,
            save_path=str(Path(output_dir) / "pareto_front.png"),
        )
        schedule_plot = plot_schedule_comparison(
            self.config.pareto_file_path,
            self.config.ranking_file_path,
            save_path=str(Path(output_dir) / "schedule.png"),
        )
        print(f"Pareto 前沿图: {pareto_plot}")
        print(f"调度过程图: {schedule_plot}")

        print("\n" + "=" * 60)
        print("优化调度完成！")
        print("=" * 60)

        return PipelineResult(
            optimization_result=opt_result,
            rankings=rankings,
            pareto_file=self.config.pareto_file_path,
            ranking_file=self.config.ranking_file_path,
        )

    def _restore_objectives(self, pareto_objectives: np.ndarray) -> np.ndarray:
        """将优化时取负的 min 方向目标还原为原始值。"""
        restored = pareto_objectives.copy()
        for i, obj in enumerate(self.objectives):
            if obj.direction == "min":
                restored[:, i] = -restored[:, i]
        return restored

    def _verify_water_balance(self, pareto_population: np.ndarray) -> None:
        """验证水量约束满足情况。"""
        total_water = self.config.optimizer.total_water
        for i, pop in enumerate(pareto_population):
            actual = np.sum(pop) * 24 * 60 * 60
            ratio = actual / total_water
            if abs(ratio - 1.0) > 0.10:
                print(f"  方案{i}: 实际水量={actual:.0f}m³, 目标={total_water:.0f}m³, 偏差={ratio:.2%}")

    def _save_optimization_result(
        self, result: OptimizationResult, original_objectives: np.ndarray
    ) -> None:
        """保存优化结果，包含元信息和原始目标值。"""
        data = {
            "metadata": {
                "total_water": self.config.optimizer.total_water,
                "scheduling_days": self.config.optimizer.scheduling_days,
                "population_size": self.config.optimizer.population_size,
                "optimizer": type(self.optimizer).__name__,
                "objective_names": [obj.name for obj in self.objectives],
                "objective_directions": [obj.direction for obj in self.objectives],
            },
            "archive_total": original_objectives.tolist(),
            "archive_population_total": result.pareto_population.tolist(),
        }
        path = Path(self.config.pareto_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _save_rankings(
        self, rankings: np.ndarray, objectives: np.ndarray, population: np.ndarray
    ) -> None:
        """保存排序结果，包含完整信息。"""
        ranked_indices = np.argsort(rankings)[::-1]
        data = {
            "g": rankings.tolist(),
            "ranked_indices": ranked_indices.tolist(),
            "objective_names": [obj.name for obj in self.objectives],
            "objective_directions": [obj.direction for obj in self.objectives],
            "objective_values": objectives.tolist(),
            "population": population.tolist(),
        }
        path = Path(self.config.ranking_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)