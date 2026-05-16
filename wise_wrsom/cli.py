"""CLI 命令行入口。

使用 Click 框架提供命令行接口，支持 Agent 集成。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .config import ProjectConfig


@click.group()
@click.version_option(version="2.0.0", prog_name="wise-wrsom")
def cli():
    """WISE-WRSOM 水资源优化调度系统。

    支持多目标优化、决策排序、结果可视化等功能。
    各模块可通过协议接口灵活替换。
    """
    pass


@cli.command()
@click.option("--config", "-c", "config_path", default=None, help="配置文件路径（YAML）")
@click.option("--algorithm", "-a", default=None, help="优化算法（smpso/nsga3/moead/auto）")
@click.option("--iterations", "-n", default=None, type=int, help="最大迭代次数")
@click.option("--population", "-p", default=None, type=int, help="种群大小")
@click.option("--objectives", "-o", default=None, help="目标函数列表（逗号分隔）")
def run(config_path, algorithm, iterations, population, objectives):
    """运行完整的优化调度流水线。"""
    config = _load_config(config_path)
    _apply_overrides(config, algorithm, iterations, population, objectives)

    from .pipeline import Pipeline
    pipeline = Pipeline(config)
    result = pipeline.run()

    click.echo(f"\nPareto 解数量: {len(result.optimization_result.pareto_objectives)}")
    click.echo(f"最高排名分: {result.rankings.max():.4f}")


@cli.command()
@click.option("--config", "-c", "config_path", default=None, help="配置文件路径")
@click.option("--algorithm", "-a", default="nsga3", help="优化算法")
@click.option("--iterations", "-n", default=5, type=int, help="迭代次数")
@click.option("--population", "-p", default=100, type=int, help="种群大小")
@click.option("--objectives", "-o", default=None, help="目标函数列表")
@click.option("--output", default=None, help="输出文件路径")
def optimize(config_path, algorithm, iterations, population, objectives, output):
    """仅执行多目标优化。"""
    config = _load_config(config_path)
    _apply_overrides(config, algorithm, iterations, population, objectives)

    from .optimizers import create_optimizer
    from .objectives import create_objectives
    from .routing.muskingum import MuskingumRouter
    from .protocols.routing import RiverParams

    optimizer = create_optimizer(algorithm)
    routing_model = MuskingumRouter()

    obj_names = objectives.split(",") if objectives else [
        "water_duration", "groundwater_recharge", "surface_area", "outflow_control"
    ]
    obj_list = create_objectives(obj_names)
    river_params = RiverParams.from_muskingum(config.muskingum)

    result = optimizer.optimize(
        objectives=obj_list,
        routing_model=routing_model,
        river_params=river_params,
        total_water=config.optimizer.total_water,
        population_size=population,
        scheduling_days=config.optimizer.scheduling_days,
        max_iterations=iterations,
        flow_min=config.optimizer.flow_min,
        flow_max=config.optimizer.flow_max,
    )

    output_path = output or config.pareto_file_path
    data = {
        "archive_total": result.pareto_objectives.tolist(),
        "archive_population_total": result.pareto_population.tolist(),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    click.echo(f"Pareto 最优解数量: {len(result.pareto_objectives)}")
    click.echo(f"结果已保存到: {output_path}")


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Pareto 结果文件路径")
@click.option("--weights", "-w", default=None, help="主观权重（逗号分隔）")
@click.option("--output", default=None, help="输出文件路径")
def rank(input_path, weights, output):
    """对 Pareto 最优解进行决策排序。"""
    from .decision.topsis import TOPSISDecision

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    import numpy as np
    objectives_matrix = np.array(data["archive_total"])

    subjective_weights = None
    if weights:
        subjective_weights = [float(w) for w in weights.split(",")]

    decision = TOPSISDecision()
    rankings = decision.rank(objectives_matrix, subjective_weights)

    output_path = output or input_path.replace("SM_model_result", "DO_g_result")
    result_data = {"g": rankings.tolist()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f)

    click.echo(f"排序完成，共 {len(rankings)} 个方案")
    click.echo(f"最高分: {rankings.max():.4f}")
    click.echo(f"结果已保存到: {output_path}")


@cli.command("list-objectives")
def list_objectives_cmd():
    """列出所有可用的调度目标。"""
    from .objectives import list_objectives, get_objective

    names = list_objectives()
    click.echo("可用的调度目标:")
    for name in names:
        cls = get_objective(name)
        obj = cls()
        direction = "↑ 最大化" if obj.direction == "max" else "↓ 最小化"
        click.echo(f"  {name:25s} {obj.name} ({direction})")


@cli.command("list-optimizers")
def list_optimizers_cmd():
    """列出所有可用的优化算法。"""
    from .optimizers import list_optimizers

    names = list_optimizers()
    click.echo("可用的优化算法:")
    descriptions = {
        "smpso": "改进的 SMPSO 粒子群算法（2-3 目标）",
        "nsga3": "NSGA-III 非支配排序遗传算法（3-5 目标）",
        "moead": "MOEA/D 基于分解的多目标进化算法（5+ 目标）",
    }
    for name in names:
        desc = descriptions.get(name, "")
        click.echo(f"  {name:15s} {desc}")


@cli.command("list-routing")
def list_routing_cmd():
    """列出所有可用的河道流量计算模型。"""
    click.echo("可用的河道流量计算模型:")
    click.echo("  muskingum          马斯京根法（默认）")
    click.echo("")
    click.echo("可通过实现 RoutingModel 协议添加自定义模型（如深度学习模型）")


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Pareto 结果文件路径")
@click.option("--type", "-t", "plot_type", default="pareto", type=click.Choice(["pareto", "schedule"]),
              help="图表类型")
@click.option("--ranking", "-r", default=None, help="排序结果文件路径（schedule 类型需要）")
def visualize(input_path, plot_type, ranking):
    """可视化结果。"""
    from .visualization.plots import plot_pareto_front, plot_schedule_comparison

    if plot_type == "pareto":
        plot_pareto_front(input_path)
    elif plot_type == "schedule":
        if not ranking:
            click.echo("错误: schedule 类型需要 --ranking 参数", err=True)
            sys.exit(1)
        plot_schedule_comparison(input_path, ranking)


@cli.command()
@click.option("--output", "-o", default="config.yaml", help="输出文件路径")
def init_config(output):
    """生成默认配置文件。"""
    config = ProjectConfig()
    config.to_yaml(output)
    click.echo(f"默认配置文件已生成: {output}")


def _load_config(config_path: str | None) -> ProjectConfig:
    """加载配置文件。"""
    if config_path:
        return ProjectConfig.from_yaml(config_path)
    return ProjectConfig()


def _apply_overrides(
    config: ProjectConfig,
    algorithm: str | None,
    iterations: int | None,
    population: int | None,
    objectives: str | None,
) -> None:
    """应用命令行参数覆盖。"""
    if algorithm:
        config.optimizer.algorithm = algorithm
    if iterations:
        config.optimizer.max_iterations_outer = iterations
    if population:
        config.optimizer.population_size = population
