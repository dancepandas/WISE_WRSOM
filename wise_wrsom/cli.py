"""CLI 命令行入口。

使用 Click 框架提供命令行接口，支持 Agent 集成。
所有命令支持 --format json 输出机器可解析结果，便于 Agent 系统调用。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import click

from .config import ProjectConfig


# ── 退出码 ──────────────────────────────────────────────
EXIT_OK = 0
EXIT_PARTIAL = 1   # 部分完成（如优化完成但可视化失败）
EXIT_ERROR = 2     # 输入/配置错误
EXIT_FAIL = 3      # 运行时失败


# ── JSON 输出辅助 ───────────────────────────────────────
def _emit(result: dict, fmt: str) -> None:
    """统一输出：json 模式输出 JSON，text 模式输出友好文本。"""
    if fmt == "json":
        clean = {k: v for k, v in result.items() if k != "_text"}
        payload = json.dumps(clean, ensure_ascii=False, indent=2)
        # Windows 下 click.echo 可能使用 GBK，强制 UTF-8
        try:
            click.echo(payload)
        except UnicodeEncodeError:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            click.echo(payload)
    else:
        for line in result.get("_text", []):
            click.echo(line)


# ── 公共选项 ────────────────────────────────────────────
_format_option = click.option(
    "--format", "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式：text（人类可读）/ json（机器可解析）",
)


@click.group()
@click.version_option(version="2.0.0", prog_name="wise-wrsom")
def cli():
    """WISE-WRSOM 水资源优化调度系统。

    支持多目标优化、决策排序、结果可视化等功能。
    各模块可通过协议接口灵活替换。
    添加 --format json 获取机器可解析输出，便于 Agent 集成。
    """
    pass


@cli.command()
@click.option("--config", "-c", "config_path", default=None, help="配置文件路径（YAML）")
@click.option("--algorithm", "-a", default=None, help="优化算法（smpso/nsga3/moead/auto）")
@click.option("--iterations", "-n", default=None, type=int, help="最大迭代次数")
@click.option("--population", "-p", default=None, type=int, help="种群大小")
@click.option("--objectives", "-o", default=None, help="目标函数列表（逗号分隔）")
@_format_option
def run(config_path, algorithm, iterations, population, objectives, fmt):
    """运行完整的优化调度流水线。"""
    config = _load_config(config_path)
    _apply_overrides(config, algorithm, iterations, population, objectives)

    from .pipeline import Pipeline
    pipeline = Pipeline(config)
    result = pipeline.run()

    n_pareto = len(result.optimization_result.pareto_objectives)
    top_score = float(result.rankings.max())

    _emit({
        "status": "ok",
        "pareto_count": n_pareto,
        "top_score": top_score,
        "pareto_file": result.pareto_file,
        "ranking_file": result.ranking_file,
        "_text": [
            f"\nPareto 解数量: {n_pareto}",
            f"最高排名分: {top_score:.4f}",
        ],
    }, fmt)


@cli.command()
@click.option("--config", "-c", "config_path", default=None, help="配置文件路径")
@click.option("--algorithm", "-a", default="nsga3", help="优化算法")
@click.option("--iterations", "-n", default=5, type=int, help="迭代次数")
@click.option("--population", "-p", default=100, type=int, help="种群大小")
@click.option("--objectives", "-o", default=None, help="目标函数列表")
@click.option("--output", default=None, help="输出文件路径")
@_format_option
def optimize(config_path, algorithm, iterations, population, objectives, output, fmt):
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
        "water_duration", "groundwater_recharge", "surface_area",
        "outflow_control", "water_balance",
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
        "metadata": {
            "total_water": config.optimizer.total_water,
            "scheduling_days": config.optimizer.scheduling_days,
            "population_size": population,
            "optimizer": type(optimizer).__name__,
            "objective_names": [obj.name for obj in obj_list],
            "objective_directions": [obj.direction for obj in obj_list],
        },
        "archive_total": result.pareto_objectives.tolist(),
        "archive_population_total": result.pareto_population.tolist(),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    n_pareto = len(result.pareto_objectives)
    _emit({
        "status": "ok",
        "pareto_count": n_pareto,
        "output_file": output_path,
        "_text": [
            f"Pareto 最优解数量: {n_pareto}",
            f"结果已保存到: {output_path}",
        ],
    }, fmt)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Pareto 结果文件路径")
@click.option("--weights", "-w", default=None, help="主观权重（逗号分隔）")
@click.option("--output", default=None, help="输出文件路径")
@_format_option
def rank(input_path, weights, output, fmt):
    """对 Pareto 最优解进行决策排序。"""
    from .decision.topsis import TOPSISDecision

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    import numpy as np
    objectives_matrix = np.array(data["archive_total"])

    # 从元数据获取目标方向
    metadata = data.get("metadata", {})
    obj_directions = metadata.get("objective_directions", None)
    # 排序仅使用前4个决策目标
    decision_matrix = objectives_matrix[:, :4]
    decision_directions = obj_directions[:4] if obj_directions else None

    subjective_weights = None
    if weights:
        subjective_weights = [float(w) for w in weights.split(",")]

    decision = TOPSISDecision()
    rankings = decision.rank(
        decision_matrix,
        subjective_weights=subjective_weights,
        directions=decision_directions,
    )

    output_path = output or input_path.replace("SM_model_result", "DO_g_result")
    ranked_indices = np.argsort(rankings)[::-1].tolist()
    result_data = {
        "g": rankings.tolist(),
        "ranked_indices": ranked_indices,
        "objective_names": metadata.get("objective_names", None),
        "objective_directions": metadata.get("objective_directions", None),
        "objective_values": objectives_matrix.tolist(),
        "population": np.array(data["archive_population_total"]).tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f)

    n_solutions = len(rankings)
    top_score = float(rankings.max())
    _emit({
        "status": "ok",
        "solution_count": n_solutions,
        "top_score": top_score,
        "output_file": output_path,
        "ranked_indices": ranked_indices[:10],
        "_text": [
            f"排序完成，共 {n_solutions} 个方案",
            f"最高分: {top_score:.4f}",
            f"结果已保存到: {output_path}",
        ],
    }, fmt)


@cli.command("list-objectives")
@_format_option
def list_objectives_cmd(fmt):
    """列出所有可用的调度目标。"""
    from .objectives import list_objectives, get_objective

    names = list_objectives()
    items = []
    for name in names:
        cls = get_objective(name)
        obj = cls()
        items.append({
            "id": name,
            "name": obj.name,
            "direction": obj.direction,
        })

    _emit({
        "status": "ok",
        "objectives": items,
        "_text": [
            "可用的调度目标:",
            *(
                f"  {it['id']:25s} {it['name']} ({'↑ 最大化' if it['direction'] == 'max' else '↓ 最小化'})"
                for it in items
            ),
        ],
    }, fmt)


@cli.command("list-optimizers")
@_format_option
def list_optimizers_cmd(fmt):
    """列出所有可用的优化算法。"""
    from .optimizers import list_optimizers

    names = list_optimizers()
    descriptions = {
        "smpso": "改进的 SMPSO 粒子群算法（快速收敛）",
        "nsga3": "NSGA-III 非支配排序遗传算法（2-3 目标推荐）",
        "moead": "MOEA/D 基于分解的多目标进化算法（4+ 目标推荐）",
    }
    items = [{"id": name, "description": descriptions.get(name, "")} for name in names]

    _emit({
        "status": "ok",
        "optimizers": items,
        "_text": [
            "可用的优化算法:",
            *(f"  {it['id']:15s} {it['description']}" for it in items),
        ],
    }, fmt)


@cli.command("list-routing")
@_format_option
def list_routing_cmd(fmt):
    """列出所有可用的河道流量计算模型。"""
    items = [{"id": "muskingum", "name": "马斯京根法", "default": True}]

    _emit({
        "status": "ok",
        "routing_models": items,
        "_text": [
            "可用的河道流量计算模型:",
            "  muskingum          马斯京根法（默认）",
            "",
            "可通过实现 RoutingModel 协议添加自定义模型（如深度学习模型）",
        ],
    }, fmt)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Pareto 结果文件路径")
@click.option("--type", "-t", "plot_type", default="pareto", type=click.Choice(["pareto", "schedule"]),
              help="图表类型")
@click.option("--ranking", "-r", default=None, help="排序结果文件路径（schedule 类型需要）")
@click.option("--output", default=None, help="图片保存路径")
@_format_option
def visualize(input_path, plot_type, ranking, output, fmt):
    """可视化结果。"""
    from .visualization.plots import plot_pareto_front, plot_schedule_comparison

    if plot_type == "pareto":
        saved = plot_pareto_front(input_path, save_path=output)
    elif plot_type == "schedule":
        if not ranking:
            _emit({
                "status": "error",
                "message": "schedule 类型需要 --ranking 参数",
                "_text": ["错误: schedule 类型需要 --ranking 参数"],
            }, fmt)
            sys.exit(EXIT_ERROR)
        saved = plot_schedule_comparison(input_path, ranking, save_path=output)
    else:
        saved = None

    _emit({
        "status": "ok",
        "plot_type": plot_type,
        "output_file": saved,
        "_text": [
            f"图表类型: {plot_type}",
            f"输出文件: {saved or '未保存'}",
        ],
    }, fmt)


@cli.command()
@click.option("--output", "-o", default="config.yaml", help="输出文件路径")
@_format_option
def init_config(output, fmt):
    """生成默认配置文件。"""
    config = ProjectConfig()
    config.to_yaml(output)

    _emit({
        "status": "ok",
        "output_file": output,
        "_text": [f"默认配置文件已生成: {output}"],
    }, fmt)


@cli.command()
@click.option("--input", "-i", "pareto_path", required=True, help="Pareto 结果文件路径")
@click.option("--ranking", "-r", "ranking_path", required=True, help="排序结果文件路径")
@_format_option
def best(pareto_path, ranking_path, fmt):
    """输出排名第一的最优方案详情。"""
    import numpy as np

    with open(pareto_path, "r", encoding="utf-8") as f:
        pareto_data = json.load(f)
    with open(ranking_path, "r", encoding="utf-8") as f:
        rank_data = json.load(f)

    objectives = np.array(pareto_data["archive_total"])
    population = np.array(pareto_data["archive_population_total"])
    rankings = np.array(rank_data["g"])
    metadata = pareto_data.get("metadata", {})

    best_idx = int(np.argmax(rankings))
    obj_names = metadata.get("objective_names", [f"目标{i+1}" for i in range(objectives.shape[1])])
    obj_dirs = metadata.get("objective_directions", ["max"] * objectives.shape[1])

    best_obj = {name: float(objectives[best_idx, i]) for i, name in enumerate(obj_names)}
    best_schedule = population[best_idx].tolist()
    total_water = metadata.get("total_water", 0)
    actual_water = float(np.sum(population[best_idx]) * 24 * 60 * 60)
    water_ratio = actual_water / total_water if total_water > 0 else 0.0

    _emit({
        "status": "ok",
        "rank": 1,
        "score": float(rankings[best_idx]),
        "objectives": best_obj,
        "objective_directions": obj_dirs,
        "schedule_days": len(best_schedule),
        "total_water_target": total_water,
        "total_water_actual": actual_water,
        "water_balance_ratio": round(water_ratio, 4),
        "schedule": best_schedule,
        "_text": [
            f"最优方案 (排名 #1, 得分 {rankings[best_idx]:.4f})",
            *[f"  {name}: {best_obj[name]:.4f} ({'max' if d == 'max' else 'min'})"
              for name, d in zip(obj_names, obj_dirs)],
            f"  调度天数: {len(best_schedule)}",
            f"  目标水量: {total_water:.0f} m³",
            f"  实际水量: {actual_water:.0f} m³",
            f"  水量比率: {water_ratio:.2%}",
        ],
    }, fmt)


@cli.command()
@click.option("--input", "-i", "pareto_path", required=True, help="Pareto 结果文件路径")
@click.option("--ranking", "-r", "ranking_path", required=True, help="排序结果文件路径")
@click.option("--top-n", "-n", default=5, type=int, help="返回前 N 名方案")
@_format_option
def export(pareto_path, ranking_path, top_n, fmt):
    """导出排名前 N 方案的完整数据（JSON 格式）。"""
    import numpy as np

    with open(pareto_path, "r", encoding="utf-8") as f:
        pareto_data = json.load(f)
    with open(ranking_path, "r", encoding="utf-8") as f:
        rank_data = json.load(f)

    objectives = np.array(pareto_data["archive_total"])
    population = np.array(pareto_data["archive_population_total"])
    rankings = np.array(rank_data["g"])
    metadata = pareto_data.get("metadata", {})

    top_indices = np.argsort(rankings)[::-1][:top_n]
    obj_names = metadata.get("objective_names", [f"目标{i+1}" for i in range(objectives.shape[1])])

    solutions = []
    for rank, idx in enumerate(top_indices, 1):
        actual_water = float(np.sum(population[idx]) * 24 * 60 * 60)
        total_water = metadata.get("total_water", 0)
        solutions.append({
            "rank": rank,
            "score": float(rankings[idx]),
            "index": int(idx),
            "objectives": {name: float(objectives[idx, i]) for i, name in enumerate(obj_names)},
            "schedule": population[idx].tolist(),
            "total_water_actual": actual_water,
            "water_balance_ratio": round(actual_water / total_water, 4) if total_water > 0 else 0.0,
        })

    _emit({
        "status": "ok",
        "total_solutions": len(rankings),
        "exported_count": len(solutions),
        "solutions": solutions,
        "_text": [
            f"共 {len(rankings)} 个方案，导出前 {len(solutions)} 名",
            *[f"  #{s['rank']}: 得分={s['score']:.4f}, 水量比率={s['water_balance_ratio']:.2%}"
              for s in solutions],
        ],
    }, fmt)


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
