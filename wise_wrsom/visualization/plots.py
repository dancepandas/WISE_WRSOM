"""可视化模块。

提供 Pareto 前沿散点图和调度方案对比时序图。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 中文字体设置
mpl.rcParams["font.sans-serif"] = ["SimSun"]
mpl.rcParams["axes.unicode_minus"] = False


def plot_pareto_front(
    file_path: str,
    objective_names: list[str] | None = None,
    objective_directions: list[str] | None = None,
    save_path: str | None = None,
) -> str | None:
    """绘制 Pareto 前沿散点图。

    支持 2/3/4 目标的可视化。自动过滤无效解（含极大负值）。

    Args:
        file_path: Pareto 结果 JSON 文件路径
        objective_names: 目标名称列表
        objective_directions: 目标方向列表
        save_path: 图片保存路径，为 None 时调用 plt.show()

    Returns:
        保存的文件路径，未保存则返回 None
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrix = np.array(data["archive_total"])
    # 过滤无效解
    valid = ~np.any(matrix < -1e100, axis=1)
    matrix = matrix[valid]
    if matrix.shape[0] == 0:
        print("无有效 Pareto 解，无法绘图")
        return None

    n_obj = matrix.shape[1]

    # 从元数据获取目标名称
    if objective_names is None:
        metadata = data.get("metadata", {})
        objective_names = metadata.get("objective_names", [f"目标{i+1}" for i in range(n_obj)])
    if objective_directions is None:
        metadata = data.get("metadata", {})
        objective_directions = metadata.get("objective_directions", ["max"] * n_obj)

    # 确保名称列表长度匹配
    if len(objective_names) < n_obj:
        objective_names = list(objective_names) + [f"目标{i+1}" for i in range(len(objective_names), n_obj)]

    if n_obj == 2:
        fig = _plot_2d(matrix, objective_names)
    elif n_obj == 3:
        fig = _plot_3d(matrix, objective_names)
    elif n_obj >= 4:
        # 只取前4个决策目标绘图（排除水量平衡约束目标）
        decision_names = [n for n in objective_names if n != "水量平衡"]
        decision_dirs = [d for n, d in zip(objective_names, objective_directions) if n != "水量平衡"]
        # 从矩阵中也排除水量平衡列
        balance_idx = [i for i, n in enumerate(objective_names) if n == "水量平衡"]
        plot_matrix = np.delete(matrix, balance_idx, axis=1)
        fig = _plot_4d(plot_matrix, decision_names, decision_dirs)
    else:
        print(f"不支持 {n_obj} 目标的可视化")
        return None

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()
    plt.close(fig)
    return None


def _plot_2d(matrix: np.ndarray, names: list[str]) -> plt.Figure:
    """2 目标散点图。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(matrix[:, 0], matrix[:, 1], c=matrix[:, 0], marker="^")
    ax.set_xlabel(names[0], fontsize=14)
    ax.set_ylabel(names[1], fontsize=14)
    fig.colorbar(sc, ax=ax).set_label(names[0])
    return fig


def _plot_3d(matrix: np.ndarray, names: list[str]) -> plt.Figure:
    """3 目标散点图。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    s = ax.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], marker="^", c=matrix[:, 0])
    ax.set_xlabel(names[0], fontsize=18, labelpad=15)
    ax.set_ylabel(names[1], fontsize=18, labelpad=15)
    ax.set_zlabel(names[2], fontsize=18, labelpad=15)
    cb = fig.colorbar(s, shrink=0.8)
    cb.set_label(label=names[0], fontsize=18)
    return fig


def _plot_4d(matrix: np.ndarray, names: list[str], directions: list[str]) -> plt.Figure:
    """4 目标散点图。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    # 映射: x=出境水量[3], y=地下水回补[1], z=水面面积[2], color=通水天数[0]
    x = matrix[:, 3]
    y = matrix[:, 1]
    z = matrix[:, 2]
    c = matrix[:, 0]

    # 智能单位缩放
    x_scale, x_label = _smart_scale(x, names[3])
    y_scale, y_label = _smart_scale(y, names[1])
    z_scale, z_label = _smart_scale(z, names[2])

    s = ax.scatter(x * x_scale, y * y_scale, z * z_scale, marker="^", c=c, cmap="viridis")
    ax.set_xlabel(x_label, fontsize=18, labelpad=15)
    ax.set_ylabel(y_label, fontsize=18, labelpad=15)
    ax.set_zlabel(z_label, fontsize=18, labelpad=15)
    ax.xaxis.set_major_locator(MaxNLocator(6))

    cb = fig.colorbar(s, shrink=0.8)
    cb.set_label(label=names[0], fontsize=18)
    cb.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.zaxis.set_tick_params(labelsize=18)
    return fig


def _smart_scale(values: np.ndarray, name: str) -> tuple[float, str]:
    """根据数值量级自动选择缩放因子和标签。"""
    vmax = np.max(np.abs(values))
    if vmax > 1e7:
        return 1e-6, f"{name}(百万m³)"
    elif vmax > 1e4:
        return 1e-3, f"{name}(千m³)"
    elif vmax > 1e2:
        return 1.0, f"{name}(ha)"
    else:
        return 1.0, name


def plot_schedule_comparison(
    pareto_file: str,
    ranking_file: str,
    top_n: int = 5,
    save_path: str | None = None,
) -> str | None:
    """绘制最佳方案调度过程时序图。

    Args:
        pareto_file: Pareto 结果 JSON 文件路径
        ranking_file: 排序结果 JSON 文件路径
        top_n: 显示排名前 N 的方案
        save_path: 图片保存路径，为 None 时调用 plt.show()

    Returns:
        保存的文件路径，未保存则返回 None
    """
    with open(pareto_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    archive_population = np.array(data["archive_population_total"])

    with open(ranking_file, "r", encoding="utf-8") as f:
        g_data = json.load(f)
    g = np.array(g_data["g"])

    # 从元数据获取目标名称
    metadata = data.get("metadata", {})
    obj_names = metadata.get("objective_names", None)

    # 取排名前 N 的方案
    top_idx = np.argsort(g)[::-1][:top_n]
    schedules = archive_population[top_idx]
    scores = g[top_idx]

    # 生成标签（排除水量平衡约束目标，仅使用决策目标）
    decision_names = [n for n in obj_names if n != "水量平衡"] if obj_names else None
    if decision_names and len(decision_names) >= top_n:
        default_labels = [f"{name}优先" for name in decision_names[:top_n]]
    else:
        default_labels = [
            "全线通水时长优先",
            "地下水回补优先",
            "水面面积优先",
            "出境水量控制优先",
            "均衡控制",
        ]
    colors = ["blue", "orange", "green", "deeppink", "m"]

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(len(schedules)):
        label = default_labels[i] if i < len(default_labels) else f"方案{i+1}"
        color = colors[i % len(colors)]
        ax.plot(
            range(len(schedules[i])),
            schedules[i],
            label=f"{label} ({scores[i]:.4f})",
            linestyle="-",
            color=color,
            marker="o",
            markevery=5,
            markersize=6,
        )

    # 水量验证标注
    total_water = metadata.get("total_water", None)
    if total_water:
        for i, idx in enumerate(top_idx):
            actual = np.sum(schedules[i]) * 24 * 60 * 60
            ratio = actual / total_water
            label = default_labels[i] if i < len(default_labels) else f"方案{i+1}"
            print(f"  {label}: 实际水量={actual:.0f}m³, 目标={total_water:.0f}m³, 比率={ratio:.2%}")

    ax.legend(fontsize=14, frameon=True)
    ax.set_xlabel("时间（天）", fontsize=16)
    ax.set_ylabel("流量（m³/s）", fontsize=16)
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    plt.show()
    plt.close(fig)
    return None