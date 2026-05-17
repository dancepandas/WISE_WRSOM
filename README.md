# WISE-WRSOM 水资源优化调度系统

## 项目简介

WISE-WRSOM（Water Resources Optimal Scheduling Model）是基于多目标进化算法的水资源优化调度系统。系统通过 MOEA/D、NSGA-III、SMPSO 等算法生成 Pareto 最优解集，再经 TOPSIS 决策排序，为水资源调度提供科学决策支持。

## 优化目标

| 目标 | 方向 | 说明 |
|------|------|------|
| 全线通水时长 | max | 所有河段均有水的天数 |
| 地下水回补水量 | max | 通过渗漏补给地下水的水量 |
| 水面面积 | max | 河道水面面积（改善生态） |
| 出境水量控制 | min | 最小化出境流量 |
| 水量平衡 | min | 调度总水量与目标水量的偏差（约束目标） |

水量平衡作为第5个约束目标参与优化，确保调度方案总水量在目标水量 ±10% 范围内。

## 系统架构

```
WISE_WRSOM/
├── wise_wrsom/                     # 核心包
│   ├── cli.py                      # CLI 命令行入口
│   ├── config.py                   # 配置管理（YAML加载）
│   ├── pipeline.py                 # 流水线编排（优化→决策→保存→可视化）
│   ├── protocols/                  # 协议接口定义
│   │   ├── objective.py            # 目标函数协议
│   │   ├── optimizer.py            # 优化算法协议
│   │   ├── routing.py              # 河道计算协议
│   │   └── decision.py             # 决策模型协议
│   ├── objectives/                 # 目标函数实现
│   │   ├── water_duration.py       # 全线通水时长
│   │   ├── groundwater.py          # 地下水回补
│   │   ├── surface_area.py         # 水面面积
│   │   ├── outflow_control.py      # 出境水量控制
│   │   ├── water_balance.py        # 水量平衡约束
│   │   ├── penalty.py              # 罚函数计算
│   │   └── registry.py             # 目标注册表
│   ├── optimizers/                  # 优化算法实现
│   │   ├── moead.py                # MOEA/D 算法
│   │   ├── nsga3.py                # NSGA-III 算法
│   │   ├── smpso.py                # SMPSO 算法
│   │   ├── segment.py              # 分段常数编码与遗传算子
│   │   ├── base.py                 # 公共组件（初始化、归档、罚函数）
│   │   └── registry.py             # 算法注册表
│   ├── routing/                    # 河道流量计算
│   │   └── muskingum.py            # 马斯京根演进模型
│   ├── decision/                   # 决策排序
│   │   └── topsis.py               # TOPSIS 方法
│   ├── visualization/              # 可视化
│   │   └── plots.py                # Pareto前沿图、调度时序图
│   └── utils.py                    # 工具函数
├── data/                           # 数据文件
│   ├── water_divide.xlsx           # 分水口数据
│   ├── SM_model_result.json        # Pareto优化结果
│   ├── DO_g_result.json            # TOPSIS排序结果
│   ├── pareto_front.png            # Pareto前沿图
│   └── schedule.png                # 调度过程图
├── tests/                          # 测试
├── config.yaml                     # 配置文件
├── requirements.txt                # 依赖
└── README.md
```

## 快速开始

### 环境要求
- Python 3.10+

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行完整流水线
```bash
python -m wise_wrsom run --config config.yaml
```

更多命令参见 [CLI 命令参考](#cli-命令参考)。

## CLI 命令参考

所有命令支持 `--format json` 输出机器可解析结果，便于 Agent 系统调用。

### `run` — 运行完整流水线

优化 → 决策排序 → 保存 → 可视化一步完成。

```bash
python -m wise_wrsom run -c config.yaml -a nsga3 -n 100 -p 100
```

| 选项 | 说明 |
|------|------|
| `-c, --config` | 配置文件路径 |
| `-a, --algorithm` | 优化算法（auto/smpso/nsga3/moead） |
| `-n, --iterations` | 最大迭代次数 |
| `-p, --population` | 种群大小 |
| `-o, --objectives` | 目标函数列表（逗号分隔） |
| `--velocity-max` | SMPSO 粒子最大速度 |
| `--velocity-min` | SMPSO 粒子最小速度 |
| `--crossover-rate` | NSGA-III/MOEA/D 交叉概率 |
| `--crossover-eta` | NSGA-III 交叉分布指数 |
| `--n-reference-divisions` | NSGA-III 参考点分割数 |
| `--n-neighbors` | MOEA/D 邻域大小 |
| `--mutation-rate` | 变异概率 |
| `--mutation-eta` | 变异分布指数 |

### `optimize` — 仅执行多目标优化

```bash
python -m wise_wrsom optimize -a smpso --mutation-rate 0.2 --velocity-max 3.0
```

选项与 `run` 相同，额外支持 `--output` 指定输出文件路径。

### `rank` — 对 Pareto 解进行决策排序

```bash
python -m wise_wrsom rank -i data/SM_model_result.json -w 0.3,0.2,0.3,0.2
```

| 选项 | 说明 |
|------|------|
| `-i, --input` | Pareto 结果文件路径 |
| `-w, --weights` | 主观权重（逗号分隔） |
| `--output` | 输出文件路径 |

### `visualize` — 可视化结果

```bash
python -m wise_wrsom visualize -i data/SM_model_result.json -t pareto
python -m wise_wrsom visualize -i data/SM_model_result.json -t schedule -r data/DO_g_result.json
```

| 选项 | 说明 |
|------|------|
| `-i, --input` | Pareto 结果文件路径 |
| `-t, --type` | 图表类型（pareto / schedule） |
| `-r, --ranking` | 排序结果文件路径（schedule 类型需要） |
| `--output` | 图片保存路径 |

### `best` — 查看排名第一的最优方案

```bash
python -m wise_wrsom best -i data/SM_model_result.json -r data/DO_g_result.json
```

### `export` — 导出前 N 名方案

```bash
python -m wise_wrsom export -i data/SM_model_result.json -r data/DO_g_result.json -n 5
```

### `list-optimizers` / `list-objectives` / `list-routing` — 列出可用组件

```bash
python -m wise_wrsom list-optimizers
python -m wise_wrsom list-objectives
python -m wise_wrsom list-routing
```

### `init-config` — 生成默认配置文件

```bash
python -m wise_wrsom init-config -o config.yaml
```

## 配置说明

配置通过 `config.yaml` 管理：

```yaml
muskingum:
  k: [0, 0, 0, 0, 13, 16, 16, 80, 150]    # 蓄量常数
  x: [0, 0, 0, 0, 0, 0, 0, 0, 0.2]         # 楔形储量权重
  t: [24, 24, 24, 24, 24, 24, 24, 24, 24]   # 时间步长(h)
  cost_coefficients: [...]                    # 渗漏损失经验方程系数
  surface_coefficients: [...]                 # 水面面积经验方程系数
  special_lose_water: [...]                   # 特殊蓄水水量
  initial_flow: [...]                         # 初始流量
  tributary_data_path: data/water_divide.xlsx

optimizer:
  population_size: 100                        # 种群大小
  scheduling_days: 80                         # 调度天数
  max_iterations_outer: 20                    # 外循环迭代次数
  flow_min: 3.0                               # 最小流量(m³/s)
  flow_max: 100.0                             # 最大流量(m³/s)
  total_water: 150000000                      # 总调度水量(m³)
  algorithm: auto                             # auto/smpso/nsga3/moead
  # SMPSO 专用
  velocity_max: 5.0                           # 粒子最大速度
  velocity_min: -5.0                          # 粒子最小速度
  # NSGA-III / MOEA/D 共用
  crossover_rate: 0.9                         # 交叉概率
  crossover_eta: 20.0                         # 交叉分布指数（仅 NSGA-III）
  # NSGA-III 专用
  n_reference_divisions: 12                   # 参考点分割数
  # MOEA/D 专用
  n_neighbors: 10                             # 邻域大小
  # 共用
  mutation_rate: 0.1                          # 变异概率
  mutation_eta: 20.0                          # 变异分布指数

decision:
  subjective_weights: [0.25, 0.25, 0.25, 0.25]  # 4个决策目标权重
  method: topsis
```

`algorithm: auto` 时根据目标数量自动选择：≤3目标用 NSGA-III，≥4目标用 MOEA/D。

## 水量平衡机制

系统通过三层机制确保调度总水量接近目标：

1. **分段编码 normalize()**：变异/交叉后等比缩放流量至目标水量附近（±5%随机偏差），clip后迭代修正直到偏差<10%
2. **加法罚函数**：偏差超出±10%容忍范围时，二次增长惩罚附加到各目标值上，不会像乘法罚函数那样在大偏差时目标归零
3. **水量平衡独立目标**：作为第5个min方向目标参与多目标优化，避免被权重向量掩盖

## 扩展指南

### 添加新目标函数
1. 在 `wise_wrsom/objectives/` 下创建新文件
2. 实现 `ObjectiveFunction` 接口（`name`, `direction`, `compute`, `apply_penalty`）
3. 用 `@register("name")` 装饰器注册
4. 在 pipeline 中启用

### 添加新优化算法
1. 在 `wise_wrsom/optimizers/` 下创建新文件
2. 实现 `optimize()` 方法，接受标准参数
3. 用 `@register("name")` 装饰器注册
4. 在 `config.yaml` 中指定算法名称

### 添加新河道模型
1. 实现 `RoutingModel` 接口的 `compute()` 方法
2. 返回 `RoutingResult`（water_duration, outflow, surface_area, infiltration, downstream_flows）
3. 在 Pipeline 构造时注入