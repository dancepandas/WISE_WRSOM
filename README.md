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

### 其他命令
```bash
# 仅优化
python -m wise_wrsom optimize --config config.yaml

# 仅排序
python -m wise_wrsom rank --input data/SM_model_result.json

# 仅可视化
python -m wise_wrsom visualize --input data/SM_model_result.json --type pareto

# 列出可用算法
python -m wise_wrsom list-optimizers

# 列出可用目标
python -m wise_wrsom list-objectives

# 生成默认配置
python -m wise_wrsom init-config --output config.yaml
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
2. 实现 `ObjectiveFunction` 协议（`name`, `direction`, `compute`, `apply_penalty`）
3. 用 `@register("name")` 装饰器注册
4. 在 pipeline 中启用

### 添加新优化算法
1. 在 `wise_wrsom/optimizers/` 下创建新文件
2. 实现 `optimize()` 方法，接受标准参数
3. 用 `@register("name")` 装饰器注册
4. 在 `config.yaml` 中指定算法名称

### 添加新河道模型
1. 实现 `RoutingModel` 协议的 `compute()` 方法
2. 返回 `RoutingResult`（water_duration, outflow, surface_area, infiltration, downstream_flows）
3. 在 Pipeline 构造时注入