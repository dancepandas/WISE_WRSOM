# WISE-WRSOM 水资源优化调度系统

## 项目简介

WISE-WRSOM（Water Resources Optimal Scheduling Model）是一个基于智能优化算法的水资源调度系统。该系统结合了改进的多目标粒子群优化算法（SMPSO）和决策优化模型，用于解决复杂的水资源调度问题。

## 主要功能

### 核心算法
- **SMPSO模型**：基于多目标粒子群优化算法，用于生成Pareto最优解集
- **决策优化模型**：基于TOPSIS方法和熵权法的多目标决策分析
- **马斯京根模型**：河道洪水演进计算模型

### 优化目标
1. **全线通水时长最大化**：提高水系连通性
2. **地下水回补水量最大化**：增强地下水补给
3. **水面面积最大化**：改善生态环境
4. **出境水量控制**：确保下游用水需求

## 系统架构

```
WISE_WRSOM/
├── src/                          # 核心源代码
│   ├── model_mian.py            # 主程序入口
│   ├── SMPSO_model.py           # SMPSO优化算法
│   ├── Decision_optimization_model.py  # 决策优化模型
│   └── untils.py                # 工具函数
├── data/                        # 数据文件
│   ├── water_divide.xlsx        # 分水口分水数据
│   ├── SM_model_result.json     # SMPSO模型结果
│   └── DO_g_result.json         # 决策优化结果
├── config.py                    # 配置参数
├── visual_untils.py             # 可视化工具
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目说明
```

## 快速开始

### 环境要求
- Python 3.7+
- 推荐使用虚拟环境

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行项目
```bash
cd src
python model_mian.py
```

## 配置说明

### 主要参数配置（config.py）

#### SMPSO模型参数
- `w_total`: 总调度水量（m³）
- `dim`: 种群大小
- `m`: 调度方案天数
- `max_p_out`: 外循环最大迭代次数
- `max_p`: 内循环最大迭代次数
- `min_q`, `max_q`: 调度流量边界（m³/s）

#### 马斯京根模型参数
- `k`: 蓄量常数
- `x`: 楔形储量权重系数
- `t`: 时间步长（小时）

#### 决策权重
- `wz`: 决策者主观权重 [时长权重, 回补权重, 面积权重, 出境水量权重]

## 功能特性

### 1. 多目标优化
- 采用SMPSO算法处理多目标优化问题
- 生成Pareto最优解集
- 支持约束条件处理

### 2. 决策支持
- 基于TOPSIS的多属性决策方法
- 结合主观权重和客观权重
- 提供方案排序和评价

### 3. 水力计算
- 马斯京根法河道演进计算
- 考虑渗透损失和蒸发损失
- 支持支流汇入计算

### 4. 结果可视化
- 3D散点图展示Pareto前沿
- 调度过程时序图
- 多方案对比分析

### 5. 高性能计算
- 支持Apache Spark分布式计算
- 自动降级到本地计算
- 优化的向量化操作

## 使用示例

### 基本使用
```python
import SMPSO_model
import Decision_optimization_model
import config

# 创建SMPSO模型
SM_model = SMPSO_model.SmpsoModel(
    dim=config.dim,
    m=config.m,
    w_total=config.w_total,
    min_q=config.min_q,
    max_q=config.max_q,
    velocity_max=config.velocity_max,
    velocity_min=config.velocity_min,
    max_p=config.max_p,
    max_p_out=config.max_p_out
)

# 运行优化
SM_model.call(
    k=config.k,
    x=config.x,
    t=config.t,
    cost_cofficient_list=config.cost_cofficient_list,
    surface_cofficient_list=config.surface_cofficient_list,
    excel_path=config.excel_path,
    calculation_river_special_lose_water=config.calculation_river_special_lose_water,
    or_q_list=config.or_q_list,
    objective_function_dim=4
)

# 决策优化
DO_model = Decision_optimization_model.Decision_optimization_model(config.wz)
g = DO_model.call()
```

### 结果可视化
```python
import visual_untils

# 3D可视化Pareto前沿
visual_untils.visualization_3D(config.SM_save_file)

# 调度方案对比
visual_untils.visualization_linear(config.SM_save_file, config.DO_save_file)
```

## 数据格式

### 输入数据
1. **分水口数据** (water_divide.xlsx)
   - 青白口分水量
   - 永引渠分水量
   - 小红门分水量
   - 大兴分水量
   - 南水分水量

### 输出数据
1. **SMPSO结果** (SM_model_result.json)
   ```json
   {
     "archive_total": [[obj1, obj2, obj3, obj4], ...],
     "archive_population_total": [[q1, q2, ..., qm], ...]
   }
   ```

2. **决策优化结果** (DO_g_result.json)
   ```json
   {
     "g": [score1, score2, ..., scoreN]
   }
   ```

## 算法说明

### SMPSO算法
1. **初始化**：生成满足约束的初始种群
2. **评价**：计算多目标函数值
3. **更新**：粒子位置和速度更新
4. **归档**：维护Pareto最优解集
5. **拥挤度**：保持解集多样性

### 决策优化
1. **标准化**：目标值归一化处理
2. **权重计算**：熵权法确定客观权重
3. **TOPSIS**：计算理想解相对贴近度
4. **排序**：生成方案优先级

## 性能优化

### Spark加速
- 自动检测Spark环境
- 并行化外循环计算
- 失败时自动降级

### 内存优化
- 向量化操作
- 高效的Pareto前沿维护
- 最小化数据复制

## 常见问题

### Q: 如何调整优化参数？
A: 修改config.py中的相关参数，主要包括种群大小、迭代次数、流量边界等。

### Q: 如何使用Spark加速？
A: 确保安装pyspark并正确设置JAVA_HOME环境变量，系统会自动检测并启用Spark。

### Q: 如何添加新的目标函数？
A: 在SMPSO_model.py的fitness方法中添加新的目标计算逻辑。

### Q: 结果文件过大怎么办？
A: 可以通过减少种群大小或调整归档策略来控制结果规模。

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)。

**重要说明**：
- ✅ 允许：学习、研究、教育用途
- ✅ 允许：非商业性质的修改和分发
- ❌ 禁止：任何形式的商业用途
- ❌ 禁止：未经授权的商业化应用

如需商业授权，请联系项目维护者。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件到Chs9710@163.com

## 更新日志

### v1.0.0
- 初始版本发布
- 实现SMPSO多目标优化算法
- 集成决策优化模型
- 支持Spark分布式计算
- 完善可视化功能

---

*注：本项目用于水资源优化调度研究，如需商业用途请联系项目维护者。*