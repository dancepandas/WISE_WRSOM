import SMPSO_model
import Decision_optimization_model
import config
import visual_untils


#######SM模型参数#######
w_total = config.w_total
# 调度流量边界
min_q = config.min_q
max_q = config.max_q
# 每次次生成调度方案个数
dim = config.dim
# 调度方案天数
m = config.m
# 外循环最大迭代次数
max_p_out = config.max_p_out
# 内循环最大迭代次数
max_p = config.max_p
#模型最大更新速度
velocity_max=config.velocity_max
#模型最大更新速度
velocity_min=config.velocity_min

#######马斯京根模型参数#######
k = config.k

x = config.x

t = config.t

cost_cofficient_list = config.cost_cofficient_list

surface_cofficient_list = config.surface_cofficient_list

calculation_river_special_lose_water = config.calculation_river_special_lose_water

or_q_list = config.or_q_list

excel_path = config.excel_path


#######决策者主观权重#######
wz=config.wz


SM_model=SMPSO_model.SmpsoModel(dim,m,w_total,min_q,max_q,velocity_max,velocity_min,max_p,max_p_out)
SM_model.call(k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path, calculation_river_special_lose_water, or_q_list, objective_function_dim=4)

DO_model=Decision_optimization_model.Decision_optimization_model(wz)
g = DO_model.call()

#######结果展示#######
visual_untils.visualization_3D(config.SM_save_file)
visual_untils.visualization_linear(config.SM_save_file,config.DO_save_file)


