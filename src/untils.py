import numpy as np
import pandas as pd


def infiltration_water_volum(upstream_flow_process,cost_cofficient,calculation_river_rank):
    upstream_flow_process_cost = np.zeros((upstream_flow_process.shape[0], upstream_flow_process.shape[1]))
    R = np.copy(cost_cofficient)
    Ry = np.array(R).T
    r0 = Ry[calculation_river_rank, 0]
    r1 = Ry[calculation_river_rank, 1]
    r2 = Ry[calculation_river_rank , 2]
    for c_i in range(upstream_flow_process_cost.shape[0]):
        for c_j in range(upstream_flow_process_cost.shape[1]):
            if upstream_flow_process[c_i, c_j] == 0:
                upstream_flow_process_cost[c_i, c_j] = 0
            else:
                upstream_flow_process_cost[c_i, c_j] = r0 * np.square(upstream_flow_process[c_i, c_j]) + r1 * upstream_flow_process[c_i, c_j] + r2
            upstream_flow_process_cost[upstream_flow_process_cost<0]=0
    return upstream_flow_process_cost


def cofficient_calculation(k,x,t):
    cofficient=[]
    k_array=np.array(k)
    x_array=np.array(x)
    t_array=np.array(t)
    norm=k_array-k_array*x_array+0.5*t_array
    c0=  (0.5 * t_array - k_array* x_array) / norm
    c1 = (0.5 * t_array + k_array * x_array) / norm
    c2=  (k_array- k_array*x_array-0.5*t_array) / norm
    cofficient.append(c0)
    cofficient.append(c1)
    cofficient.append(c2)
    cofficient_result=np.array(cofficient).T
    target_first=[1,1,-1]
    for i in range(cofficient_result.shape[0]):
        target=cofficient_result[i]-target_first
        if all(target==0):
            cofficient_result[i,0]=1
            cofficient_result[i, 1] = 0
            cofficient_result[i, 2] = 0
    return cofficient_result



def muskingen_s(upstream_flow_process,k,x,t,calculation_river_rank,cost_cofficient,w,tributary_flow_process,or_q):
    upstream_flow_process_cost=infiltration_water_volum(upstream_flow_process,cost_cofficient,calculation_river_rank)
    upstream_flow_process_calculation_=upstream_flow_process-upstream_flow_process_cost
    upstream_flow_process_calculation=np.zeros((upstream_flow_process_calculation_.shape[0],upstream_flow_process_calculation_.shape[1]))
    if np.any(tributary_flow_process!=0):
        for up_i in range(upstream_flow_process_calculation_.shape[0]):
            for up_j in range(upstream_flow_process_calculation_.shape[1]):
                upstream_flow_process_calculation[up_i,up_j]= upstream_flow_process_calculation_[up_i,up_j]+tributary_flow_process[up_j]
                if upstream_flow_process_calculation[up_i,up_j]<=0:
                    upstream_flow_process_calculation[up_i,up_j]=0
    else:
        for up_is in range(upstream_flow_process_calculation_.shape[0]):
            for up_js in range(upstream_flow_process_calculation_.shape[1]):
                upstream_flow_process_calculation[up_is, up_js] = upstream_flow_process_calculation_[up_is, up_js]
    a=np.copy(upstream_flow_process_calculation)
    b=np.copy(upstream_flow_process_calculation)
    upstream_flow_process_calculation_cut_first=a[:,1:]
    upstream_flow_process_calculation_cut_last=b[:,:-1]
    downstream_flow_process=np.zeros((upstream_flow_process.shape[0],upstream_flow_process.shape[1]))
    downstream_flow_process[:,0]=or_q
    downstream_flow_process_correct = np.zeros((upstream_flow_process.shape[0], upstream_flow_process.shape[1]))
    downstream_flow_process_correct[:, 0] = or_q
    cofficient_result=cofficient_calculation(k,x,t)
    c0=cofficient_result[calculation_river_rank,0]
    c1=cofficient_result[calculation_river_rank,1]
    c2=cofficient_result[calculation_river_rank, 2]
    for i in range(upstream_flow_process_calculation_cut_first.shape[0]):
        for j in range(upstream_flow_process_calculation_cut_last.shape[1]):
            downstream_flow_process[i,j+1]=c0*upstream_flow_process_calculation_cut_first[i,j]+c1*upstream_flow_process_calculation_cut_last[i,j]+c2*downstream_flow_process[i,j]
            downstream_flow_process[downstream_flow_process < 0] = 0
            if np.sum(downstream_flow_process[i:, 0:j + 2])*24*60*60 <= w[calculation_river_rank]:
                downstream_flow_process_correct[i, j + 1]=0
            elif np.sum(downstream_flow_process[i:, 0:j + 2])*24*60*60 >w[calculation_river_rank] and np.sum(downstream_flow_process[i:, 0:j + 1])*24*60*60 <w[calculation_river_rank]:
                downstream_flow_process[i, j + 1]=(np.sum(downstream_flow_process[i:, 0:j + 2])*24*60*60 -w[calculation_river_rank])/(24*60*60)
            else:
                downstream_flow_process_correct[i,j+1]=downstream_flow_process[i,j+1]
            downstream_flow_process_correct[downstream_flow_process < 0] = 0
    return downstream_flow_process_correct



def water_surface_area_value(upstream_flow_process,calculation_river_rank,surface_cofficient):
    water_surface_area_value=np.zeros((upstream_flow_process.shape[0],upstream_flow_process.shape[1]))
    R=np.copy(surface_cofficient)
    Ry=np.array(R).T
    r0=Ry[calculation_river_rank,0]
    r1 = Ry[calculation_river_rank, 1]
    r2 = Ry[calculation_river_rank, 2]
    for i in range(water_surface_area_value.shape[0]):
        for j in range(water_surface_area_value.shape[1]):
            if upstream_flow_process[i,j]==0:
                water_surface_area_value[i, j] =0
            else:
                water_surface_area_value[i,j]=r0*np.square(upstream_flow_process[i,j])+r1*upstream_flow_process[i,j]+r2
    return water_surface_area_value


def summation(summation_list):
    summation_list_arry=np.array(summation_list)
    summation_total_first=np.zeros((len(summation_list_arry),len(summation_list_arry[0])))
    for t_i in range(len(summation_list_arry)):
        for t_j in range(len(summation_list_arry[t_i])):
            summation_total_first[t_i,t_j]=sum(summation_list_arry[t_i,t_j])*24*60*60
    summation_total_final=np.zeros(len(summation_list_arry[0]))
    for k_i in range(summation_total_first.shape[1]):
        summation_total_final[k_i]=round(sum(summation_total_first[:,k_i]),2)
    return summation_total_final


def summation_max(summation_list):
    summation_list_arry=np.array(summation_list).T
    summation_total_first=np.zeros((len(summation_list_arry),len(summation_list_arry[0])))
    for t_i in range(len(summation_list_arry)):
        for t_j in range(len(summation_list_arry[t_i])):
            summation_total_first[t_i,t_j]=sum(summation_list_arry[t_i,t_j])
    summation_total_final=np.zeros(len(summation_list_arry[0]))
    for k_i in range(summation_total_first.shape[1]):
        summation_total_final[k_i]=round(max(summation_total_first[:,k_i]),2)
    return summation_total_final,summation_total_first


def summation_any_cross(downstream_flow_process_list,calculation_cross_rank=-1):
    a=np.copy(downstream_flow_process_list)
    a=np.array(a)
    outflow_water=np.zeros(len(a[calculation_cross_rank]))
    for i in range(outflow_water.shape[0]):
        outflow_water[i]=round(sum(a[calculation_cross_rank,i])*24*60*60,2)
    return outflow_water


def calculation_time(downstream_flow_process_list):
    infinity = 1e+11
    a_t=np.copy(downstream_flow_process_list)
    a=np.array(a_t)
    time=[]
    for i in range(a.shape[1]):
        total_days=0
        for j in range(a[0].shape[1]):
            martix=a[:,i,j]
            if all(martix>0):
                total_days+=1
        time.append(total_days)
    return time


def muskingen(k:list,x:list,t:list,cost_cofficient_list:list,surface_cofficient_list:list,excel_path:str,calculation_river_special_lose_water:list,or_q_list:list,population):
    upstream_flow_process_cost_list = []
    water_surface_area_value_list = []
    downstream_flow_process_list = []

    cost_cofficient = np.array(cost_cofficient_list)
    surface_cofficient = np.array(surface_cofficient_list)

    df_a = pd.read_excel(excel_path)
    qingbaikou = df_a[df_a.columns[0]].values.tolist()
    yong_yin = df_a[df_a.columns[1]].values.tolist()
    xiao_hong_men = df_a[df_a.columns[2]].values.tolist()
    daxing = df_a[df_a.columns[3]].values.tolist()
    nan_shui = df_a[df_a.columns[4]].values.tolist()

    upstream_flow_process = population
    downstream_flow_process_list.append(list(upstream_flow_process))
    tributary_flow_process_list = list(np.zeros((9, len(qingbaikou))))
    tributary_flow_process_list[1] = np.array(qingbaikou)
    tributary_flow_process_list[4] = np.array(yong_yin) * (-1)
    tributary_flow_process_list[5] = np.array(xiao_hong_men)
    tributary_flow_process_list[6] = np.array(nan_shui) + np.array(daxing) * (-1)
    tributary_flow_process_list = np.array(tributary_flow_process_list)
    for G_m in range(len(k)):
        calculation_river_rank = G_m
        # Tributary_flow_process should be a quantity that varies with calculation_river_rank
        or_q = or_q_list[G_m]
        tributary_flow_process = tributary_flow_process_list[calculation_river_rank]
        downstream_flow_process = muskingen_s(upstream_flow_process, k, x, t, calculation_river_rank,cost_cofficient, calculation_river_special_lose_water,tributary_flow_process, or_q)
        downstream_flow_process_list.append(list(downstream_flow_process))
        upstream_flow_process_cost = infiltration_water_volum(upstream_flow_process, cost_cofficient,calculation_river_rank)
        upstream_flow_process_cost_list.append(list(upstream_flow_process_cost))
        water_surface_area_value_c = water_surface_area_value(upstream_flow_process, calculation_river_rank,surface_cofficient)
        water_surface_area_value_list.append(list(water_surface_area_value_c))
        upstream_flow_process = downstream_flow_process
    time = calculation_time(downstream_flow_process_list)
    outflow = summation_any_cross(downstream_flow_process_list)
    water_surface_area_value_max = summation_max(water_surface_area_value_list)[0]
    upstream_flow_process_cost_total = summation(upstream_flow_process_cost_list)
    return time,outflow,water_surface_area_value_max,upstream_flow_process_cost_total,downstream_flow_process_list


def update_archive_efficient(archive_total, archive_population_total, archive, archive_population):

    if len(archive_total)==0:
        for a_i in range(len(archive)):
            archive_total.append(archive[a_i])
            archive_population_total.append(archive_population[a_i])
    else:
        archive_total_np = np.array(archive_total)
        archive_np = np.array(archive)
        existing_tuples = set(tuple(map(tuple, archive_total_np)))
        for k in range(len(archive_np)):
            b = archive_np[k]
            b_tuple = tuple(b)

            if b_tuple  in existing_tuples:
                continue

            dominated = np.any(
                np.all(archive_total_np >= b, axis=1) &
                np.any(archive_total_np > b, axis=1)
            )

            if dominated:
                continue

            dominates_mask = (
                    np.all(b >= archive_total_np, axis=1) &
                    np.any(b > archive_total_np, axis=1)
            )

            to_keep = ~dominates_mask
            archive_total_np = archive_total_np[to_keep]

            archive_population_total = [list(pop) for pop, keep in zip(archive_population_total, to_keep) if keep]
            existing_tuples = {tuple(vec) for vec in archive_total_np}
            archive_total_np = np.vstack([archive_total_np, b]) if archive_total_np.size else np.array([b])
            archive_population_total.append(list(archive_population[k]))
            existing_tuples.add(b_tuple)
        archive_total[:] = archive_total_np.tolist()
    return archive_total,archive_population_total