
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
import json

mpl.rcParams['font.sans-serif'] = ['SimSun']# 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def visualization_3D(file_path):
    with open(file_path, "r") as r_file:
        data = json.load(r_file)
    archive_total = data['archive_total']
    matrix=np.array(archive_total)
    if matrix.shape[1] == 3:
        # 3目标：直接绘制x/y/z
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(projection='3d')
        x = matrix[:, 0]
        y = matrix[:, 1]
        z = matrix[:, 2]
        s=ax1.scatter(x,y,z,marker='^',c=x)
        ax1.set_xlabel('目标1', fontsize=18, labelpad=15)
        ax1.set_ylabel('目标2', fontsize=18, labelpad=15)
        ax1.set_zlabel('目标3', fontsize=18, labelpad=15)
        cb=fig.colorbar(s,shrink=0.8)
        cb.set_label(label='目标1',fontsize=18)
        plt.show()
        return

    if matrix.shape[1] >= 4:
        # 4目标：沿用原语义映射
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(projection='3d')
        y = matrix[:, 1]/1000000
        z = matrix[:, 2]
        x = (1/matrix[:, 3])/1000000
        r = matrix[:, 0]
        s=ax1.scatter(x,y,z,marker='^',c=r)
        ax1.set_ylabel('地下水回补水量(百万m³)', fontsize=18, labelpad=15)
        ax1.set_zlabel('水面面积(ha)', fontsize=18, labelpad=15)
        ax1.set_xlabel('出境水量(百万m³)', fontsize=18, labelpad=15)
        ax1.xaxis.set_major_locator(MaxNLocator(6))
        cb=fig.colorbar(s,shrink=0.8)
        cb.set_label(label='全线通水时长(天)',fontsize=18)
        cb.ax.tick_params(labelsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax = plt.gca()
        ax.zaxis.set_tick_params(labelsize=18)
        plt.show()
        return

    # 2目标：改为二维散点
    if matrix.shape[1] == 2:
        plt.figure(figsize=(8,6))
        plt.scatter(matrix[:,0], matrix[:,1], c=matrix[:,0], marker='^')
        plt.xlabel('目标1')
        plt.ylabel('目标2')
        plt.colorbar().set_label('目标1')
        plt.show()
        return


def fun1(SM_file_path,DO_file_path):
    with open(SM_file_path, "r") as r_file:
        data = json.load(r_file)
    archive_total = data['archive_total']
    archive_population_total = data['archive_population_total']

    with open(DO_file_path, "r") as r_file:
        data = json.load(r_file)
    g = data['g']

    archive_result = np.array(archive_total)
    archive_result[:, -1] = 1 / archive_result[:, -1]
    arg_g_idx = np.argsort(g)[::-1]
    arg_g_n = arg_g_idx[:10]
    archive_population_result = np.array(archive_population_total)
    archive_result[:, -1] = 1 / archive_result[:, -1]
    result_value_programme = archive_result[arg_g_n]
    result_process_programme = archive_population_result[arg_g_n]
    return result_process_programme,result_value_programme



def fun3(matrix_population_list):
    fig1=plt.figure(figsize=(12,8))
    matrix_population=np.array(matrix_population_list)
    plt.plot(range(len(matrix_population[0])),matrix_population[0],label='全线通水时长优先',linestyle='-',color='blue',marker='o',markevery=5,markersize=8)
    plt.plot(range(len(matrix_population[0])),matrix_population[1],label='地下水回补优先',linestyle='-',color='orange',marker='o',markevery=5,markersize=8)
    plt.plot(range(len(matrix_population[0])),matrix_population[2],label='水面面积优先',linestyle='-',color='green',marker='o',markevery=5,markersize=8)
    plt.plot(range(len(matrix_population[0])),matrix_population[3],label='出境水量控制优先',linestyle='-',color='deeppink',marker='o',markevery=5,markersize=8)
    plt.plot(range(len(matrix_population[0])),matrix_population[4],label='均衡控制',linestyle='-',color='m',marker='o',markevery=5,markersize=8)
    plt.legend(loc='lower center', fontsize=20, frameon=False, ncol=2, bbox_to_anchor=(0.38, 0.7))
    plt.xlabel('时间（天）')
    plt.ylabel('流量（m³/s）')
    plt.xticks(range(0,50,10),fontsize=25)
    plt.yticks(fontsize=25)
    ax=plt.gca()
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    bwith=2
    ax.spines['bottom'].set_linewidth(bwith)#图框下边
    ax.spines['left'].set_linewidth(bwith)#图框左边
    ax.spines['top'].set_linewidth(bwith)#图框上边
    ax.spines['right'].set_linewidth(bwith)#图框右边
    plt.subplots_adjust( bottom=0.15)
    plt.show()
    return

def visualization_linear(SM_file_path,DO_file_path,rank=0):
    result_process_programme, result_value_programme = fun1(SM_file_path,DO_file_path)
    fun3(result_process_programme)
