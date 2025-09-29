
import numpy as np
import random
import pandas as pd
from untils import muskingen
import time
from tqdm import tqdm
import os
import sys
import traceback
try:
    from pyspark.sql import SparkSession
    _spark_import_error = None
except Exception as e:
    SparkSession = None
    _spark_import_error = e
import config
import json



class SmpsoModel:
    def __init__(self,population_dim,population_value_dim,W_total,Q_min,Q_max,velocity_max,velocity_min,Max_p,Max_p_out):
        self.population_dim=population_dim
        self.population_value_dim=population_value_dim
        self.W_total=W_total
        self.Q_min=Q_min
        self.Q_max=Q_max
        self.velocity_max=velocity_max
        self.velocity_min=velocity_min
        self.Max_p=Max_p
        self.Max_p_out=Max_p_out
        self.archive = []
        self.archive_population = []
        self.archive_total = []
        self.archive_population_total = []

    def POP_test(self):
        number_of_segments = np.random.randint(3, 6)
        v1 = np.random.dirichlet([random.uniform(1.2, 2.2) for _ in range(number_of_segments)], self.population_dim)
        number_of_days = np.round(self.population_value_dim * v1).astype(int)
        number_of_days = np.where(number_of_days == 0, 5, number_of_days)
        w_split = np.round(self.W_total * v1).astype(int)
        value_day = np.sum(number_of_days, axis=1) - self.population_value_dim
        value_w = np.sum(w_split, axis=1) - self.W_total

        for i in range(self.population_dim):

            if value_day[i] < 0:
                min_idx = np.argmin(number_of_days[i])
                number_of_days[i, min_idx] += -value_day[i]

            elif value_day[i] > 0:
                max_idx = np.argmax(number_of_days[i])
                number_of_days[i, max_idx] -= value_day[i]

            elif value_w[i] < 0:
                min_idx_w = np.argmin(w_split[i])
                w_split[i, min_idx_w] += -value_w[i]

            elif value_w[i] > 0:
                max_idx_w = np.argmax(w_split[i])
                w_split[i, max_idx_w] -= value_w[i]

        q_np = w_split / (24 * 60 * 60 * number_of_days)
        population = []
        for j in range(self.population_dim):
            pop = []
            for n in range(q_np.shape[1]):
                zp = [q_np[j, n] for _ in range(number_of_days[j, n])]
                pop = pop + zp
            population.append(pop)
        p_result = np.array(population)
        return p_result, number_of_days

    def creat_population_efficient(self):
        start_time = time.time()
        population, number_of_days = self.POP_test()
        while np.any(population) > self.Q_max:
            population, number_of_days = self.POP_test()
        end_time = time.time()
        time_total = end_time - start_time
        print(f'初始化种群计算时间：{time_total}')
        return population, number_of_days

    def creat_velocity(self,population):
        velocity_dim = population.shape[0]
        velocity_value_dim = population.shape[1]
        velocity = np.zeros((velocity_dim, velocity_value_dim), dtype=float)
        for i in range(velocity_dim):
            a = population[i]
            days = []
            Q_value = [a[0]]
            for k in range(len(a)):
                if a[k] not in Q_value:
                    days.append(k - sum(days))
                    Q_value.append(a[k])
            days.append(len(a) - sum(days))

            def list_velocity_value(days, velocity_max, velocity_min):
                velocity_value = list(np.random.randint(velocity_min, velocity_max, len(days) - 1))
                total = 0
                for v_i in range(len(velocity_value)):
                    total += velocity_value[v_i] * days[v_i]
                velocity_value.append((0 - total) // days[-1])
                return velocity_value

            velocity_value = list_velocity_value(days, self.velocity_max, self.velocity_min)
            while any(np.array(velocity_value) < self.velocity_min) or any(np.array(velocity_value) > self.velocity_max):
                velocity_value = list_velocity_value(days, self.velocity_max, self.velocity_min)
                if all(np.array(velocity_value) > self.velocity_min) and all(np.array(velocity_value) < self.velocity_max):
                    break
            velocity_list = []
            for v_k in range(len(velocity_value)):
                for v_j in range(days[v_k]):
                    velocity_list.append(velocity_value[v_k])
            for j in range(velocity_value_dim):
                velocity[i, j] = velocity_list[j]
        return velocity

    def update_velocity(self,population, velocity):
        start_time = time.time()
        archive_population = np.array(self.archive_population)
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        w = np.random.uniform(0.1, 0.5)
        c1 = np.random.uniform(1.5, 2.5)
        c2 = np.random.uniform(1.5, 2.5)
        if (c1 + c2) > 4:
            phi = c1 + c2
        else:
            phi = 0
        chi = 2 / (2 - phi - ((phi ** 2) - 4 * phi) ** (1 / 2))
        velocity_ = np.zeros((velocity.shape[0], velocity.shape[1]))
        crowding = self.crowding_distance_function()
        ind_2 = crowding.argmax()
        delta = (self.velocity_max - self.velocity_min) / 2
        if len(self.archive) > len(population):
            index = np.argsort(crowding)
            arg = index[:len(population)]
            pbest = archive_population[arg]
        else:
            pbest = population
        for i in range(velocity_.shape[0]):
            for j in range(velocity_.shape[1]):
                velocity_[i, j] = round((w * velocity[i, j] + c1 * r1 * (pbest[i, j] - population[i, j]) + c2 * r2 * (
                            archive_population[ind_2, j] - population[i, j])) * chi)
                velocity_[i, j] = np.clip(velocity_[i, j], -delta, delta)

        v_sum = np.sum(velocity_, axis=1).reshape(-1, 1)
        velocity_ = velocity_ - v_sum / velocity_.shape[1]
        end_time = time.time()
        time_total = end_time - start_time
        print(f'\n速度更新计算时间：{time_total}')
        return velocity_

    def update_population(self,population, velocity,number_of_days):
        q_total = self.W_total / (24 * 60 * 60)
        result = population + velocity
        result = np.where(result > self.Q_max, self.Q_max, result)
        result = np.where(result < self.Q_min, self.Q_min, result)
        result_list = result.tolist()
        population_new_list = []
        for i in range(result.shape[0]):
            pop = result_list[i]
            pop_new = []
            for j in range(len(number_of_days[i])):
                if j == 0:
                    pop_value = sum(pop[:number_of_days[i][j] + 1]) / number_of_days[i][j]
                    pop_new = [pop_value for _ in range(number_of_days[i][j])]
                else:
                    pop_value = sum(pop[number_of_days[i][j - 1]:number_of_days[i][j - 1] + number_of_days[i][j] + 1]) / \
                                number_of_days[i][j]
                    pop_new = pop_new + [pop_value for _ in range(number_of_days[i][j])]
            population_new_list.append(pop_new)
        population_new = np.array(population_new_list)
        population_new_sum = np.sum(population_new, axis=1).reshape(-1, 1)
        pq_g = (population_new_sum - q_total) / population_new.shape[1]
        population_new = population_new - pq_g
        population_new = np.where(population_new > self.Q_max, self.Q_max, population_new)
        population_new = np.where(population_new < self.Q_min, self.Q_min, population_new)
        return population_new

    def crowding_distance_function(self):
        start_time = time.time()
        infinity = 1e+11
        a = np.array(self.archive)
        if a.size == 0:
            return np.array([])
        if a.shape[0] <= 2:
            return np.full(a.shape[0], infinity)
        num_obj = a.shape[1]
        crowd = np.zeros(a.shape[0], dtype=float)
        for obj_idx in range(num_obj):
            order = np.argsort(a[:, obj_idx])
            sorted_vals = a[order, obj_idx]
            min_v = sorted_vals[0]
            max_v = sorted_vals[-1]
            denom = max(max_v - min_v, 1e-6)
            dist = np.zeros(a.shape[0])
            dist[order[0]] = infinity
            dist[order[-1]] = infinity
            for k in range(1, a.shape[0] - 1):
                prev_v = a[order[k - 1], obj_idx]
                next_v = a[order[k + 1], obj_idx]
                dist[order[k]] = (next_v - prev_v) / denom
            finite_mask = ~np.isinf(dist)
            crowd[finite_mask] += dist[finite_mask]
            crowd[~finite_mask] = infinity
        end_time = time.time()
        time_total = end_time - start_time
        print(f'\n拥挤度计算时间：{time_total}')
        return crowd


    def fitness(self,k: list, x: list, t: list, cost_cofficient_list: list, surface_cofficient_list: list, excel_path: str,
                calculation_river_special_lose_water: list, or_q_list: list, population,objective_function_dim=4):
        objective_value = np.zeros((population.shape[0], objective_function_dim), dtype=float)
        result = muskingen(k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path,
                           calculation_river_special_lose_water, or_q_list, population)
        for i in range(objective_value.shape[0]):
            downstream_flow_process_list = result[4]
            otq_total = np.sum(downstream_flow_process_list[0][i])
            orq_total = self.W_total / (24 * 60 * 60)
            sum_q_g = otq_total - orq_total
            p = np.exp(sum_q_g) - 1
            print(f'总流量差值==={sum_q_g}===')
            print(f'罚函数值==={p}===')
            objective_value[i, 0] = result[0][i] - p
            print('time', result[0][i] - p)
            objective_value[i, 1] = result[3][i] - p
            print('dxs', result[3][i] - p)
            objective_value[i, 2] = result[2][i] - p
            objective_value[i, 3] = (1 / result[1][i]) - p
        return objective_value

    def pareto_front_points_efficient(self, objective_value, population):
        start_time = time.time()
        if len(self.archive) == 0:
            self.archive.append(list(objective_value[0]))
            self.archive_population.append(list(population[0]))
        else:
            archive_np = np.array(self.archive)
            existing_tuples = set(tuple(map(tuple, archive_np)))
            for k in range(len(objective_value)):
                b = objective_value[k]
                b_tuple = tuple(b)

                if b_tuple in existing_tuples:
                    continue

                dominated = np.any(
                    np.all(archive_np >= b, axis=1) &
                    np.any(archive_np > b, axis=1)
                )
                if dominated:
                    continue

                dominates_mask = (
                        np.all(b >= archive_np, axis=1) &
                        np.any(b > archive_np, axis=1)
                )

                to_keep = ~dominates_mask
                archive_np = archive_np[to_keep]

                self.archive_population = [pop for pop, keep in zip(self.archive_population, to_keep) if keep]
                existing_tuples = {tuple(vec) for vec in archive_np}
                archive_np = np.vstack([archive_np, b]) if archive_np.size else np.array([b])
                self.archive_population.append(population[k])
                existing_tuples.add(b_tuple)
            self.archive[:] = archive_np.tolist()
        end_time = time.time()
        time_total = end_time - start_time
        print(f'内部档案更新计算时间：{time_total}')
        return self.archive, self.archive_population

    def update_archive_efficient(self):

        if len(self.archive_total) == 0:
            for a_i in range(len(self.archive)):
                self.archive_total.append(self.archive[a_i])
                self.archive_population_total.append(self.archive_population[a_i])
        else:
            archive_total_np = np.array(self.archive_total)
            archive_np = np.array(self.archive)
            existing_tuples = set(tuple(map(tuple, archive_total_np)))
            for k in range(len(archive_np)):
                b = archive_np[k]
                b_tuple = tuple(b)

                if b_tuple in existing_tuples:
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

                self.archive_population_total = [list(pop) for pop, keep in zip(self.archive_population_total, to_keep) if keep]
                existing_tuples = {tuple(vec) for vec in archive_total_np}
                archive_total_np = np.vstack([archive_total_np, b]) if archive_total_np.size else np.array([b])
                self.archive_population_total.append(list(self.archive_population[k]))
                existing_tuples.add(b_tuple)
            self.archive_total[:] = archive_total_np.tolist()
        return self.archive_total, self.archive_population_total

    def write_results_to_file(self):
        save_file_path=config.SM_save_file
        data = {'archive_total':self. archive_total, 'archive_population_total': self.archive_population_total}
        with open(save_file_path, "w") as write_file:
            json.dump(data, write_file)

    def _single_outer_iteration(self, k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path,
                                calculation_river_special_lose_water, or_q_list, objective_function_dim=4):
        population, number_of_days = self.creat_population_efficient()
        velocity = self.creat_velocity(population)
        objective_value = self.fitness(k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path,
                                       calculation_river_special_lose_water, or_q_list, population, objective_function_dim)
        self.archive, self.archive_population = self.pareto_front_points_efficient(objective_value, population)
        for P in tqdm(range(self.Max_p)):
            population = self.update_population(population, velocity, number_of_days)
            objective_value = self.fitness(k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path,
                                           calculation_river_special_lose_water, or_q_list, population, objective_function_dim)
            velocity = self.update_velocity(population, velocity)
            self.archive, self.archive_population = self.pareto_front_points_efficient(objective_value, population)
        return self.archive, self.archive_population


    def call(self, k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path,
             calculation_river_special_lose_water, or_q_list, objective_function_dim=4, use_spark=True, spark_num_slices=8):
        if use_spark and SparkSession is not None:
            try:
                print("[Spark] 尝试创建 SparkSession ...")
                print(f"[Spark] 环境变量: JAVA_HOME={os.environ.get('JAVA_HOME')}, SPARK_HOME={os.environ.get('SPARK_HOME')}, PYSPARK_PYTHON={os.environ.get('PYSPARK_PYTHON')}")
                # 避免外部 Spark 与 pip 安装的 pyspark 版本/类路径冲突（'JavaPackage' object is not callable 常见于此）
                restore_spark_home = None
                if os.environ.get("SPARK_HOME"):
                    restore_spark_home = os.environ.get("SPARK_HOME")
                    print(f"[Spark] 检测到 SPARK_HOME={restore_spark_home}，为避免版本冲突将临时忽略该环境变量")
                    os.environ.pop("SPARK_HOME", None)
                spark = (
                    SparkSession.builder
                    .appName("SMPSO_Optimization")
                    .master("local[*]")
                    .config("spark.ui.showConsoleProgress", "false")
                    .config("spark.driver.bindAddress", "127.0.0.1")
                    .config("spark.driver.host", "127.0.0.1")
                    .config("spark.pyspark.python", sys.executable)
                    .config("spark.pyspark.driver.python", sys.executable)
                    .getOrCreate()
                )
                sc = spark.sparkContext
                print(f"[Spark] 版本: {spark.version}, master: {sc.master}, defaultParallelism: {sc.defaultParallelism}")
                # 准备参数，避免在闭包中捕获 self（减少序列化问题）
                params_for_workers = dict(
                    population_dim=self.population_dim,
                    population_value_dim=self.population_value_dim,
                    W_total=self.W_total,
                    Q_min=self.Q_min,
                    Q_max=self.Q_max,
                    velocity_max=self.velocity_max,
                    velocity_min=self.velocity_min,
                    Max_p=self.Max_p,
                    Max_p_out=self.Max_p_out,
                )
                rdd = sc.parallelize([params_for_workers] * self.Max_p_out, numSlices=spark_num_slices)
                print(f"[Spark] RDD 分区数: {rdd.getNumPartitions()}")

                def _runner(params):
                    model = SmpsoModel(
                        params["population_dim"],
                        params["population_value_dim"],
                        params["W_total"],
                        params["Q_min"],
                        params["Q_max"],
                        params["velocity_max"],
                        params["velocity_min"],
                        params["Max_p"],
                        params["Max_p_out"],
                    )
                    archive, archive_population = model._single_outer_iteration(
                        k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path,
                        calculation_river_special_lose_water, or_q_list, objective_function_dim)
                    return [(archive, archive_population)]

                results = rdd.flatMap(_runner).collect()
                print(f"[Spark] 任务完成，收集到 {len(results)} 组结果")
                for archive, archive_population in results:
                    self.archive = archive
                    self.archive_population = archive_population
                    self.update_archive_efficient()
                try:
                    spark.stop()
                    print("[Spark] SparkSession 已关闭")
                except Exception:
                    pass
                finally:
                    if restore_spark_home is not None:
                        os.environ["SPARK_HOME"] = restore_spark_home
            except Exception as e:
                print(f"Spark 加速失败，改为本地执行：{e}")
                traceback.print_exc()
                for _ in range(self.Max_p_out):
                    self._single_outer_iteration(k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path,
                                                 calculation_river_special_lose_water, or_q_list, objective_function_dim)
                    self.update_archive_efficient()
        else:
            if use_spark and SparkSession is None:
                print("[Spark] 未找到 pyspark，已回退本地执行。若需使用 Spark，请检查 pyspark 安装与环境变量。")
                if _spark_import_error is not None:
                    print(f"[Spark] 导入错误: {_spark_import_error}")
            for _ in range(self.Max_p_out):
                self._single_outer_iteration(k, x, t, cost_cofficient_list, surface_cofficient_list, excel_path,
                                             calculation_river_special_lose_water, or_q_list, objective_function_dim)
                self.update_archive_efficient()

        self.write_results_to_file()
        return self.archive_total, self.archive_population_total



