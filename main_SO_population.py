import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from Decode import Decode
from Encode import Encode
from SO import SO  # 导入此是改进的ISO
import os
import re
import importlib.util
from typing import List, Dict, Any

# ISO改变:去掉非线性惯性权重，ExplorationPhaseFoodExists才是替换为WOA，ExplorationPhaseNoFood换回标准的

def generate_color_map(Job_serial_number: Dict[str, List[str]]) -> (Dict[str, Any], Dict[str, List[str]]):
    """生成颜色映射表，接收Job_serial_number作为参数"""
    # 定义色系与对应的工件ID
    job_groups = {
        'Reds': Job_serial_number["A"],  # A
        'Blues': Job_serial_number["B"],  # B
        'Greens': Job_serial_number["C"],  # C
        'Oranges': Job_serial_number["D"] + Job_serial_number.get("D_component1", []) + Job_serial_number.get(
            "D_component2", []),  # D
        'Purples': Job_serial_number["E"] + Job_serial_number.get("E_component1", []) + Job_serial_number.get(
            "E_component2", []),  # E
        'YlOrBr': Job_serial_number["F"],  # F
        'PuBu': Job_serial_number["G"],  # G
        'Greys': Job_serial_number["M1"],  # M1
        'BuGn': Job_serial_number["M2"],  # M2
        'RdPu': Job_serial_number["M3"]  # M3
    }

    # 创建颜色映射表
    color_map = {}

    # 遍历每个色系及其对应的工件ID
    for colormap_name, jobs in job_groups.items():
        # 获取对应组的颜色映射
        cmap = plt.get_cmap(colormap_name)

        # 分配颜色给每个工件，保证同组内有差异但属于同一个色系
        for i, job_id in enumerate(jobs):
            # 使用颜色映射的前80%颜色，避免最浅和最深的颜色
            color = cmap(i / (len(jobs) or 1) * 0.4 + 0.3)
            color_map[job_id] = color

    return color_map, job_groups

# def Gantt(Machines, tn, Job_serial_number, Special_Machine_ID) -> List[List[Any]]:     # 只输出结果实际并不画图
#     """绘制甘特图，增加Job_serial_number和Special_Machine_ID参数"""
#     color_map, job_groups = generate_color_map(Job_serial_number)
#     ans = []
#     for machine_index, Machine in enumerate(Machines):
#         start_times = Machine.O_start
#         end_times = Machine.O_end
#         # 计算当前机器的实际绘图位置
#         machine_id = machine_index + 1
#         mi = [machine_id]
#         for task_index, (start, end) in enumerate(zip(start_times, end_times)):
#             job_serial_number_val = Machine.assigned_task[task_index][0]
#             # 根据工件组分配标签前缀
#             if job_serial_number_val in job_groups['Reds']:
#                 b = f"P{job_serial_number_val}A"
#             elif job_serial_number_val in job_groups['Blues']:
#                 b = f"P{job_serial_number_val}B"
#             elif job_serial_number_val in job_groups['Greens']:
#                 b = f"P{job_serial_number_val}C"
#             elif job_serial_number_val in job_groups['Oranges']:
#                 b = f"P{job_serial_number_val}D"
#             elif job_serial_number_val in job_groups['Purples']:
#                 b = f"P{job_serial_number_val}E"
#             elif job_serial_number_val in job_groups['YlOrBr']:
#                 b = f"P{job_serial_number_val}F"
#             elif job_serial_number_val in job_groups['PuBu']:
#                 b = f"P{job_serial_number_val}G"
#             elif job_serial_number_val in job_groups['Greys']:
#                 b = f"P{job_serial_number_val}MTZ1"
#             elif job_serial_number_val in job_groups['BuGn']:
#                 b = f"P{job_serial_number_val}MTZ2"
#             elif job_serial_number_val in job_groups['RdPu']:
#                 b = f"P{job_serial_number_val}MTZ3"
#             # 产线最后一台机器标记为F前缀
#             if machine_id in (Special_Machine_ID["L1_last_machine_ID"],
#                               Special_Machine_ID["L2_last_machine_ID"],
#                               Special_Machine_ID["L3_last_machine_ID"],
#                               Special_Machine_ID["L4_last_machine_ID"]):
#                 b = 'F' + b[1:]
#             mi.append(b)
#         ans.append(mi)
#     return ans

def Gantt(Machines, tn, Job_serial_number, Special_Machine_ID) -> List[List[Any]]:
    """绘制甘特图，增加Job_serial_number和Special_Machine_ID参数"""
    color_map, job_groups = generate_color_map(Job_serial_number)

    # 设置画布大小
    plt.figure(figsize=(20, 10), dpi=300)

    group_spacing = 2  # 组之间的间距
    machine_offset = {
        1: 0,
        Special_Machine_ID["L1_last_machine_ID"] + 2: group_spacing,
        Special_Machine_ID["L2_pre_assembly_machine_ID_low"] + 2: 0.3,
        Special_Machine_ID["L2_last_machine_ID"] + 1: 0.3,
        Special_Machine_ID["L2_last_machine_ID"] + 2: group_spacing,
        Special_Machine_ID["L4_last_machine_ID"] + 1: group_spacing
    }

    ans = []

    for machine_index, Machine in enumerate(Machines):
        start_times = Machine.O_start
        end_times = Machine.O_end

        # 计算当前机器的实际绘图位置
        machine_id = machine_index + 1
        adjusted_index = machine_index + sum([offset for key, offset in machine_offset.items() if machine_id >= key])

        mi = [machine_id]

        for task_index, (start, end) in enumerate(zip(start_times, end_times)):
            job_serial_number_val = Machine.assigned_task[task_index][0]
            color = color_map.get(job_serial_number_val, 'gray')

            # 根据工件组分配标签前缀
            if job_serial_number_val in job_groups['Reds']:
                b = f"P{job_serial_number_val}A"
            elif job_serial_number_val in job_groups['Blues']:
                b = f"P{job_serial_number_val}B"
            elif job_serial_number_val in job_groups['Greens']:
                b = f"P{job_serial_number_val}C"
            elif job_serial_number_val in job_groups['Oranges']:
                b = f"P{job_serial_number_val}D"
            elif job_serial_number_val in job_groups['Purples']:
                b = f"P{job_serial_number_val}E"
            elif job_serial_number_val in job_groups['YlOrBr']:
                b = f"P{job_serial_number_val}F"
            elif job_serial_number_val in job_groups['PuBu']:
                b = f"P{job_serial_number_val}G"
            elif job_serial_number_val in job_groups['Greys']:
                b = f"P{job_serial_number_val}MTZ1"
            elif job_serial_number_val in job_groups['BuGn']:
                b = f"P{job_serial_number_val}MTZ2"
            elif job_serial_number_val in job_groups['RdPu']:
                b = f"P{job_serial_number_val}MTZ3"

            # 产线最后一台机器标记为F前缀
            if machine_index in (Special_Machine_ID["L1_last_machine_ID"],
                              Special_Machine_ID["L2_last_machine_ID"],
                              Special_Machine_ID["L3_last_machine_ID"],
                              Special_Machine_ID["L4_last_machine_ID"]):
                b = 'F' + b[1:]

            # 绘制甘特条
            plt.barh(adjusted_index, width=end - start, height=0.8, left=start,
                     color=color, edgecolor='black')
            plt.text(x=start + (end - start) / 2, y=adjusted_index,
                     s=b, va='center', ha='center')

            mi.append(b)

        ans.append(mi)

    # 设置Y轴刻度标签
    yticks = []
    yticklabels = []
    for i, machine in enumerate(Machines, start=1):
        adjusted_index = i - 1 + sum([offset for key, offset in machine_offset.items() if i >= key])
        yticks.append(adjusted_index)
        yticklabels.append(f'M{i}')

    plt.yticks(yticks, yticklabels)

    # 添加组标签
    group_labels = {
        'Line1': (0 + Special_Machine_ID["L1_last_machine_ID"] + 1) / 2 - 0.5,
        'Line2': (Special_Machine_ID["L1_last_machine_ID"] + 2 + Special_Machine_ID[
            "L2_last_machine_ID"] + 1) / 2 - 1 + group_spacing + 0.4,
        'Line3': (Special_Machine_ID["L2_last_machine_ID"] + 2 + Special_Machine_ID[
            "L3_last_machine_ID"] + 1) / 2 + 1 + group_spacing + 0.8,
        'Line4': (Special_Machine_ID["L4_last_machine_ID"] + 1 + Special_Machine_ID[
            "L4_last_machine_ID"] + 1) / 2 + 1 + group_spacing + 2.8
    }

    # 绘制组标签
    for label, position in group_labels.items():
        plt.text(-3, position, label, fontsize=14, rotation=90, va='center', ha='right')

    # 在横轴上标出tn数组中的各个时刻，并画垂直虚线
    for t in tn:
        t_rounded = round(t, 2)
        plt.axvline(x=t_rounded, color='gray', linestyle='--', linewidth=0.8)
        plt.text(x=t_rounded + 0.2, y=-1.2, s=f'{t_rounded:.2f}', ha='center', va='top', fontsize=12)

    plt.ylabel('Line', labelpad=20, fontsize=14)
    plt.xlabel('makespan (minute)', fontsize=14)
    plt.tight_layout()
    # plt.savefig('random_result_fig/SO_random_result_fig/gantt.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    return ans


def run_single_experiment(run_num: int, Job_serial_number: Dict[str, List[str]],
                          Processing_time: List[List[int]], J: List[List[int]],
                          J_num: int, M_num: int, O_num: int, kn: int,
                          Special_Machine_ID: Dict[str, int]) -> Dict[str, Any]:
    """运行单次实验，增加数据集变量作为参数"""
    start_time = time.time()

    Optimal_fit = 9999
    Optimal_CHS = None

    e = Encode(Processing_time, J, J_num, M_num)
    e.Get_Map_base_value()
    s = SO(O_num, Processing_time, J, M_num, kn, Job_serial_number, Special_Machine_ID)
    X = s.SO_initial()

    # X_mapped = e.Coding_mapping_conversion(X)
    # 记录种群历史的列表
    population_history = []
    # 记录初始种群
    population_history.append(X)

    Best_fit = []
    Fit = s.fitness(e, X, O_num)

    # 计算全局最佳适应度
    g_best = np.argmin(Fit)
    gy_best = Fit[g_best]
    Optimal_fit = gy_best
    Best_fit.append(round(gy_best, 3))

    food = X[g_best, :]
    food_mapped_individual = e.Individual_Coding_mapping_conversion(food)
    d = Decode(J, Processing_time, M_num, kn, Job_serial_number, Special_Machine_ID)
    y, Matching_result_all, tn = d.decode(food_mapped_individual, O_num)
    print("种群初始时food的适应度：", y)
    print("总配套关系为：", Matching_result_all)
    print("配套时刻：", tn)

    # 种群分离
    male_number = int(np.round(s.Pop_size / 2))
    female_number = s.Pop_size - male_number
    male = X[0:male_number, :]
    female = X[male_number:, :]
    male_individual_fitness = Fit[0:male_number]
    female_individual_fitness = Fit[male_number:]

    # 计算雌雄种群最佳个体
    male_fitness_best_index = np.argmin(male_individual_fitness)
    male_fitness_best_value = male_individual_fitness[male_fitness_best_index]
    male_best_fitness_individual = male[male_fitness_best_index, :]

    female_fitness_best_index = np.argmin(female_individual_fitness)
    female_fitness_best_value = female_individual_fitness[female_fitness_best_index]
    female_best_fitness_individual = female[female_fitness_best_index, :]
    count=0
    # 迭代过程
    for t in range(1, s.Max_Itertions + 1):   # t从1开始取值
        print('-' * 30)
        print(f"iter_{t}")
        temp = math.exp(-(t / s.Max_Itertions))
        # quantity = math.sin(random.random() + math.pi * t / 4 / s.Max_Itertions) * s.C1 * math.exp(
        #     (t - s.Max_Itertions) / s.Max_Itertions)
        quantity = s.C1 * math.exp((t - s.Max_Itertions) / s.Max_Itertions)


        if quantity > 1:
            quantity = 1

        new_male = np.matrix(np.zeros((male_number, e.Len_Chromo * 2)))
        new_female = np.matrix(np.zeros((female_number, e.Len_Chromo * 2)))

        if quantity < s.food_threshold:
            new_male, new_female = s.ExplorationPhaseNoFood(male_number, male, male_individual_fitness, new_male, female_number, female, female_individual_fitness, new_female)
        else:
            if temp > s.temp_threshold:
                new_male, new_female = s.ExplorationPhaseFoodExists(food, temp, male_number, male, new_male,
                                                                    female_number, female, new_female)
                count = count + 1
            else:
                model = random.random()
                if model < s.model_threshold:
                    new_male, new_female = s.fight(quantity, male, male_number, male_individual_fitness,
                                                   male_fitness_best_value,
                                                   male_best_fitness_individual, new_male, female, female_number,
                                                   female_individual_fitness,
                                                   female_fitness_best_value, female_best_fitness_individual,
                                                   new_female)
                else:
                    new_male, new_female = s.mating(quantity, male, male_number, male_individual_fitness, new_male,
                                                    female,
                                                    female_number, female_individual_fitness, new_female)

        # 更新种群和适应度
        male_best_fitness_individual, female_best_fitness_individual, food, gy_best, male, male_individual_fitness, male_fitness_best_value, female, female_individual_fitness, female_fitness_best_value = s.update(
            t, gy_best, O_num, e, food, male, male_number, male_individual_fitness, male_fitness_best_value, new_male,
            male_best_fitness_individual, female, female_number, female_individual_fitness, female_fitness_best_value,
            new_female, female_best_fitness_individual)

        if gy_best < Optimal_fit:
            Optimal_fit = gy_best
            food_mapped_individual1 = e.Individual_Coding_mapping_conversion(food)
            d = Decode(J, Processing_time, M_num, kn, Job_serial_number, Special_Machine_ID)
            y, Matching_result_all, tn = d.decode(food_mapped_individual1, O_num)
            ans = Gantt(d.Machines, tn, Job_serial_number, Special_Machine_ID)
            print("每台机器上工件的加工顺序：", ans)
            print("总配套关系为：", Matching_result_all)
            print("配套时刻为：", tn)

        print("当前代最优适应度：", round(gy_best, 3))
        Best_fit.append(round(gy_best, 3))

        # 合并当前种群并记录到历史中
        combined_population = np.vstack((male, female))
        # combined_population_mapped = e.Coding_mapping_conversion(combined_population)
        # population_history.append(combined_population_mapped.copy())
        population_history.append(combined_population)


    # 最后一次迭代结果
    food_mapped_individual1 = e.Individual_Coding_mapping_conversion(food)
    d = Decode(J, Processing_time, M_num, kn, Job_serial_number, Special_Machine_ID)
    y, Matching_result_all, tn = d.decode(food_mapped_individual1, O_num)
    ans = Gantt(d.Machines, tn, Job_serial_number, Special_Machine_ID)
    print("每台机器上工件的加工顺序：", ans)
    print("总配套关系为：", Matching_result_all)
    print("配套时刻为：", tn)

    # 绘制适应度收敛图
    # x = list(range(s.Max_Itertions + 1))
    # plt.plot(x, Best_fit, '-k')
    # plt.ylabel('Fitness')
    # plt.xlabel('Iteraions')
    # plt.savefig('适应度收敛图.png')
    # plt.show()
    print("每代最好适应度Best_fit：", Best_fit)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"程序运行时间为: {runtime:.3f}秒")
    print("count:", count)

    return {
        "run_num": run_num,
        "Matching_result_all": Matching_result_all,
        "tn": tn,
        "Best_fit": Best_fit,
        "runtime": runtime,
        "Optimal_fit": Optimal_fit,
        "Process_sequencing": ans,
        "population_history": population_history
    }


def run_experiment_for_dataset(dataset_path: str, dataset_name: str, result_subdir: str) -> None:
    """为指定数据集运行实验，result_subdir指定结果子目录"""
    result_dir = os.path.join('SO_result_redo', result_subdir)
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{dataset_name}_SO_result.txt")
    results = []

    # 加载数据集模块并获取变量
    spec = importlib.util.spec_from_file_location("dataset_module", dataset_path)
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)

    Job_serial_number = dataset_module.Job_serial_number
    Processing_time = dataset_module.Processing_time
    J = dataset_module.J
    J_num = dataset_module.J_num
    M_num = dataset_module.M_num
    O_num = dataset_module.O_num
    kn = dataset_module.kn
    Special_Machine_ID = dataset_module.Special_Machine_ID

    # 写入结果文件头
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"实验记录 - {dataset_name} - {run_number_each_experiment}次运行结果\n")
        f.write("=" * 50 + "\n\n")

    # 运行5次实验
    for i in range(1, run_number_each_experiment + 1):
        print(f"\n\n{'=' * 50}")
        print(f"开始 {dataset_name} 的第 {i} 次运行")
        print(f"{'=' * 50}\n")

        result = run_single_experiment(
            i, Job_serial_number, Processing_time, J, J_num,
            M_num, O_num, kn, Special_Machine_ID
        )
        results.append(result)

        # 保存种群历史数据到单独的文件
        os.makedirs("population_history", exist_ok=True)
        population_history_file = os.path.join("population_history", f"SOA_{dataset_name}_run_{i}_population_history.npz")
        # 保存为npz格式文件
        np.savez_compressed(
            population_history_file,
            populations=np.array(result['population_history'], dtype=object),
        )

        # 写入单次实验结果
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(f"第 {i} 次运行结果:\n")
            f.write(f"最优适应度: \n{result['Optimal_fit']}\n")
            f.write(f"每代最好适应度Best_fit: \n{result['Best_fit']}\n")
            f.write(f"每台机器上工件的加工顺序: \n{result['Process_sequencing']}\n")
            f.write(f"总配套关系为: \n{result['Matching_result_all']}\n")
            f.write(f"配套时刻为: \n{result['tn']}\n")
            f.write(f"运行时间: \n{result['runtime']:.3f}秒\n")
            f.write("-" * 50 + "\n\n")

    # 写入汇总统计
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write("\n\n汇总统计信息:\n")
        f.write("=" * 50 + "\n")

        run_times = [r['runtime'] for r in results]
        optimal_fits = [r['Optimal_fit'] for r in results]

        f.write(f"平均运行时间: {sum(run_times) / len(run_times):.3f}秒\n")
        f.write(f"最短运行时间: {min(run_times):.3f}秒\n")
        f.write(f"最长运行时间: {max(run_times):.3f}秒\n")
        f.write(f"平均最优适应度: {sum(optimal_fits) / len(optimal_fits):.3f}\n")
        f.write(f"最佳适应度: {min(optimal_fits):.3f}\n")
        f.write(f"最差适应度: {max(optimal_fits):.3f}\n")

    print(f"\n{dataset_name} 的所有实验已完成，结果已保存到 {result_file} 文件中")


def get_dataset_files(dataset_type: str = None) -> List[Dict[str, str]]:
    """
    获取指定类型的数据集文件
    Args:
        dataset_type: 可选值：None（所有）, 'factoryDataset', 'randomDataset', 'randomDataset/small', 'randomDataset/middle', 'randomDataset/big', 'randomDataset/small/SET1', 等
    """
    dataset_files = []
    base_dir = 'dataset_redo'

    # 处理特殊情况：factoryDataset
    if dataset_type == 'factoryDataset':
        source_dir = os.path.join(base_dir, 'factoryDataset')
        result_subdir = 'factoryDatasetResult'
        dataset_files.extend(_scan_dataset_directory(source_dir, result_subdir))
        return dataset_files

    # 处理随机数据集相关情况
    if dataset_type and dataset_type.startswith('randomDataset'):
        parts = dataset_type.split('/')
        if len(parts) > 3:
            print(f"错误: 数据集路径格式不正确，最多允许3级目录: {dataset_type}")
            return dataset_files

        # 构建源目录和结果子目录
        source_dir = base_dir
        result_subdir = 'randomDatasetResult'

        # 修复：确保randomDataset作为第一级目录
        if len(parts) >= 1 and parts[0] == 'randomDataset':
            source_dir = os.path.join(source_dir, 'randomDataset')
            # 跳过parts[0]，从下一级开始构建路径
            for part in parts[1:]:
                source_dir = os.path.join(source_dir, part)
                result_subdir = os.path.join(result_subdir, part)
        else:
            print(f"错误: 随机数据集路径必须以'randomDataset'开头: {dataset_type}")
            return dataset_files

        # 如果指定了SET级别，则直接扫描该目录
        if len(parts) >= 3 and parts[2].startswith('SET'):
            dataset_files.extend(_scan_dataset_directory(source_dir, result_subdir))
        # 如果指定了规模级别(small/middle/big)，则扫描其下所有SET目录
        elif len(parts) == 2:
            if not os.path.exists(source_dir):
                print(f"警告: 数据集目录 '{source_dir}' 不存在")
                return dataset_files

            # 扫描所有SET目录
            for set_dir in os.listdir(source_dir):
                set_path = os.path.join(source_dir, set_dir)
                if os.path.isdir(set_path) and set_dir.startswith('SET'):
                    set_result_subdir = os.path.join(result_subdir, set_dir)
                    dataset_files.extend(_scan_dataset_directory(set_path, set_result_subdir))
        # 如果只指定了randomDataset，则扫描所有规模和SET
        elif len(parts) == 1:
            sizes = ['small', 'middle', 'big']
            for size in sizes:
                size_dir = os.path.join(source_dir, size)
                size_result_subdir = os.path.join(result_subdir, size)

                if not os.path.exists(size_dir):
                    continue

                # 扫描该规模下的所有SET目录
                for set_dir in os.listdir(size_dir):
                    set_path = os.path.join(size_dir, set_dir)
                    if os.path.isdir(set_path) and set_dir.startswith('SET'):
                        set_result_subdir = os.path.join(size_result_subdir, set_dir)
                        dataset_files.extend(_scan_dataset_directory(set_path, set_result_subdir))
        else:
            print(f"警告: 无法解析数据集路径 '{dataset_type}'")

    # 如果未指定类型或类型不匹配，则扫描所有可能的数据集
    elif dataset_type is None:
        # 扫描factoryDataset
        factory_dir = os.path.join(base_dir, 'factoryDataset')
        if os.path.exists(factory_dir):
            dataset_files.extend(_scan_dataset_directory(factory_dir, 'factoryDatasetResult'))

        # 扫描randomDataset下的所有规模和SET
        random_dir = os.path.join(base_dir, 'randomDataset')
        if os.path.exists(random_dir):
            sizes = ['small', 'middle', 'big']
            for size in sizes:
                size_dir = os.path.join(random_dir, size)
                if not os.path.exists(size_dir):
                    continue

                # 扫描该规模下的所有SET目录
                for set_dir in os.listdir(size_dir):
                    set_path = os.path.join(size_dir, set_dir)
                    if os.path.isdir(set_path) and set_dir.startswith('SET'):
                        result_subdir = os.path.join('randomDatasetResult', size, set_dir)
                        dataset_files.extend(_scan_dataset_directory(set_path, result_subdir))
    else:
        print(f"错误: 未知的数据集类型 '{dataset_type}'")

    return dataset_files


def _scan_dataset_directory(source_dir: str, result_subdir: str) -> List[Dict[str, str]]:
    """扫描指定目录下的数据集文件"""
    files = []
    pattern = r'^Instance_L\(\d+,\d+,\d+\)_P\(\d+,\d+,\d+\)\.py$'

    if not os.path.exists(source_dir):
        print(f"警告: 数据集目录 '{source_dir}' 不存在")
        return files

    for filename in os.listdir(source_dir):
        if re.match(pattern, filename):
            full_path = os.path.join(source_dir, filename)
            if os.path.isfile(full_path):
                dataset_name = os.path.splitext(filename)[0]
                files.append({
                    'name': dataset_name,
                    'path': full_path,
                    'result_subdir': result_subdir
                })

    return files


def main(dataset_type: str = None) -> None:
    """
    主函数，支持指定数据集类型运行
    Args:
        dataset_type: 可选值：None（所有）, 'factoryDataset', 'randomDataset', 'randomDataset/small', 'randomDataset/middle', 'randomDataset/big', 'randomDataset/small/SET1', 等
    """
    print(f"开始处理数据集类型: {dataset_type or '所有'}")
    dataset_files = get_dataset_files(dataset_type)

    if not dataset_files:
        print("未找到符合条件的数据集文件")
    else:
        print(f"找到 {len(dataset_files)} 个数据集文件")
        for dataset in dataset_files:
            run_experiment_for_dataset(
                dataset['path'],
                dataset['name'],
                dataset['result_subdir']
            )

    print("\n所有实验已完成")


if __name__ == '__main__':
    # 设置要运行的数据集类型，None表示运行所有数据集
    # 可选值:
    #   None - 所有数据集
    #   'factoryDataset' - 仅工厂数据集
    #   'randomDataset' - 所有随机数据集
    #   'randomDataset/small' - 随机数据集中的small规模
    #   'randomDataset/middle/SET3' - 随机数据集中的middle规模下的SET3
    #   以此类推...

    # DATASET_TYPE = 'randomDataset/middle/SET1'
    DATASET_TYPE = 'factoryDataset'
    # DATASET_TYPE = 'randomDataset/small/SET4'

    run_number_each_experiment = 1    # 每个实验case跑的次数
    main(DATASET_TYPE)