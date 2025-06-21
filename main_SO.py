import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from Decode import Decode
from Encode import Encode
from SO import SO            # 导入此是改进的ISO
# from SO_beifen import SO   # 导入此是未改进的标准SO
from Instance_2 import *


def generate_color_map(k):
    # 定义色系与对应的工件ID
    job_groups = {
        'Reds': [1, 2, 3],                        # A
        'Blues': [11, 12, 13],                    # B
        'Greens': [21, 22, 23],                   # C
        'Oranges': [4, 5, 6, 14, 15, 16],         # D
        'Purples': [24, 25, 26],                  # E
        'YlOrBr': [7, 8, 9, 17, 18, 19],          # F
        'PuBu': [27, 28, 29],                     # G
        'Greys': [10],                            # M1
        'BuGn': [20],                             # M2
        'RdPu': [30]                              # M3
    }

    # 遍历job_groups中的每个键值对
    for key, values in job_groups.items():
        original_values = values[:]  # 保存原始数组的副本
        for i in range(1, k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 30 * i for x in original_values]  # 计算新的元素值
            job_groups[key].extend(new_values)  # 将新元素添加到数组中

    # 创建颜色映射表
    color_map = {}

    # 遍历每个色系及其对应的工件ID
    for colormap_name, jobs in job_groups.items():
        # 获取对应组的颜色映射
        cmap = plt.get_cmap(colormap_name)

        # 分配颜色给每个工件，保证同组内有差异但属于同一个色系
        for i, job_id in enumerate(jobs):
            # 使用颜色映射的前80%颜色，避免最浅和最深的颜色
            # 这里使用了线性插值来确保颜色分布在色带的中间部分
            color = cmap(i / (len(jobs)) * 0.4 + 0.3)
            color_map[job_id] = color

    return color_map, job_groups

#v3
def Gantt(Machines,k,tn):
    color_map, job_groups = generate_color_map(k)

    # 设置画布大小
    plt.figure(figsize=(20, 10), dpi=300)
    # plt.figure(figsize=(10, 6), dpi=300)

    group_spacing = 2  # 组之间的间距
    # machine_offset = {1: 0, 15: group_spacing, 22: group_spacing}
    machine_offset = {1: 0, 15: group_spacing, 18: 0.3, 21: 0.3, 22: group_spacing, 26: group_spacing}

    ans = []

    for machine_index, Machine in enumerate(Machines):
        start_times = Machine.O_start
        end_times = Machine.O_end

        # 计算当前机器的实际绘图位置
        machine_id = machine_index + 1
        adjusted_index = machine_index + sum([offset for key, offset in machine_offset.items() if machine_id >= key])

        mi=[]
        mi.append(machine_id)

        for task_index, (start, end) in enumerate(zip(start_times, end_times)):
            job_serial_number = Machine.assigned_task[task_index][0]
            # job_operation_num = Machine.assigned_task[task_index][1]
            color = color_map.get(job_serial_number, 'gray')

            if job_serial_number in job_groups['Reds']:
                b = f"P{job_serial_number}A"
            elif job_serial_number in job_groups['Blues']:
                b = f"P{job_serial_number}B"
            elif job_serial_number in job_groups['Greens']:
                b = f"P{job_serial_number}C"
            elif job_serial_number in job_groups['Oranges']:
                b = f"P{job_serial_number}D"
            elif job_serial_number in job_groups['Purples']:
                b = f"P{job_serial_number}E"
            elif job_serial_number in job_groups['YlOrBr']:
                b = f"P{job_serial_number}F"
            elif job_serial_number in job_groups['PuBu']:
                b = f"P{job_serial_number}G"
            elif job_serial_number in job_groups['Greys']:
                b = f"P{job_serial_number}MTZ1"
            elif job_serial_number in job_groups['BuGn']:
                b = f"P{job_serial_number}MTZ2"
            elif job_serial_number in job_groups['RdPu']:
                b = f"P{job_serial_number}MTZ3"

            if machine_id in (14,21,25,26):
                b = 'F' + b[1:]

            # 绘制甘特条
            plt.barh(adjusted_index, width=end - start, height=0.8, left=start,
                     color=color, edgecolor='black')
            # 在甘特条中间添加任务ID

            # plt.text(x=start + (end - start) / 2, y=adjusted_index,
            #          s=str(job_serial_number), va='center', ha='center')
            plt.text(x=start + (end - start) / 2, y=adjusted_index,
                     s=b, va='center', ha='center')

            mi.append(b)

        ans.append(mi)

    print("每台机器上工件的加工顺序：",ans)

    # 设置Y轴刻度标签
    yticks = []
    yticklabels = []
    for i, machine in enumerate(Machines, start=1):
        adjusted_index = i - 1 + sum([offset for key, offset in machine_offset.items() if i >= key])
        yticks.append(adjusted_index)
        yticklabels.append('M{}'.format(i))

    plt.yticks(yticks, yticklabels)

    # 添加组标签
    group_labels = {
        'Line1': (0 + 14) / 2 - 0.5,  # 第一组中间位置
        'Line2': (15 + 21) / 2 - 1 + group_spacing + 0.4,  # 第二组中间位置，考虑偏移
        'Line3': (22 + 25) / 2 + 1  + group_spacing + 0.8,  # 第三组中间位置，考虑偏移
        'Line4': (26 + 26) / 2 + 1 + group_spacing + 2.8  # 第三组中间位置，考虑偏移
    }

    # 绘制组标签
    for label, position in group_labels.items():
        # plt.text(-0.3, position, label, fontsize=8, rotation=90, va='center', ha='right')  # 适用成品少
        plt.text(-1.5, position, label, fontsize=12, rotation=90, va='center', ha='right')    # 适用成品多

    # 在横轴上标出tn数组中的各个时刻，并画垂直虚线
    for t in tn:
        t_rounded = round(t, 2)  # 保留两位小数
        plt.axvline(x=t_rounded, color='gray', linestyle='--', linewidth=0.8)
        plt.text(x=t_rounded+0.2, y=-1.2, s=f'{t_rounded:.2f}', ha='center', va='top', fontsize=12)  # 标记在下方
    # 调整横轴范围，避免遮挡
    # plt.ylim(-1.5, plt.ylim()[1])  # 为下方标记留出空间

    # 添加标题和坐标轴标签
    # plt.title('Scheduling Gantt chart')
    plt.ylabel('Line', labelpad=20, fontsize=12)
    plt.xlabel('makespan (minute)', fontsize=12)
    # 保存并显示图像
    plt.tight_layout()
    plt.savefig('优化后排程方案的甘特图.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    start_time = time.time()  # 记录开始时间

    Optimal_fit = 9999  # 最佳适应度（初始化）
    Optimal_CHS = None  # 最佳适应度对应的基因个体（初始化）

    e = Encode(Processing_time, J, J_num, M_num)
    e.Get_Map_base_value()
    s = SO(e.Len_Chromo, k)
    X = s.SO_initial()
    
    Best_fit = []  # 记录适应度在迭代过程中的变化，便于绘图
    
    Fit = s.fitness(e, X, J, Processing_time, M_num, O_num)

    # 计算出全局最佳适应度, 因为这个是第一次进行操作，不用和别的进行对比
    g_best = np.argmin(Fit)
    gy_best = Fit[g_best]
    Optimal_fit = gy_best # Optimal_fit存放比上代更优的适应度
    Best_fit.append(round(gy_best,3))

    # 得到食物的位置，其实就是当前全局最佳适应度的位置 食物也是全局最优个体
    food = X[g_best, :]   # 种群初始化时的最优个体
    food_mapped_individual = e.Individual_Coding_mapping_conversion(food)
    d = Decode(J, Processing_time, M_num, k)
    y, Matching_result_all, tn = d.decode(food_mapped_individual, O_num)
    print("种群初始时food的适应度：",y)
    # Gantt(d.Machines,k)   # 种群初始化时的最优个体 解码后 对应的甘特图
    print("总配套关系为：", Matching_result_all)
    print("配套时刻：", tn)

    # 将种群进行分离,一半归为雌性，一半归为雄性
    male_number = int(np.round(s.Pop_size / 2))
    female_number = s.Pop_size - male_number
    male = X[0:male_number, :]
    female = X[male_number:, :]
    # 从总的适应度中分离出雄性的适应度
    male_individual_fitness = Fit[0:male_number]
    # 从总的适应度中分理处雌性的适应度
    female_individual_fitness = Fit[male_number:]

    # 计算雄性种群中的个体最佳
    male_fitness_best_index = np.argmin(male_individual_fitness)
    male_fitness_best_value = male_individual_fitness[male_fitness_best_index]
    # 雄性中最优个体
    male_best_fitness_individual = male[male_fitness_best_index, :]
    # 计算雌性种群中的个体最佳
    female_fitness_best_index = np.argmin(female_individual_fitness)
    female_fitness_best_value = female_individual_fitness[female_fitness_best_index]
    # 雌性中最优个体
    female_best_fitness_individual = female[female_fitness_best_index, :]

    # 迭代
    for t in range(1, s.Max_Itertions+1):
        print("iter_{}".format(t))
        # 计算温度
        temp = math.exp(-(t / s.Max_Itertions))
        # 计算食物的质量
        quantity = s.C1 * math.exp((t - s.Max_Itertions) / s.Max_Itertions)
        # 正弦变化的自适应惯性权重 食物指数更新策略
        quantity = math.sin(random.random() + math.pi * t / 4 / s.Max_Itertions) * s.C1 * math.exp((t - s.Max_Itertions) / s.Max_Itertions)

        # 更新位置之后的male
        new_male = np.matrix(np.zeros((male_number, e.Len_Chromo * 2)))
        # 更新位置之后的female
        new_female = np.matrix(np.zeros((female_number, e.Len_Chromo * 2)))

        if quantity > 1:
            quantity = 1
        # 先判断食物的质量是不是超过了阈值
        if quantity < s.food_threshold:
            # 如果当前是没有食物的就寻找食物
            # new_male, new_female = s.ExplorationPhaseNoFood(male_number, male, male_individual_fitness, new_male, female_number, female, female_individual_fitness, new_female)
            new_male, new_female = s.ExplorationPhaseNoFood(food, male_number, male, male_individual_fitness, new_male, female_number, female, female_individual_fitness, new_female) # WOA螺旋

        else:
            # 当前有食物开始进入探索阶段
            # 先判断当前的温度是冷还是热
            if temp > s.temp_threshold:  # 表示当前是热的
                # 热了就不进行交配，开始向食物的位置进行移动
                # 雄性先移动
                new_male, new_female = s.ExplorationPhaseFoodExists(food, temp, male_number, male, new_male, female_number, female, new_female)
            else:
                # 如果当前的温度是比较的冷的，就比较适合战斗和交配
                # 生成一个随机值来决定是要战斗还是要交配
                model = random.random()
                if model < s.model_threshold:
                    # 当前进入战斗模式
                    new_male, new_female = s.fight(quantity, male, male_number, male_individual_fitness, male_fitness_best_value,
                              male_best_fitness_individual, new_male, female, female_number, female_individual_fitness,
                              female_fitness_best_value, female_best_fitness_individual, new_female)
                else:
                    # 当前将进入交配模式
                    new_male, new_female = s.mating(quantity, male, male_number, male_individual_fitness, new_male, female,
                               female_number, female_individual_fitness, new_female)

        # 将更新后的位置进行处理
        male_best_fitness_individual, female_best_fitness_individual, food, gy_best, male, male_individual_fitness, male_fitness_best_value, female, female_individual_fitness, female_fitness_best_value = s.update(t, gy_best, O_num, e, food, male, male_number, male_individual_fitness, male_fitness_best_value, new_male, male_best_fitness_individual, female, female_number, female_individual_fitness, female_fitness_best_value, new_female, female_best_fitness_individual)


        if gy_best < Optimal_fit:
            Optimal_fit = gy_best
            #food 就是最优个体
            food_mapped_individual1 = e.Individual_Coding_mapping_conversion(food)
            d = Decode(J, Processing_time, M_num, k)
            y, Matching_result_all, tn = d.decode(food_mapped_individual1, O_num)
            Gantt(d.Machines,k,tn)  # 种群初始化时的最优个体 解码后 对应的甘特图
            print("总配套关系为：",Matching_result_all)
            print("配套时刻为：",tn)

        print("当前代最优适应度：", round(gy_best, 3))
        Best_fit.append(round(gy_best, 3))

    x = [_ for _ in range(s.Max_Itertions+1)]  # 横坐标 迭代数
    plt.plot(x, Best_fit, '-k')
    plt.title('the maximum completion time of each iteration')
    plt.ylabel('Cmax')
    plt.xlabel('Test Num')
    plt.savefig('加权配套时间总和过程.png')
    plt.show()
    print("每代最好适应度Best_fit：", Best_fit)

    end_time = time.time()  # 记录结束时间
    print("程序运行时间为: {:.3f}秒".format(end_time - start_time))
