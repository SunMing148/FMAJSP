import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from Decode import Decode
from Encode import Encode
from GA import GA
from Instance_2 import *
from matplotlib.colors import hsv_to_rgb
import matplotlib.colors as mcolors

# 绘制甘特图   v1
# def Gantt(Machines):
#     M = ['red', 'blue', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta',
#          'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
#          'navajowhite', 'navy', 'sandybrown', 'moccasin']
#     for i in range(len(Machines)):
#         Machine = Machines[i]
#         Start_time = Machine.O_start
#         End_time = Machine.O_end
#         for i_1 in range(len(End_time)):
#             plt.barh(i, width=End_time[i_1] - Start_time[i_1], height=0.8, left=Start_time[i_1],
#                      color=M[Machine.assigned_task[i_1][0] - 1], edgecolor='black')
#             plt.text(x=Start_time[i_1] + (End_time[i_1] - Start_time[i_1]) / 2 - 0.5, y=i,
#                      s=Machine.assigned_task[i_1][0])
#     plt.yticks(np.arange(len(Machines) + 1), np.arange(1, len(Machines) + 2))
#     plt.title('Scheduling Gantt chart')
#     plt.ylabel('Machines')
#     plt.xlabel('Time(min)')
#     plt.savefig('优化后排程方案的甘特图.png')
#     plt.show()




# #v2
# def Gantt(Machines):
#     # 颜色映射表
#     color_map = {
#         1: 'red', 2: 'blue', 3: 'yellow', 4: 'orange', 5: 'green',
#         6: 'palegoldenrod', 7: 'purple', 8: 'pink', 9: 'thistle', 10: 'magenta',
#         11: 'slateblue', 12: 'royalblue', 13: 'cyan', 14: 'aqua', 15: 'floralwhite',
#         16: 'ghostwhite', 17: 'goldenrod', 18: 'mediumslateblue', 19: 'navajowhite',
#         20: 'navy', 21: 'sandybrown', 22: 'moccasin', 23: 'lightcoral', 24: 'darkred',
#         25: 'salmon', 26: 'darksalmon', 27: 'sienna', 28: 'tan', 29: 'plum',
#         30: 'violet', 31: 'orchid', 32: 'fuchsia', 33: 'deeppink', 34: 'hotpink',
#         35: 'crimson', 36: 'lightpink', 37: 'lavenderblush', 38: 'papayawhip',
#         39: 'moccasin', 40: 'peachpuff', 41: 'palevioletred', 42: 'lightsalmon',
#         43: 'orangered', 44: 'darkorange', 45: 'gold', 46: 'khaki', 47: 'lemonchiffon',
#         48: 'lightgoldenrodyellow', 49: 'palegoldenrod', 50: 'papayawhip'
#     }
#     # 设置画布大小
#     plt.figure(figsize=(10, 6), dpi=300)
#     for machine_index, Machine in enumerate(Machines):
#         start_times = Machine.O_start
#         end_times = Machine.O_end
#
#         for task_index, (start, end) in enumerate(zip(start_times, end_times)):   # task_index为当前机器上的第几个任务
#             job_serial_number = Machine.assigned_task[task_index][0]   # 当前工件的序号
#             job_operation_num = Machine.assigned_task[task_index][1]   # 当前工件的工序序号
#             color = color_map.get(job_serial_number, 'gray')  # 默认颜色为灰色
#             # 绘制甘特条
#             plt.barh(machine_index, width=end - start, height=0.8, left=start,
#                      color=color, edgecolor='black')
#             # 在甘特条中间添加任务ID
#             # plt.text(x=start + (end - start) / 2 , y=machine_index,
#             #          s=str(job_serial_number) + '-' +str(job_operation_num), va='center', ha='center')
#             plt.text(x=start + (end - start) / 2 , y=machine_index,
#                      s=str(job_serial_number), va='center', ha='center')
#
#     # 设置Y轴刻度标签
#     plt.yticks(np.arange(len(Machines)), ['{}'.format(i + 1) for i in range(len(Machines))])
#     # 添加标题和坐标轴标签
#     plt.title('Scheduling Gantt chart')
#     plt.ylabel('Machines')
#     plt.xlabel('Time(min)')
#     # 保存并显示图像
#     plt.tight_layout()  # 调整布局防止裁剪文字
#     plt.savefig('优化后排程方案的甘特图.png', bbox_inches='tight')
#     plt.show()




#  同一组工件是一个色系
# def generate_color_map(job_count):
#     # 定义三个大色系的基准颜色
#     base_colormaps = ['Reds', 'Blues', 'Greens']
#
#     # 创建颜色映射表
#     color_map = {}
#
#     # 每个组的工件数量
#     jobs_per_group = 9
#     # 确定组的数量
#     group_count = job_count // jobs_per_group
#
#     for group in range(group_count):
#         # 获取对应组的颜色映射
#         cmap = plt.get_cmap(base_colormaps[group % len(base_colormaps)])
#
#         # 分配颜色给每个工件，保证同组内有差异但属于同一个色系
#         for job in range(jobs_per_group):
#             job_serial_number = group * jobs_per_group + job + 1
#             # 使用颜色映射的前80%颜色，避免最浅和最深的颜色
#             color = cmap(job / (jobs_per_group - 1) * 0.8 + 0.2)
#             color_map[job_serial_number] = color
#
#     return color_map


def generate_color_map():
    # 定义色系与对应的工件ID
    job_groups = {
        'Reds': [1, 2, 3],
        'Blues': [10, 11, 12],
        'Greens': [19, 20, 21],
        'Oranges': [4, 5, 13, 14, 6, 15],
        'Purples': [22, 23, 24],
        'YlOrBr': [7, 8, 9, 16, 17, 18],
        'PuBu': [25, 26, 27]
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
            # 这里使用了线性插值来确保颜色分布在色带的中间部分
            color = cmap(i / (len(jobs) - 1) * 0.4 + 0.3)
            color_map[job_id] = color

    return color_map

#v3
def Gantt(Machines):
    # color_map = {
    #     # 红色系 - 第一组机器
    #     1: 'darkred', 2: 'firebrick', 3: 'indianred',
    #     10: 'brown', 11: 'salmon', 12: 'lightsalmon',
    #     19: 'tomato', 20: 'coral', 21: 'orangered',
    #
    #     # 蓝色系 - 第二组机器
    #     4: 'navy', 5: 'midnightblue', 6: 'darkblue',
    #     13: 'mediumblue', 14: 'royalblue', 15: 'cornflowerblue',
    #     22: 'steelblue', 23: 'deepskyblue', 24: 'dodgerblue',
    #
    #     # 绿色系 - 第三组机器
    #     7: 'darkgreen', 8: 'forestgreen', 9: 'seagreen',
    #     16: 'mediumseagreen', 17: 'limegreen', 18: 'springgreen',
    #     25: 'mediumspringgreen', 26: 'aquamarine', 27: 'turquoise'
    # }

    # color_map = {
    #     # 红色系
    #     1: 'darkred', 2: 'firebrick', 3: 'indianred',
    #     # 蓝色系
    #     10: 'navy', 11: 'royalblue', 12: 'cornflowerblue',
    #     # 绿色系
    #     19: 'darkgreen', 20: 'forestgreen', 21: 'seagreen',
    #     # 紫色系 (用于ID 4, 5, 13, 14, 6, 15)
    #     4: 'purple', 5: 'darkviolet', 6: 'mediumorchid',
    #     13: 'plum', 14: 'rebeccapurple', 15: 'mediumpurple',
    #     # 橙色系 (用于ID 22, 23, 24)
    #     22: 'orange', 23: 'darkorange', 24: 'coral',
    #     # 青色系 (用于ID 7, 8, 9, 16, 17, 18)
    #     7: 'teal', 8: 'aqua', 9: 'cyan',
    #     16: 'lightseagreen', 17: 'mediumturquoise', 18: 'paleturquoise',
    #     # 黄色系 (用于ID 25, 26, 27)
    #     25: 'gold', 26: 'yellow', 27: 'khaki'
    # }

    color_map = generate_color_map()

    # 设置画布大小
    plt.figure(figsize=(10, 6), dpi=300)

    group_spacing = 2  # 组之间的间距
    # machine_offset = {1: 0, 15: group_spacing, 22: group_spacing}
    machine_offset = {1: 0, 15: group_spacing, 18: 0.3, 21: 0.3,22: group_spacing}


    for machine_index, Machine in enumerate(Machines):
        start_times = Machine.O_start
        end_times = Machine.O_end

        # 计算当前机器的实际绘图位置
        machine_id = machine_index + 1
        adjusted_index = machine_index + sum([offset for key, offset in machine_offset.items() if machine_id >= key])

        for task_index, (start, end) in enumerate(zip(start_times, end_times)):
            job_serial_number = Machine.assigned_task[task_index][0]
            # job_operation_num = Machine.assigned_task[task_index][1]
            color = color_map.get(job_serial_number, 'gray')

            if job_serial_number in (1, 2, 3):
                b = f"{job_serial_number}A"
            elif job_serial_number in (10, 11, 12):
                b = f"{job_serial_number}B"
            elif job_serial_number in (19, 20, 21):
                b = f"{job_serial_number}C"
            elif job_serial_number in (4, 5, 13, 14, 6, 15):
                b = f"{job_serial_number}D"
            elif job_serial_number in (22, 23, 24):
                b = f"{job_serial_number}E"
            elif job_serial_number in (7, 8, 9, 16, 17, 18):
                b = f"{job_serial_number}F"
            elif job_serial_number in (25, 26, 27):
                b = f"{job_serial_number}G"

            # 绘制甘特条
            plt.barh(adjusted_index, width=end - start, height=0.8, left=start,
                     color=color, edgecolor='black')
            # 在甘特条中间添加任务ID

            # plt.text(x=start + (end - start) / 2, y=adjusted_index,
            #          s=str(job_serial_number), va='center', ha='center')
            plt.text(x=start + (end - start) / 2, y=adjusted_index,
                     s=b, va='center', ha='center')

    # 设置Y轴刻度标签
    yticks = []
    yticklabels = []
    for i, machine in enumerate(Machines, start=1):
        adjusted_index = i - 1 + sum([offset for key, offset in machine_offset.items() if i >= key])
        yticks.append(adjusted_index)
        yticklabels.append('{}'.format(i))

    plt.yticks(yticks, yticklabels)

    # 添加组标签
    group_labels = {
        'Proxima Sensor Line': (0 + 14) / 2 - 0.5,  # 第一组中间位置
        'Mechanism Line': (15 + 21) / 2 - 1 + group_spacing +0.4,  # 第二组中间位置，考虑偏移
        'Arc Chute Line': (22 + 25) / 2 + 1  + group_spacing +0.8  # 第三组中间位置，考虑偏移
    }

    # 绘制组标签
    for label, position in group_labels.items():
        plt.text(-0.3, position, label, fontsize=8, rotation=90, va='center', ha='right')

    # 添加标题和坐标轴标签
    plt.title('Scheduling Gantt chart')
    plt.ylabel('Machines', labelpad=20)
    plt.xlabel('Time(min)')
    # 保存并显示图像
    plt.tight_layout()
    plt.savefig('优化后排程方案的甘特图.png', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    Optimal_fit = 9999  # 最佳适应度（初始化）
    Optimal_CHS = None  # 最佳适应度对应的基因个体（初始化）
    g = GA()
    e = Encode(Processing_time, g.Pop_size, J, J_num, M_num)
    CHS1 = e.Global_initial()
    CHS2 = e.Random_initial()
    CHS3 = e.Local_initial()
    C = np.vstack((CHS1, CHS2, CHS3))
    Best_fit = []  # 记录适应度在迭代过程中的变化，便于绘图
    for i in range(g.Max_Itertions):
        print("iter_{}".format(i))
        Fit = g.fitness(C, J, Processing_time, M_num, O_num)
        Best = C[Fit.index(min(Fit))]
        best_fitness = min(Fit)
        if best_fitness < Optimal_fit:
            Optimal_fit = best_fitness
            Optimal_CHS = Best
            Best_fit.append(Optimal_fit)
            print('best_fitness', best_fitness)
            d = Decode(J, Processing_time, M_num)
            # Fit.append(d.decode(Optimal_CHS, O_num))
            d.decode(Optimal_CHS, O_num)
            Gantt(d.Machines)
        else:
            Best_fit.append(Optimal_fit)

        Select_idx = g.Select(Fit)
        C = [C[Select_i] for Select_i in Select_idx]

        if i == (g.Max_Itertions - 1):
            print("最优染色体Optimal_CHS：", Optimal_CHS)

        for j in range(len(C)):
            offspring = []
            if random.random() < g.Pc:
                N_i = random.choice(np.arange(len(C)))
                if random.random() < g.Pv:
                    Cross = g.machine_cross(C[j], C[N_i], O_num)
                else:
                    Cross = g.operation_cross(C[j], C[N_i], O_num, J_num)
                offspring.append(Cross[0])
                offspring.append(Cross[1])
                offspring.append(C[j])
            if random.random() < g.Pm:
                if random.random() < g.Pw:
                    Variance = g.machine_variation(C[j], Processing_time, O_num, J)
                # else:
                #     Variance = g.operation_variation(C[j], O_num, J_num, J, Processing_time, M_num)
                    offspring.append(Variance)
            if offspring != []:
                Fit = g.fitness(offspring, J, Processing_time, M_num, O_num)
                C[j] = offspring[Fit.index(min(Fit))]
                # C[j] = random.choice(offspring)

    # x = np.linspace(0, 50, g.Max_Itertions)
    x = [_ for _ in range(g.Max_Itertions)]  # 横坐标 迭代数
    plt.plot(x, Best_fit, '-k')
    plt.title('the maximum completion time of each iteration')
    plt.ylabel('Cmax')
    plt.xlabel('Test Num')
    plt.savefig('最大完成时间的优化过程.png')
    plt.show()
    print("每代最好适应度Best_fit：", Best_fit)
