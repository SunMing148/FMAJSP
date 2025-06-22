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

    # print("每台机器上工件的加工顺序：",ans)

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

    return ans

# v4
# def Gantt(Machines, k, tn):
#     color_map, job_groups = generate_color_map(k)
#
#     # 分离机器数据
#     main_machines = Machines[:25]  # 前25台机器 (Line1-Line3)
#     line4_machine = Machines[25:]  # 第26台机器 (Line4)
#
#     # ==============================================
#     # 绘制主甘特图 (Line1-Line3)
#     # ==============================================
#     plt.figure(figsize=(20, 10), dpi=300)
#     group_spacing = 2  # 组之间的间距
#     machine_offset = {1: 0, 15: group_spacing, 18: 0.3, 21: 0.3, 22: group_spacing, 26: group_spacing}
#     ans = []
#
#     for machine_index, Machine in enumerate(main_machines):
#         start_times = Machine.O_start
#         end_times = Machine.O_end
#
#         # 计算当前机器的实际绘图位置
#         machine_id = machine_index + 1
#         adjusted_index = machine_index + sum([offset for key, offset in machine_offset.items() if machine_id >= key])
#
#         mi = []
#         mi.append(machine_id)
#
#         for task_index, (start, end) in enumerate(zip(start_times, end_times)):
#             job_serial_number = Machine.assigned_task[task_index][0]
#             color = color_map.get(job_serial_number, 'gray')
#
#             if job_serial_number in job_groups['Reds']:
#                 b = f"P{job_serial_number}A"
#             elif job_serial_number in job_groups['Blues']:
#                 b = f"P{job_serial_number}B"
#             elif job_serial_number in job_groups['Greens']:
#                 b = f"P{job_serial_number}C"
#             elif job_serial_number in job_groups['Oranges']:
#                 b = f"P{job_serial_number}D"
#             elif job_serial_number in job_groups['Purples']:
#                 b = f"P{job_serial_number}E"
#             elif job_serial_number in job_groups['YlOrBr']:
#                 b = f"P{job_serial_number}F"
#             elif job_serial_number in job_groups['PuBu']:
#                 b = f"P{job_serial_number}G"
#             elif job_serial_number in job_groups['Greys']:
#                 b = f"P{job_serial_number}MTZ1"
#             elif job_serial_number in job_groups['BuGn']:
#                 b = f"P{job_serial_number}MTZ2"
#             elif job_serial_number in job_groups['RdPu']:
#                 b = f"P{job_serial_number}MTZ3"
#
#             if machine_id in (14, 21, 25, 26):
#                 b = 'F' + b[1:]
#
#             # 绘制甘特条
#             plt.barh(adjusted_index, width=end - start, height=0.8, left=start,
#                      color=color, edgecolor='black')
#             plt.text(x=start + (end - start) / 2, y=adjusted_index,
#                      s=b, va='center', ha='center')
#
#             mi.append(b)
#
#         ans.append(mi)
#
#     print("每台机器上工件的加工顺序：", ans)
#
#     # 设置Y轴刻度标签
#     yticks = []
#     yticklabels = []
#     for i, machine in enumerate(main_machines, start=1):
#         adjusted_index = i - 1 + sum([offset for key, offset in machine_offset.items() if i >= key])
#         yticks.append(adjusted_index)
#         yticklabels.append('M{}'.format(i))
#
#     plt.yticks(yticks, yticklabels)
#
#     # 添加组标签
#     group_labels = {
#         'Line1': (0 + 14) / 2 - 0.5,
#         'Line2': (15 + 21) / 2 - 1 + group_spacing + 0.4,
#         'Line3': (22 + 25) / 2 + 1 + group_spacing + 0.8
#     }
#
#     for label, position in group_labels.items():
#         plt.text(-1.5, position, label, fontsize=12, rotation=90, va='center', ha='right')
#
#     # 在横轴上标出tn数组中的各个时刻，并画垂直虚线
#     for t in tn:
#         t_rounded = round(t, 2)
#         plt.axvline(x=t_rounded, color='gray', linestyle='--', linewidth=0.8)
#         plt.text(x=t_rounded + 0.2, y=-1.2, s=f'{t_rounded:.2f}', ha='center', va='top', fontsize=12)
#
#     plt.ylabel('Line', labelpad=20, fontsize=12)
#     plt.xlabel('makespan (minute)', fontsize=12)
#     plt.tight_layout()
#     plt.savefig('优化后排程方案的甘特图_主产线.png', bbox_inches='tight')
#     plt.show()
#
#     # ==============================================
#     # 绘制Line4甘特图 (单独的第26台机器)
#     # ==============================================
#     if line4_machine:  # 确保有第26台机器数据
#         plt.figure(figsize=(20, 2), dpi=300)  # 较小的图高度，因为只有一台机器
#
#         Machine = line4_machine[0]
#         start_times = Machine.O_start
#         end_times = Machine.O_end
#
#         mi = []
#         mi.append(26)  # 机器编号
#
#         for task_index, (start, end) in enumerate(zip(start_times, end_times)):
#             job_serial_number = Machine.assigned_task[task_index][0]
#             color = color_map.get(job_serial_number, 'gray')
#
#             if job_serial_number in job_groups['Reds']:
#                 b = f"P{job_serial_number}A"
#             elif job_serial_number in job_groups['Blues']:
#                 b = f"P{job_serial_number}B"
#             elif job_serial_number in job_groups['Greens']:
#                 b = f"P{job_serial_number}C"
#             elif job_serial_number in job_groups['Oranges']:
#                 b = f"P{job_serial_number}D"
#             elif job_serial_number in job_groups['Purples']:
#                 b = f"P{job_serial_number}E"
#             elif job_serial_number in job_groups['YlOrBr']:
#                 b = f"P{job_serial_number}F"
#             elif job_serial_number in job_groups['PuBu']:
#                 b = f"P{job_serial_number}G"
#             elif job_serial_number in job_groups['Greys']:
#                 b = f"P{job_serial_number}MTZ1"
#             elif job_serial_number in job_groups['BuGn']:
#                 b = f"P{job_serial_number}MTZ2"
#             elif job_serial_number in job_groups['RdPu']:
#                 b = f"P{job_serial_number}MTZ3"
#
#             b = 'F' + b[1:]  # 第26台机器是F开头
#
#             # 绘制甘特条，x轴从tn[0]开始
#             plt.barh(0, width=end - start, height=0.8, left=start,
#                      color=color, edgecolor='black')
#             plt.text(x=start + (end - start) / 2, y=0,
#                      s=b, va='center', ha='center')
#
#             mi.append(b)
#
#         print("Line4机器上工件的加工顺序：", mi)
#
#         # 设置Y轴
#         plt.yticks([0], ['M26'])
#
#         # 设置X轴从tn[0]开始
#         plt.xlim(left=tn[0])
#
#         # 添加组标签
#         # plt.text(-1.5, 0, 'Line4', fontsize=12, rotation=90, va='center', ha='right')
#
#         # 在横轴上标出tn数组中的各个时刻，并画垂直虚线
#         for t in tn:
#             if t >= tn[0]:  # 只显示大于等于起始时间的标记
#                 t_rounded = round(t, 2)
#                 plt.axvline(x=t_rounded, color='gray', linestyle='--', linewidth=0.8)
#                 plt.text(x=t_rounded + 0.2, y=-0.8, s=f'{t_rounded:.2f}', ha='center', va='top', fontsize=12)
#
#         plt.ylabel('Line', labelpad=20, fontsize=12)
#         plt.xlabel('makespan (minute)', fontsize=12)
#         plt.tight_layout()
#         plt.savefig('优化后排程方案的甘特图_Line4.png', bbox_inches='tight')
#         plt.show()

# # v5
# def Gantt(Machines, k, tn):
#     color_map, job_groups = generate_color_map(k)
#
#     # 分离机器数据
#     main_machines = Machines[:25]  # 前25台机器 (Line1-Line3)
#     line4_machine = Machines[25:]  # 第26台机器 (Line4)
#
#     # 计算全局时间范围
#     all_start = min([min(machine.O_start) for machine in Machines])
#     all_end = max([max(machine.O_end) for machine in Machines])
#     time_range = all_end - all_start
#
#     # 设置统一的图形参数
#     common_params = {
#         'figsize': (20, 10),  # 统一图形大小
#         'dpi': 300,
#         'bar_height': 0.8,
#         'fontsize': 12,
#         'time_marker_offset': -1.2,
#         'group_label_offset': -1.5,
#         'time_range_padding': 0.1 * time_range  # 时间轴两侧留10%空白
#     }
#
#     # ==============================================
#     # 绘制主甘特图 (Line1-Line3)
#     # ==============================================
#     plt.figure(figsize=common_params['figsize'], dpi=common_params['dpi'])
#
#     machine_offset = {1: 0, 15: 2, 18: 0.3, 21: 0.3, 22: 2, 26: 2}
#     ans = []
#
#     for machine_index, Machine in enumerate(main_machines):
#         start_times = Machine.O_start
#         end_times = Machine.O_end
#
#         # 计算当前机器的实际绘图位置
#         machine_id = machine_index + 1
#         adjusted_index = machine_index + sum([offset for key, offset in machine_offset.items() if machine_id >= key])
#
#         mi = []
#         mi.append(machine_id)
#
#         for task_index, (start, end) in enumerate(zip(start_times, end_times)):
#             job_serial_number = Machine.assigned_task[task_index][0]
#             color = color_map.get(job_serial_number, 'gray')
#             b = get_job_label(job_serial_number, job_groups, machine_id)
#
#             # 绘制甘特条
#             plt.barh(adjusted_index, width=end - start, height=common_params['bar_height'],
#                      left=start, color=color, edgecolor='black')
#             plt.text(x=start + (end - start) / 2, y=adjusted_index,
#                      s=b, va='center', ha='center', fontsize=common_params['fontsize'])
#
#             mi.append(b)
#
#         ans.append(mi)
#
#     print("每台机器上工件的加工顺序：", ans)
#
#     # 设置Y轴刻度标签
#     yticks = []
#     yticklabels = []
#     for i, machine in enumerate(main_machines, start=1):
#         adjusted_index = i - 1 + sum([offset for key, offset in machine_offset.items() if i >= key])
#         yticks.append(adjusted_index)
#         yticklabels.append('M{}'.format(i))
#
#     plt.yticks(yticks, yticklabels, fontsize=common_params['fontsize'])
#
#     # 添加组标签
#     group_labels = {
#         'Line1': (0 + 14) / 2 - 0.5,
#         'Line2': (15 + 21) / 2 - 1 + 2 + 0.4,
#         'Line3': (22 + 25) / 2 + 1 + 2 + 0.8
#     }
#
#     for label, position in group_labels.items():
#         plt.text(common_params['group_label_offset'], position, label,
#                  fontsize=common_params['fontsize'], rotation=90, va='center', ha='right')
#
#     # 设置统一的时间轴范围
#     plt.xlim([all_start - common_params['time_range_padding'],
#               all_end + common_params['time_range_padding']])
#
#     # 在横轴上标出tn数组中的各个时刻，并画垂直虚线
#     for t in tn:
#         t_rounded = round(t, 2)
#         plt.axvline(x=t_rounded, color='gray', linestyle='--', linewidth=0.8)
#         plt.text(x=t_rounded + 0.2, y=common_params['time_marker_offset'],
#                  s=f'{t_rounded:.2f}', ha='center', va='top',
#                  fontsize=common_params['fontsize'])
#
#     plt.ylabel('Line', labelpad=20, fontsize=common_params['fontsize'])
#     plt.xlabel('makespan (minute)', fontsize=common_params['fontsize'])
#     plt.tight_layout()
#     plt.savefig('优化后排程方案的甘特图_主产线.png', bbox_inches='tight')
#     plt.show()
#
#     # ==============================================
#     # 绘制Line4甘特图 (单独的第26台机器)
#     # ==============================================
#     if line4_machine:
#         plt.figure(figsize=(common_params['figsize'][0], 3),  # 保持宽度一致，调整高度
#                    dpi=common_params['dpi'])
#
#         Machine = line4_machine[0]
#         start_times = Machine.O_start
#         end_times = Machine.O_end
#
#         mi = []
#         mi.append(26)
#
#         for task_index, (start, end) in enumerate(zip(start_times, end_times)):
#             job_serial_number = Machine.assigned_task[task_index][0]
#             color = color_map.get(job_serial_number, 'gray')
#             b = get_job_label(job_serial_number, job_groups, 26)
#             b = 'F' + b[1:]  # 第26台机器是F开头
#
#             # 绘制甘特条，保持与主图相同的样式
#             plt.barh(0, width=end - start, height=common_params['bar_height'],
#                      left=start, color=color, edgecolor='black')
#             plt.text(x=start + (end - start) / 2, y=0,
#                      s=b, va='center', ha='center', fontsize=common_params['fontsize'])
#
#             mi.append(b)
#
#         print("Line4机器上工件的加工顺序：", mi)
#
#         # 设置Y轴
#         plt.yticks([0], ['M26'], fontsize=common_params['fontsize'])
#
#         # 设置与主图相同的时间轴范围
#         plt.xlim([all_start - common_params['time_range_padding'],
#                   all_end + common_params['time_range_padding']])
#
#         # 添加组标签
#         plt.text(common_params['group_label_offset'], 0, 'Line4',
#                  fontsize=common_params['fontsize'], rotation=90, va='center', ha='right')
#
#         # 在横轴上标出tn数组中的各个时刻，并画垂直虚线
#         for t in tn:
#             t_rounded = round(t, 2)
#             plt.axvline(x=t_rounded, color='gray', linestyle='--', linewidth=0.8)
#             plt.text(x=t_rounded + 0.2, y=common_params['time_marker_offset'],
#                      s=f'{t_rounded:.2f}', ha='center', va='top',
#                      fontsize=common_params['fontsize'])
#
#         plt.ylabel('Line', labelpad=20, fontsize=common_params['fontsize'])
#         plt.xlabel('makespan (minute)', fontsize=common_params['fontsize'])
#         plt.tight_layout()
#         plt.savefig('优化后排程方案的甘特图_Line4.png', bbox_inches='tight')
#         plt.show()
#
#
# def get_job_label(job_serial_number, job_groups, machine_id):
#     """统一获取工件标签的函数"""
#     if job_serial_number in job_groups['Reds']:
#         b = f"P{job_serial_number}A"
#     elif job_serial_number in job_groups['Blues']:
#         b = f"P{job_serial_number}B"
#     elif job_serial_number in job_groups['Greens']:
#         b = f"P{job_serial_number}C"
#     elif job_serial_number in job_groups['Oranges']:
#         b = f"P{job_serial_number}D"
#     elif job_serial_number in job_groups['Purples']:
#         b = f"P{job_serial_number}E"
#     elif job_serial_number in job_groups['YlOrBr']:
#         b = f"P{job_serial_number}F"
#     elif job_serial_number in job_groups['PuBu']:
#         b = f"P{job_serial_number}G"
#     elif job_serial_number in job_groups['Greys']:
#         b = f"P{job_serial_number}MTZ1"
#     elif job_serial_number in job_groups['BuGn']:
#         b = f"P{job_serial_number}MTZ2"
#     elif job_serial_number in job_groups['RdPu']:
#         b = f"P{job_serial_number}MTZ3"
#
#     if machine_id in (14, 21, 25, 26):
#         b = 'F' + b[1:]
#
#     return b

def run_single_experiment(run_num):
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
        print('-'*30)
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
            ans = Gantt(d.Machines, k, tn)  # 种群初始化时的最优个体 解码后 对应的甘特图
            print("每台机器上工件的加工顺序：", ans)
            print("总配套关系为：",Matching_result_all)
            print("配套时刻为：",tn)

        print("当前代最优适应度：", round(gy_best, 3))
        Best_fit.append(round(gy_best, 3))

    # 最后一次迭代，把配套关系与配套时刻打印出来
    food_mapped_individual1 = e.Individual_Coding_mapping_conversion(food)
    d = Decode(J, Processing_time, M_num, k)
    y, Matching_result_all, tn = d.decode(food_mapped_individual1, O_num)
    ans = Gantt(d.Machines, k, tn)  # 种群初始化时的最优个体 解码后 对应的甘特图
    print("每台机器上工件的加工顺序：",ans)
    print("总配套关系为：", Matching_result_all)
    print("配套时刻为：", tn)

    x = [_ for _ in range(s.Max_Itertions+1)]  # 横坐标 迭代数
    plt.plot(x, Best_fit, '-k')
    # plt.title('the fitness of each iteration')
    plt.ylabel('Fitness')
    plt.xlabel('Iteraions')
    plt.savefig('适应度收敛图.png')
    plt.show()
    print("每代最好适应度Best_fit：", Best_fit)   # Best_fit有Max_Itertions+1个，第一个为初始种群最好适应度，第2至Max_Itertions+1才对应于第1次迭代值Max_Itertions次迭代值

    end_time = time.time()  # 记录结束时间
    runtime = end_time - start_time
    print("程序运行时间为: {:.3f}秒".format(runtime))

    return {
        "run_num": run_num,
        "Matching_result_all": Matching_result_all,
        "tn": tn,
        "Best_fit": Best_fit,
        "runtime": runtime,
        "Optimal_fit": Optimal_fit,
        "Process_sequencing": ans
    }

if __name__ == '__main__':
    results = []

    # 创建结果文件
    with open('experiment_results.txt', 'w', encoding='utf-8') as f:
        f.write("实验记录 - 12次运行结果\n")
        f.write("=" * 50 + "\n\n")

    for i in range(1, 12 + 1):
        print(f"\n\n{'=' * 50}")
        print(f"开始第 {i} 次运行")
        print(f"{'=' * 50}\n")

        result = run_single_experiment(i)
        results.append(result)

        # 将每次结果写入文件
        with open('experiment_results.txt', 'a', encoding='utf-8') as f:
            f.write(f"第 {i} 次运行结果:\n")
            f.write(f"最优适应度: \n{result['Optimal_fit']}\n")
            f.write(f"每代最好适应度Best_fit: \n{result['Best_fit']}\n")
            f.write(f"每台机器上工件的加工顺序: \n{result['Process_sequencing']}\n")
            f.write(f"总配套关系为: \n{result['Matching_result_all']}\n")
            f.write(f"配套时刻为: \n{result['tn']}\n")
            f.write(f"运行时间: \n{result['runtime']:.3f}秒\n")
            f.write("-" * 50 + "\n\n")

    # 汇总统计信息
    with open('experiment_results.txt', 'a', encoding='utf-8') as f:
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

    print("\n所有实验已完成，结果已保存到 experiment_results.txt 文件中")