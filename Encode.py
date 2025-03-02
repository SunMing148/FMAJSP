import random

import numpy as np
import math

class Encode:
    def __init__(self, Matrix, J, J_num, M_num):
        """
        :param Matrix: 机器加工时间矩阵
        # :param Pop_size: 种群数量
        :param J: 各工件对应的工序数
        :param J_num: 工件数
        :param M_num: 机器数
        """
        self.Matrix = Matrix
        self.J = J
        self.J_num = J_num
        self.M_num = M_num
        self.CHS = []

        self.MS_base = []
        self.OS_base = []

        self.Len_Chromo = 0
        for i in J.values():
            self.Len_Chromo += i

    def Get_Map_base_value(self):
        GJ_List = [i_1 for i_1 in range(self.J_num)]  # 生成工件集
        for g in GJ_List:  # 选择第一个工件
            h = self.Matrix[g]
            for j in range(len(h)):  # 选择第一个工件的第一个工序
                D = h[j]  # 此工件第一个工序可加工的机器对应的时间矩阵
                List_Machine_weizhi = []
                for k in range(len(D)):
                    Useing_Machine = D[k]
                    if Useing_Machine != 9999:
                        List_Machine_weizhi.append(k)
                self.MS_base.append(len(List_Machine_weizhi))

        for k, v in self.J.items():
            OS_add = [k - 1 for j in range(v)]
            self.OS_base.extend(OS_add)

        return self.MS_base, self.OS_base

    def Coding_mapping_conversion(self, CSO):
        CSO_mapped = []
        for individual in CSO:  # 遍历种群中的每个个体
            # 前半部分处理
            front_half = [math.floor(a_val * b_val) for a_val, b_val in zip(individual[ :self.Len_Chromo], self.MS_base)]
            # 后半部分处理
            # 使用zip将a和b配对，并根据a中的值排序
            sorted_pairs = sorted(zip(individual[self.Len_Chromo: ], self.OS_base))
            # 提取排序后b对应的值
            back_half = [pair[1] for pair in sorted_pairs]
            # 合并前后两部分
            mapped_individual = front_half + back_half
            CSO_mapped.append(mapped_individual)
        return CSO_mapped

    def Individual_Coding_mapping_conversion(self, individual):
        # 前半部分处理
        front_half = [math.floor(a_val * b_val) for a_val, b_val in zip(individual[ :self.Len_Chromo], self.MS_base)]
        # 后半部分处理
        # 使用zip将a和b配对，并根据a中的值排序
        sorted_pairs = sorted(zip(individual[self.Len_Chromo: ], self.OS_base))
        # 提取排序后b对应的值
        back_half = [pair[1] for pair in sorted_pairs]
        # 合并前后两部分
        mapped_individual = front_half + back_half
        return mapped_individual


