import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Decode import Decode

# MWOA 这里并没涉及WOA算法
class WOA():
    def __init__(self, Len_Chromo, Processing_time, J, M_num, kn, Job_serial_number, Special_Machine_ID):
        self.Pop_size = 60  # 种群数量

        self.Max_Itertions = 300 # 最大迭代次数
        self.Len_Chromo = Len_Chromo

        self.vec_flag = [1, -1]
        self.ub = 1
        self.lb = 0

        self.Processing_time = Processing_time
        self.J = J
        self.M_num = M_num
        self.kn = kn
        self.Job_serial_number = Job_serial_number
        self.Special_Machine_ID = Special_Machine_ID

    def logistic_map(self, N, dim, miu=4.0, epsilon=1e-10):
        """
        使用Logistic映射生成混沌矩阵（miu=4时需特殊处理）

        参数:
        miu: 混沌系数，设为4.0时进入完全混沌状态
        epsilon: 避免边界值的小量
        """
        # 初始值范围调整为 [epsilon, 1-epsilon]
        logistic = epsilon + (1 - 2 * epsilon) * np.random.rand(N, dim)

        for i in range(N):
            for j in range(1, dim):
                # 计算Logistic映射
                logistic[i, j] = miu * logistic[i, j - 1] * (1 - logistic[i, j - 1])

                # 确保结果在(0,1)开区间内（处理舍入误差）
                if logistic[i, j] <= 0:
                    logistic[i, j] = epsilon
                elif logistic[i, j] >= 1:
                    logistic[i, j] = 1 - epsilon

        return logistic

    def WOA_initial(self):
        """
        混合初始化方法：20%随机生成 + 80% Logistic映射生成
        """
        # 计算随机生成和Logistic映射生成的个体数量
        random_size = int(self.Pop_size * 0.2)
        logistic_size = self.Pop_size - random_size
        # 随机生成部分
        X_random = self.lb + np.random.random_sample((random_size, self.Len_Chromo * 2)) * (self.ub - self.lb)
        # Logistic映射生成部分
        X_logistic = self.logistic_map(logistic_size, self.Len_Chromo * 2)
        # 合并两部分个体
        X = np.vstack((X_random, X_logistic))
        # 打乱顺序（可选）
        np.random.shuffle(X)
        return X

    def fitness(self, e, CHS, Len):
        # 种群映射转换
        CHS = e.Coding_mapping_conversion(CHS)
        Fit = []
        for i in range(len(CHS)):
            d = Decode(self.J, self.Processing_time, self.M_num, self.kn, self.Job_serial_number, self.Special_Machine_ID)
            y, Matching_result_all, tn = d.decode(CHS[i], Len)
            Fit.append(y)
        return Fit

    # 复现论文中的最优混沌策略
    def optimal_chaos_strategy(self, e, Leader_pos, Leader_score, epsilon=1e-10):
        # 应用线性映射公式，映射到（-1，1）
        min_val = np.min(Leader_pos)
        max_val = np.max(Leader_pos)
        mapped_Leader_pos = 2 * (Leader_pos - min_val + epsilon) / (max_val - min_val) - 1
        mapped_Leader_pos[mapped_Leader_pos <= -1] = -1+epsilon
        new_Leader_pos = Leader_pos
        new_Leader_score = Leader_score
        # 迭代次数（可根据需求设置，比如算法里的迭代步长）
        iter_num = 3
        for t in range(1, iter_num):
            # 逐维应用混沌公式更新
            for d in range(mapped_Leader_pos.shape[0]):
                mapped_Leader_pos[d] = 1 - 2 * (mapped_Leader_pos[d] ** 2)
            current_mapped_Leader_pos = mapped_Leader_pos.copy()
            current_mapped_Leader_pos_to_original_range = (max_val - min_val) / 2 * current_mapped_Leader_pos + (max_val - min_val) / 2
            current_mapped_Leader_pos_to_original_range[current_mapped_Leader_pos_to_original_range < 0] = epsilon
            d = Decode(self.J, self.Processing_time, self.M_num, self.kn, self.Job_serial_number, self.Special_Machine_ID)
            current_mapped_Leader_pos_to_original_range = e.Individual_Coding_mapping_conversion(current_mapped_Leader_pos_to_original_range)
            y, Matching_result_all, tn = d.decode(current_mapped_Leader_pos_to_original_range, self.Len_Chromo)
            if y < Leader_score:
                new_Leader_score = y
                new_Leader_pos = current_mapped_Leader_pos_to_original_range
        return new_Leader_pos, new_Leader_score
