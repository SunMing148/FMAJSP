import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Decode import Decode

# WOA 这里并没涉及WOA算法
class WOA():
    def __init__(self, Len_Chromo, Processing_time, J, M_num, kn, Job_serial_number, Special_Machine_ID):
        self.Pop_size = 30  # 种群数量

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


    def WOA_initial(self):
        # 随机生成
        X = self.lb + np.random.random_sample((self.Pop_size, self.Len_Chromo*2)) * (self.ub - self.lb)
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



