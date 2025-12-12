import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Decode import Decode

# IABC
class ABC():
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

    def ABC_initial(self):
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

    def SortFitness(self, Fit):
        '''适应度排序'''
        '''
        输入为适应度值
        输出为排序后的适应度值，和索引
        '''
        fitness = np.sort(Fit, axis=0)
        index = np.argsort(Fit, axis=0)
        return fitness, index

    def SortPosition(self, X, index):
        '''根据适应度对位置进行排序'''
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnew[i, :] = X[index[i], :]
        return Xnew

    def RouletteWheelSelection(self, P):
        '''轮盘赌策略'''
        C = np.cumsum(P)  # 累加
        r = np.random.random() * C[-1]  # 定义选择阈值，将随机概率与总和的乘积作为阈值
        out = 0
        # 若大于或等于阈值，则输出当前索引，并将其作为结果，循环结束
        for i in range(P.shape[0]):
            if r < C[i]:
                out = i
                break
        return out