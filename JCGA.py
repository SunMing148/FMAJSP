import itertools
import random
import numpy as np
from Decode import Decode

class GA():
    def __init__(self, Len_Chromo, Processing_time, J, J_num, M_num, kn, Job_serial_number, Special_Machine_ID):
        self.Pop_size = 30  # 种群数量
        self.Pc = 0.8  # 交叉概率
        self.Pm = 0.3  # 变异概率
        self.Pv = 0.5  # 选择何种方式进行交叉的概率阈值
        self.Pw = 0.95  # 选择何种方式进行变异的概率阈值

        self.GS_num = int(0 * self.Pop_size)  # 全局选择初始化
        self.LS_num = int(0 * self.Pop_size)  # 局部选择初始化
        self.RS_num = int(1 * self.Pop_size)  # 随机选择初始化

        self.Max_Itertions = 300 # 最大迭代次数
        self.Len_Chromo = Len_Chromo

        self.Processing_time = Processing_time
        self.J = J
        self.J_num = J_num
        self.M_num = M_num
        self.kn = kn
        self.Job_serial_number = Job_serial_number
        self.Special_Machine_ID = Special_Machine_ID

    # 适应度
    def fitness(self, CHS):
        Fit = []
        for i in range(len(CHS)):
            d = Decode(self.J, self.Processing_time, self.M_num, self.kn, self.Job_serial_number, self.Special_Machine_ID)
            y, Matching_result_all, tn = d.decode(CHS[i], self.Len_Chromo)
            Fit.append(y)
        return Fit

    # 机器部分交叉
    def machine_cross(self, CHS1, CHS2, T0):
        """
        :param CHS1: 机器选择部分的基因1
        :param CHS2: 机器选择部分的基因2
        :param T0: 工序总数
        :return: 交叉后的机器选择部分的基因
        """
        T_r = [j for j in range(T0)]
        r = random.randint(1, 10)  # 在区间[1,T0]内产生一个整数r
        random.shuffle(T_r)
        R = T_r[0:r]  # 按照随机数r产生r个互不相等的整数
        OS_1 = CHS1[self.Len_Chromo:2 * T0]
        OS_2 = CHS2[self.Len_Chromo:2 * T0]
        MS_1 = CHS2[0:T0]
        MS_2 = CHS1[0:T0]
        for i in R:
            K, K_2 = MS_1[i], MS_2[i]
            MS_1[i], MS_2[i] = K_2, K
        CHS1 = np.hstack((MS_1, OS_1))
        CHS2 = np.hstack((MS_2, OS_2))
        return CHS1, CHS2

    # 工序部分交叉
    def operation_cross(self, CHS1, CHS2, T0, J_num):
        """
        :param CHS1: 工序选择部分的基因1
        :param CHS2: 工序选择部分的基因2
        :param T0: 工序总数
        :param J_num: 工件总数
        :return: 交叉后的工序选择部分的基因
        """
        OS_1 = CHS1[T0:2 * T0]
        OS_2 = CHS2[T0:2 * T0]
        MS_1 = CHS1[0:T0]
        MS_2 = CHS2[0:T0]
        Job_list = [i for i in range(J_num)]
        random.shuffle(Job_list)
        r = random.randint(1, J_num - 1)
        Set1 = Job_list[0:r]
        new_os = list(np.zeros(T0, dtype=int))
        for k, v in enumerate(OS_1):
            if v in Set1:
                new_os[k] = v + 1
        for i in OS_2:
            if i not in Set1:
                Site = new_os.index(0)
                new_os[Site] = i + 1
        new_os = np.array([j - 1 for j in new_os])
        CHS1 = np.hstack((MS_1, new_os))
        CHS2 = np.hstack((MS_2, new_os))
        return CHS1, CHS2

    # 机器部分变异
    def machine_variation(self, CHS, O, T0, J):
        """
        :param CHS: 机器选择部分的基因
        :param O: 加工时间矩阵
        :param T0: 工序总数
        :param J: 各工件加工信息
        :return: 变异后的机器选择部分的基因
        """
        Tr = [i_num for i_num in range(T0)]
        MS = CHS[0:T0]
        OS = CHS[T0:2 * T0]
        # 机器选择部分
        r = random.randint(1, T0 - 1)  # 在变异染色体中选择r个位置
        random.shuffle(Tr)
        T_r = Tr[0:r]
        for num in T_r:
            T_0 = [j for j in range(T0)]
            K = []
            Site = 0
            for k, v in J.items():
                K.append(T_0[Site:Site + v])
                Site += v
            for i in range(len(K)):
                if num in K[i]:
                    O_i = i
                    O_j = K[i].index(num)
                    break
            Machine_using = O[O_i][O_j]
            Machine_time = []
            for j in Machine_using:
                if j != 9999:
                    Machine_time.append(j)
            Min_index = Machine_time.index(min(Machine_time))
            MS[num] = Min_index
        CHS = np.hstack((MS, OS))
        return CHS

    # 工序部分变异
    def operation_variation(self, CHS, T0, J_num, J, O, M_num):
        """
        :param CHS: 工序选择部分的基因
        :param T0: 工序总数
        :param J_num: 工件总数
        :param J: 各工件加工信息
        :param O: 加工时间矩阵
        :param M_num: 机器总数
        :return: 变异后的工序选择部分的基因
        """
        MS = CHS[0:T0]
        OS = list(CHS[T0:2 * T0])
        r = random.randint(1, J_num - 1)
        Tr = [i for i in range(J_num)]
        random.shuffle(Tr)
        Tr = Tr[0:r]
        J_os = dict(enumerate(OS))  # 随机选择r个不同的基因
        J_os = sorted(J_os.items(), key=lambda d: d[1])
        Site = []
        for i in range(r):
            Site.append(OS.index(Tr[i]))
        A = list(itertools.permutations(Tr, r))
        A_CHS = []
        for i in range(len(A)):
            for j in range(len(A[i])):
                OS[Site[j]] = A[i][j]
            C_I = np.hstack((MS, OS))
            A_CHS.append(C_I)
        Fit = []
        for i in range(len(A_CHS)):
            d = Decode(J, O, M_num)
            Fit.append(d.decode(CHS, T0))
        return A_CHS[Fit.index(min(Fit))]

    def Select(self,Fit_value):
        Fit=[]
        for i in range(len(Fit_value)):
            fit=1/Fit_value[i]
            Fit.append(fit)
        Fit=np.array(Fit)
        idx = np.random.choice(np.arange(len(Fit_value)), size=len(Fit_value), replace=True,
                               p=(Fit) / (Fit.sum()))
        return idx

    # 生成工序准备的部分
    def OS_List(self):
        OS_list = []
        for k, v in self.J.items():
            OS_add = [k - 1 for j in range(v)]
            OS_list.extend(OS_add)
        return OS_list

    # 生成初始化矩阵
    def CHS_Matrix(self, C_num):
        return np.zeros([C_num, self.Len_Chromo], dtype=int)

    # 定位每个工件的每道工序的位置
    def Site(self, Job, Operation):
        O_num = 0
        for i in range(len(self.J)):
            if i == Job:
                return O_num + Operation
            else:
                O_num = O_num + self.J[i + 1]
        return O_num

    # 全局初始化
    def Global_initial(self):
        MS = self.CHS_Matrix(self.GS_num)  # 根据GS_num生成种群
        OS_list = self.OS_List()
        OS = self.CHS_Matrix(self.GS_num)
        for i in range(self.GS_num):
            Machine_time = np.zeros(self.M_num, dtype=int)  # 步骤1 生成一个整型数组，长度为机器数，且初始化每个元素为0
            random.shuffle(OS_list)  # 生成工序排序部分
            OS[i] = np.array(OS_list)  # 随机打乱后将其赋值给OS的某一行（因为有一个种群，第i则是赋值在OS的第i行，以此生成完整的OS）
            GJ_list = [i_1 for i_1 in range(self.J_num)]  # 生成工件集
            random.shuffle(GJ_list)  # 随机打乱工件集,为的是下一步可以随机抽出第一个工件
            for g in GJ_list:  # 选择第一个工件（由于上一步已经打乱工件集，抽出第一个也是“随机”）
                h = self.Processing_time[g]  # h为第一个工件包含的工序对应的时间矩阵
                for j in range(len(h)):  # 从此工件的第一个工序开始
                    D = h[j]  # D为第一个工件的第一个工序对应的时间矩阵
                    List_Machine_weizhi = []
                    for k in range(len(D)):  # 确定工序可用的机器位于第几个位置
                        Useing_Machine = D[k]
                        if Useing_Machine != 9999:
                            List_Machine_weizhi.append(k)
                    Machine_Select = []
                    for Machine_add in List_Machine_weizhi:  # 将机器时间数组对应位置和工序可选机器的时间相加
                        Machine_Select.append(Machine_time[Machine_add] + D[Machine_add])
                    Min_time = min(Machine_Select)  # 选出时间最小的机器
                    K = Machine_Select.index(Min_time)  # 第一次出现最小时间的位置，确定最小负荷为哪个机器,即为该工序可选择的机器里的第K个机器，并非Mk
                    I = List_Machine_weizhi[K]  # 所有机器里的第I个机器，即Mi
                    Machine_time[I] += Min_time  # 相应的机器位置加上最小时间
                    site = self.Site(g, j)  # 定位每个工件的每道工序的位置
                    MS[i][site] = K  # 即将每个工序选择的第K个机器赋值到每个工件的每道工序的位置上去 即生成MS的染色体
        CHS1 = np.hstack((MS, OS))  # 将MS和OS整合为一个矩阵
        return CHS1

    # 局部初始化
    def Local_initial(self):
        MS = self.CHS_Matrix(self.LS_num)  # 根据LS_num生成局部选择的种群大小
        OS_list = self.OS_List()
        OS = self.CHS_Matrix(self.LS_num)
        for i in range(self.LS_num):
            random.shuffle(OS_list)  # 生成工序排序部分
            OS[i] = np.array(OS_list)  # 随机打乱后将其赋值给OS的某一行（因为有一个种群，第i则是赋值在OS的第i行，以此生成完整的OS）
            GJ_List = [i_1 for i_1 in range(self.J_num)]  # 生成工件集
            for g in GJ_List:  # 选择第一个工件（注意：不用随机打乱了）
                Machine_time = np.zeros(self.M_num,
                                        dtype=int)  # 设置一个整型数组 并初始化每一个元素为0，由于局部初始化，每个工件的所有工序结束后都要重新初始化，所以和全局初始化不同，此步骤应放在此处
                h = self.Processing_time[g]  # h为第一个工件包含的工序对应的时间矩阵
                for j in range(len(h)):  # 从选择的工件的第一个工序开始
                    D = h[j]  # 此工件第一个工序对应的机器加工时间矩阵
                    List_Machine_weizhi = []
                    for k in range(len(D)):  # 确定工序可用的机器位于第几个位置
                        Useing_Machine = D[k]
                        if Useing_Machine != 9999:
                            List_Machine_weizhi.append(k)
                    Machine_Select = []
                    for Machine_add in List_Machine_weizhi:  # 将机器时间数组对应位置和工序可选机器的时间相加
                        Machine_Select.append(Machine_time[Machine_add] + D[Machine_add])
                    Min_time = min(Machine_Select)  # 选出这些时间里最小的
                    K = Machine_Select.index(Min_time)  # 第一次出现最小时间的位置，确定最小负荷为哪个机器,即为该工序可选择的机器里的第K个机器，并非Mk
                    I = List_Machine_weizhi[K]  # 所有机器里的第I个机器，即Mi
                    Machine_time[I] += Min_time
                    site = self.Site(g, j)  # 定位每个工件的每道工序的位置
                    MS[i][site] = K  # 即将每个工序选择的第K个机器赋值到每个工件的每道工序的位置上去
        CHS1 = np.hstack((MS, OS))  # 将MS和OS整合为一个矩阵
        return CHS1

    # 随机初始化
    def Random_initial(self):
        MS = self.CHS_Matrix(self.RS_num)  # 根据RS_num生成随机选择的种群大小
        OS_list = self.OS_List()
        OS = self.CHS_Matrix(self.RS_num)
        for i in range(self.RS_num):
            random.shuffle(OS_list)
            OS[i] = np.array(OS_list)
            GJ_List = [i_1 for i_1 in range(self.J_num)]  # 生成工件集
            for g in GJ_List:  # 选择第一个工件
                h = self.Processing_time[g]
                for j in range(len(h)):  # 选择第一个工件的第一个工序
                    D = h[j]  # 此工件第一个工序可加工的机器对应的时间矩阵
                    List_Machine_weizhi = []
                    for k in range(len(D)):
                        Useing_Machine = D[k]
                        if Useing_Machine != 9999:
                            List_Machine_weizhi.append(k)
                    number = random.choice(List_Machine_weizhi)  # 从可选择的机器编号中随机选择一个（此编号就是机器编号）
                    K = List_Machine_weizhi.index(number)  # 即为该工序可选择的机器里的第K个机器，并非Mk
                    site = self.Site(g, j)  # 定位每个工件的每道工序的位置
                    MS[i][site] = K  # 即将每个工序选择的第K个机器赋值到每个工件的每道工序的位置上去
        CHS1 = np.hstack((MS, OS))
        return CHS1
