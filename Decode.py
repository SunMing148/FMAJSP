import copy
import numpy as np

from Job import Job
from Machine import Machine_Time_window


class Decode:
    def __init__(self, J, Processing_time, M_num):
        """
        :param J: 各工件对应的工序数字典
        :param Processing_time: 各工件的加工时间矩阵
        :param M_num: 加工机器数
        """
        self.Processing_time = Processing_time   # 就是Instance里的Processing_time
        self.M_num = M_num
        self.J = J
        self.Machines = []  # 存储机器类 一维数组  数组长度等于M_num机器数量 数组元素为Machine_Time_window(j)的对象
        self.Scheduled = []  # 已经排产过的工序
        self.fitness = 0  # 适应度
        self.Machine_State = np.zeros(M_num, dtype=int)  # 在机器上加工的工件是哪个 一维数组 数组元素全是0 数组长度等于M_num机器数量
        self.Jobs = []  # 存储工件类
        for j in range(M_num):
            self.Machines.append(Machine_Time_window(j))  # 为每一台机器都创建了一个Machine_Time_window对象
        for k, v in J.items():
            self.Jobs.append(Job(k, v))          # 为每一个工件都创建了一个Job对象

    # 时间顺序矩阵和机器顺序矩阵，根据基因的MS部分转换
    def Order_Matrix(self, MS):
        JM = []
        T = []
        Ms_decompose = []
        Site = 0
        # 按照基因的MS部分按工件序号划分
        for S_i in self.J.values():
            Ms_decompose.append(MS[Site:Site + S_i])
            Site += S_i
        for i in range(len(Ms_decompose)):  # len(Ms_decompose)表示工件数
            JM_i = []
            T_i = []
            for j in range(len(Ms_decompose[i])):  # len(Ms_decompose[i])表示每一个工件对应的工序数
                O_j = self.Processing_time[i][j]  # 工件i的工序j可选择的加工时间列表
                M_ij = []
                T_ij = []
                for Mac_num in range(len(O_j)):  # 寻找MS对应部分的机器时间和机器顺序
                    if O_j[Mac_num] != 9999:
                        M_ij.append(Mac_num)
                        T_ij.append(O_j[Mac_num])
                    else:
                        continue
                JM_i.append(M_ij[Ms_decompose[i][j]])
                T_i.append(T_ij[Ms_decompose[i][j]])
            JM.append(JM_i)
            T.append(T_i)
        return JM, T

    # 确定工序的最早加工时间
    def Earliest_Start(self, Job, O_num, Machine):
        P_t = self.Processing_time[Job][O_num][Machine]
        last_O_end = self.Jobs[Job].Last_Processing_end_time  # 上道工序结束时间
        Selected_Machine = Machine
        M_window = self.Machines[Selected_Machine].Empty_time_window()  # 当前机器的空格时间
        M_Tstart = M_window[0]
        M_Tend = M_window[1]
        M_Tlen = M_window[2]
        Machine_end_time = self.Machines[Selected_Machine].End_time
        ealiest_start = max(last_O_end, Machine_end_time)
        if M_Tlen is not None:  # 此处为全插入时窗
            for le_i in range(len(M_Tlen)):
                # 当前空格时间比加工时间大可插入
                if M_Tlen[le_i] >= P_t:
                    # 当前空格开始时间比该工件上一工序结束时间大可插入该空格，以空格开始时间为这一工序开始
                    if M_Tstart[le_i] >= last_O_end:
                        ealiest_start = M_Tstart[le_i]
                        break
                    # 当前空格开始时间比该工件上一工序结束时间小但空格可满足插入该工序，以该工序的上一工序的结束为开始
                    if M_Tstart[le_i] < last_O_end and M_Tend[le_i] - last_O_end >= P_t:
                        ealiest_start = last_O_end
                        break
        M_Ealiest = ealiest_start  # 当前工件当前工序的最早开始时间
        End_work_time = M_Ealiest + P_t  # 当前工件当前工序的结束时间
        return M_Ealiest, Selected_Machine, P_t, O_num, last_O_end, End_work_time

    # 解码操作
    def decode(self, CHS, Len_Chromo):
        """
        :param CHS: 种群基因
        :param Len_Chromo: MS与OS的分解线
        :return: 适应度，即最大加工时间
        """
        MS = list(CHS[0:Len_Chromo])
        OS = list(CHS[Len_Chromo:2 * Len_Chromo])
        Needed_Matrix = self.Order_Matrix(MS)
        JM = Needed_Matrix[0]
        for i in OS:
            if i in [5,14,23]: #6,15,24
                continue
            Job = i
            O_num = self.Jobs[Job].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[Job][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            Para = self.Earliest_Start(Job, O_num, Machine)
            # Para[0] ：M_Ealiest 当前工件当前工序的最早开始时间
            # Para[5] ：End_work_time 当前工件当前工序的结束时间
            # Para[1] ：Selected_Machine 所选择的机器
            self.Jobs[Job]._Input(Para[0], Para[5], Para[1])  # 工件完成该工序
            if Para[5] > self.fitness:
                self.fitness = Para[5]
            # Para[0] ：M_Ealiest 当前工件当前工序的最早开始时间
            # Para[2] ：P_t 对应具体工件的工序的具体机器加工时间
            # Para[3] ：O_num 现在加工的工序 当前工件的第几道工序
            self.Machines[Machine]._Input(Job, Para[0], Para[2], Para[3])  # 机器完成该工件该工序



        a = copy.deepcopy(self.Machines[20 - 1].O_end)
        b = copy.deepcopy(self.Machines[20 - 1].assigned_task)
        c = copy.deepcopy(self.Machines[17 - 1].O_end)
        d = copy.deepcopy(self.Machines[17 - 1].assigned_task)


        OS_Mechanism_Assembly = [num for num in OS if num in {5, 14, 23}]   # 获得OS中的5，14，23这三个，并保留其在OS中的顺序

        if OS_Mechanism_Assembly == [5, 14, 23]: #完成
            O_num = self.Jobs[5].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[5][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[5][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start1 = max(Machine_end_time,min(a[b.index([5, 3])], a[b.index([14, 3])]), min(c[d.index([4, 3])], c[d.index([13, 3])]))
            End_work_time = start1 + P_t
            self.Jobs[5]._Input(start1, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(5, start1, P_t, O_num)  # 机器完成该工件该工序


            O_num = self.Jobs[14].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[14][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[14][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start2 = max(Machine_end_time,max(max(a[b.index([5,3])],a[b.index([14,3])]),max(c[d.index([4,3])],c[d.index([13,3])])))
            End_work_time = start2 + P_t
            self.Jobs[14]._Input(start2, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(14, start2, P_t, O_num)  # 机器完成该工件该工序

            O_num = self.Jobs[23].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[23][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[23][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start3 = max(Machine_end_time,a[b.index([23,3])],c[d.index([22,3])])
            End_work_time = start3 + P_t
            self.Jobs[23]._Input(start3, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(23, start3, P_t, O_num)  # 机器完成该工件该工序

        elif OS_Mechanism_Assembly == [5, 23, 14]: #完成
            O_num = self.Jobs[5].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[5][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[5][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start1 = max(Machine_end_time,min(a[b.index([5, 3])], a[b.index([14, 3])]), min(c[d.index([4, 3])], c[d.index([13, 3])]))
            End_work_time = start1 + P_t
            self.Jobs[5]._Input(start1, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(5, start1, P_t, O_num)  # 机器完成该工件该工序

            O_num = self.Jobs[23].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[23][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[23][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start2 = max(Machine_end_time, a[b.index([23, 3])], c[d.index([22, 3])])
            End_work_time = start2 + P_t
            self.Jobs[23]._Input(start2, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(23, start2, P_t, O_num)  # 机器完成该工件该工序

            O_num = self.Jobs[14].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[14][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[14][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start3 = max(Machine_end_time, max(max(a[b.index([5, 3])], a[b.index([14, 3])]),
                                      max(c[d.index([4, 3])], c[d.index([13, 3])])))
            End_work_time = start3 + P_t
            self.Jobs[14]._Input(start3, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(14, start3, P_t, O_num)  # 机器完成该工件该工序

        elif OS_Mechanism_Assembly == [14, 5, 23]: #完成
            O_num = self.Jobs[14].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[14][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[14][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start1 = max(Machine_end_time,min(a[b.index([5, 3])], a[b.index([14, 3])]), min(c[d.index([4, 3])], c[d.index([13, 3])]))
            End_work_time = start1 + P_t
            self.Jobs[14]._Input(start1, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(14, start1, P_t, O_num)  # 机器完成该工件该工序

            O_num = self.Jobs[5].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[5][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[5][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start2 = max(Machine_end_time, max(max(a[b.index([5, 3])], a[b.index([14, 3])]),
                                      max(c[d.index([4, 3])], c[d.index([13, 3])])))
            End_work_time = start2 + P_t
            self.Jobs[5]._Input(start2, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(5, start2, P_t, O_num)  # 机器完成该工件该工序


            O_num = self.Jobs[23].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[23][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[23][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start3 = max(Machine_end_time, a[b.index([23, 3])], c[d.index([22, 3])])
            End_work_time = start3 + P_t
            self.Jobs[23]._Input(start3, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(23, start3, P_t, O_num)  # 机器完成该工件该工序

        elif OS_Mechanism_Assembly == [14, 23, 5]: #完成
            O_num = self.Jobs[14].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[14][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[14][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start1 = max(Machine_end_time,min(a[b.index([5, 3])], a[b.index([14, 3])]), min(c[d.index([4, 3])], c[d.index([13, 3])]))
            End_work_time = start1 + P_t
            self.Jobs[14]._Input(start1, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(14, start1, P_t, O_num)  # 机器完成该工件该工序


            O_num = self.Jobs[23].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[23][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[23][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start2 = max(Machine_end_time, a[b.index([23, 3])], c[d.index([22, 3])])
            End_work_time = start2 + P_t
            self.Jobs[23]._Input(start2, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(23, start2, P_t, O_num)  # 机器完成该工件该工序


            O_num = self.Jobs[5].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[5][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[5][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start3 = max(Machine_end_time, max(max(a[b.index([5, 3])], a[b.index([14, 3])]),
                                      max(c[d.index([4, 3])], c[d.index([13, 3])])))
            End_work_time = start3 + P_t
            self.Jobs[5]._Input(start3, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(5, start3, P_t, O_num)  # 机器完成该工件该工序

        elif OS_Mechanism_Assembly == [23, 14, 5]:
            O_num = self.Jobs[23].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[23][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[23][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start1 = max(Machine_end_time,a[b.index([23, 3])], c[d.index([22, 3])])
            End_work_time = start1 + P_t
            self.Jobs[23]._Input(start1, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(23, start1, P_t, O_num)  # 机器完成该工件该工序


            O_num = self.Jobs[14].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[14][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[14][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start2 = max(Machine_end_time,max(min(a[b.index([5, 3])], a[b.index([14, 3])]), min(c[d.index([4, 3])], c[d.index([13, 3])])))
            End_work_time = start2 + P_t
            self.Jobs[14]._Input(start2, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(14, start2, P_t, O_num)  # 机器完成该工件该工序

            O_num = self.Jobs[5].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[5][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[5][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start3 = max(Machine_end_time, max(max(a[b.index([5, 3])], a[b.index([14, 3])]),
                                      max(c[d.index([4, 3])], c[d.index([13, 3])])))
            End_work_time = start3 + P_t
            self.Jobs[5]._Input(start3, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(5, start3, P_t, O_num)  # 机器完成该工件该工序

        elif OS_Mechanism_Assembly == [23, 5, 14]:
            O_num = self.Jobs[23].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[23][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[23][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start1 = max(Machine_end_time,a[b.index([23, 3])], c[d.index([22, 3])])
            End_work_time = start1 + P_t
            self.Jobs[23]._Input(start1, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(23, start1, P_t, O_num)  # 机器完成该工件该工序

            O_num = self.Jobs[5].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[5][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[5][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start2 = max(Machine_end_time,max(min(a[b.index([5, 3])], a[b.index([14, 3])]), min(c[d.index([4, 3])], c[d.index([13, 3])])))
            End_work_time = start2 + P_t
            self.Jobs[5]._Input(start2, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(5, start2, P_t, O_num)  # 机器完成该工件该工序

            O_num = self.Jobs[14].Current_Processed()  # 现在加工的工序 当前工件的第几道工序
            Machine = JM[14][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号
            P_t = self.Processing_time[14][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start3 = max(Machine_end_time, max(max(a[b.index([5, 3])], a[b.index([14, 3])]),
                                      max(c[d.index([4, 3])], c[d.index([13, 3])])))
            End_work_time = start3 + P_t
            self.Jobs[14]._Input(start3, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(14, start3, P_t, O_num)  # 机器完成该工件该工序


        return self.fitness
