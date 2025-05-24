import copy
import numpy as np
from Job import Job
from Machine import Machine_Time_window

class Decode:
    def __init__(self, J, Processing_time, M_num, k):
        """
        :param J: 各工件对应的工序数字典
        :param Processing_time: 各工件的加工时间矩阵
        :param M_num: 加工机器数
        """
        self.k = k     # instance中的k
        self.n = 3*k   # 总成品数
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

        self.Ap=[5,14,23]  # 机器21装配工序 （第一次装配工序） 对应的Job编号（编号减1）
        temp_Ap = copy.deepcopy(self.Ap)
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in temp_Ap]  # 计算新的元素值
            self.Ap.extend(new_values)  # 将新元素添加到数组中

        self.D = [6, 15]
        temp_D = copy.deepcopy(self.D)
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in temp_D]  # 计算新的元素值
            self.D.extend(new_values)  # 将新元素添加到数组中

        self.E = [24]
        temp_E = copy.deepcopy(self.E)
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in temp_E]  # 计算新的元素值
            self.E.extend(new_values)  # 将新元素添加到数组中

        self.D_component1 = [4, 13]
        temp_D_component1 = copy.deepcopy(self.D_component1)
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in temp_D_component1]  # 计算新的元素值
            self.D_component1.extend(new_values)  # 将新元素添加到数组中

        self.D_component2 = [5, 14]
        temp_D_component2 = copy.deepcopy(self.D_component2)
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in temp_D_component2]  # 计算新的元素值
            self.D_component2.extend(new_values)  # 将新元素添加到数组中

        self.E_component1 = [22]
        temp_E_component1 = copy.deepcopy(self.E_component1)
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in temp_E_component1]  # 计算新的元素值
            self.E_component1.extend(new_values)  # 将新元素添加到数组中

        self.E_component2 = [23]
        temp_E_component2 = copy.deepcopy(self.E_component2)
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in temp_E_component2]  # 计算新的元素值
            self.E_component2.extend(new_values)  # 将新元素添加到数组中

        self.A = [1, 2, 3]
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in copy.deepcopy(self.A)]  # 计算新的元素值
            self.A.extend(new_values)  # 将新元素添加到数组中

        self.B = [10, 11, 12]
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in copy.deepcopy(self.B)]  # 计算新的元素值
            self.B.extend(new_values)  # 将新元素添加到数组中

        self.C = [19, 20, 21]
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in copy.deepcopy(self.C)]  # 计算新的元素值
            self.C.extend(new_values)  # 将新元素添加到数组中

        self.F = [7, 8, 9]
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in copy.deepcopy(self.F)]  # 计算新的元素值
            self.F.extend(new_values)  # 将新元素添加到数组中

        self.G = [16, 17, 18, 25, 26, 27]
        for i in range(1, self.k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in copy.deepcopy(self.G)]  # 计算新的元素值
            self.G.extend(new_values)  # 将新元素添加到数组中

    def find_earliest_completion(self,machine_data, component_list):
        """
        查找特定组件列表中最早完成加工的时间及对应的组件编号。
        :param machine_data: 机器的加工数据（如Machine_processing_data_17）。
        :param component_list: 组件列表（如D_component1）。
        :return: 最早完成时间及对应的组件编号。
        """
        earliest_time = float('inf')
        component_id = None
        for item in machine_data:
            if item[0] in component_list and item[1] < earliest_time:
                earliest_time = item[1]
                component_id = item[0]
        return earliest_time, component_id

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

    # 总装配配套结果
    def calculate_schedule(self, machine_14, machine_21, machine_25, Matching_result_21):
        # 构建组件编号到类型的映射
        groups = {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'D': self.D,
            'E': self.E,
            'F': self.F,
            'G': self.G,
        }
        component_type = {}
        for type_name, nums in groups.items():
            for num in nums:
                component_type[num] = type_name

        # 初始化各组件类型的加工队列
        component_queues = {t: [] for t in groups.keys()}
        machine_data = {
            14: machine_14,
            21: machine_21,
            25: machine_25,
        }
        for machine_id, data in machine_data.items():
            for entry in data:
                component_id, time = entry
                type_name = component_type.get(component_id)
                if type_name:
                    component_queues[type_name].append((component_id, time))

        # 成品需求及组件顺序
        product_requirements = {
            'O': {'A': 3, 'D': 1, 'F': 3},
            'P': {'B': 3, 'D': 1, 'G': 3},
            'Q': {'C': 3, 'E': 1, 'G': 3}
        }
        req_order = {
            'O': ['A', 'D', 'F'],
            'P': ['B', 'D', 'G'],
            'Q': ['C', 'E', 'G']
        }

        t = []
        Matching_result_all = []  # 总装配套结果

        for step in range(1, self.n + 1):
            candidates = []
            # 检查每个成品是否可生产
            for product in ['O', 'P', 'Q']:
                req = product_requirements[product]
                # 检查组件是否足够
                valid = True
                for comp_type, count in req.items():
                    if len(component_queues[comp_type]) < count:
                        valid = False
                        break
                if not valid:
                    continue

                # 计算该成品的最早完成时间
                components_info = {}
                max_time = 0
                for comp_type, count in req.items():
                    comps = component_queues[comp_type][:count]
                    times = [c[1] for c in comps]
                    current_max = max(times)
                    if current_max > max_time:
                        max_time = current_max
                    components_info[comp_type] = comps
                candidates.append((max_time, product, components_info))

            if not candidates:
                break  # 理论上不会发生

            # 选择时间最早且优先级高的成品
            candidates.sort(key=lambda x: (x[0], ['O', 'P', 'Q'].index(x[1])))
            selected_time, selected_product, selected_components = candidates[0]

            t.append(selected_time)

            # 构建配套信息
            parts = []
            for comp_type in req_order[selected_product]:
                comps = selected_components[comp_type]
                ids = tuple(c[0] for c in comps)
                parts.append(ids)
            matching_entry = [(step, selected_product)] + parts
            Matching_result_all.append(matching_entry)

            # 移除已使用的组件
            for comp_type, count in product_requirements[selected_product].items():
                component_queues[comp_type] = component_queues[comp_type][count:]

            # 新增逻辑：将数组Matching_result_21填充到Matching_result_all
            Matching_result_21_dict = {item[0]: item for item in Matching_result_21}
            for entry in Matching_result_all:
                # 提取第三个元组的第一个组件ID（即D/E组件的编号）
                key = entry[2][0]
                # 用a中对应的元组替换原元组
                entry[2] = Matching_result_21_dict[key]

        return t, Matching_result_all

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
            if i in self.Ap:
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

        Machine_processing_data_20 = [[pair[0], a_val] for pair, a_val in zip(b, a)]
        Machine_processing_data_17 = [[pair[0], c_val] for pair, c_val in zip(d, c)]

        D = copy.deepcopy(self.D)
        E = copy.deepcopy(self.E)
        D_component1 = copy.deepcopy(self.D_component1)
        D_component2 = copy.deepcopy(self.D_component2)
        E_component1 = copy.deepcopy(self.E_component1)
        E_component2 = copy.deepcopy(self.E_component2)

        Matching_result_21 = []   # 机器21装配工序 装配配套结果

        while D or E:
            finished_item = -1
            comp1_id = -1
            comp2_id = -1
            earliest_time = -1

            # 对于D类型成品
            time_D_comp1, comp1_id_D = self.find_earliest_completion(Machine_processing_data_17, D_component1)
            time_D_comp2, comp2_id_D = self.find_earliest_completion(Machine_processing_data_20, D_component2)
            earliest_time_D = max(time_D_comp1, time_D_comp2)

            # 对于E类型成品
            time_E_comp1, comp1_id_E = self.find_earliest_completion(Machine_processing_data_17, E_component1)
            time_E_comp2, comp2_id_E = self.find_earliest_completion(Machine_processing_data_20, E_component2)
            earliest_time_E = max(time_E_comp1, time_E_comp2)

            if (not D) or (E and earliest_time_E < earliest_time_D):
                # 加工E类型成品
                if E:
                    finished_item = E.pop(0)
                    comp1_id = comp1_id_E
                    comp2_id = comp2_id_E
                    earliest_time = earliest_time_E
            else:
                # 加工D类型成品
                if D:
                    finished_item = D.pop(0)
                    comp1_id = comp1_id_D
                    comp2_id = comp2_id_D
                    earliest_time = earliest_time_D

            # 使用列表推导式过滤已配套的工件
            Matching_result_21.append((finished_item, comp1_id, comp2_id))
            current_job = finished_item - 1
            Machine_processing_data_17 = [sublist for sublist in Machine_processing_data_17 if sublist[0] != comp1_id]
            Machine_processing_data_20 = [sublist for sublist in Machine_processing_data_20 if sublist[0] != comp2_id]
            O_num = self.Jobs[current_job].Current_Processed()  # 现在加工的工序 当前工件的第几道工序       0
            Machine = JM[current_job][O_num]  # 用基因的OS部分的工件序号以及工序序号索引机器顺序矩阵的机器序号  20  21-1
            P_t = self.Processing_time[current_job][O_num][Machine]  # P_t对应具体工件的工序的具体机器加工时间
            Machine_end_time = self.Machines[Machine].End_time
            start = max(Machine_end_time, earliest_time)
            End_work_time = start + P_t
            self.Jobs[current_job]._Input(start, End_work_time, Machine)  # 工件完成该工序
            if End_work_time > self.fitness:
                self.fitness = End_work_time
            self.Machines[Machine]._Input(current_job, start, P_t, O_num)  # 机器完成该工件该工序


        e = copy.deepcopy(self.Machines[25 - 1].O_end)
        f = copy.deepcopy(self.Machines[25 - 1].assigned_task)
        g = copy.deepcopy(self.Machines[21 - 1].O_end)
        h = copy.deepcopy(self.Machines[21 - 1].assigned_task)
        i = copy.deepcopy(self.Machines[14 - 1].O_end)
        j = copy.deepcopy(self.Machines[14 - 1].assigned_task)

        Machine_processing_data_25 = [[pair[0], a_val] for pair, a_val in zip(f, e)]
        Machine_processing_data_21 = [[pair[0], c_val] for pair, c_val in zip(h, g)]
        Machine_processing_data_14 = [[pair[0], c_val] for pair, c_val in zip(j, i)]

        t, Matching_result_all = self.calculate_schedule(Machine_processing_data_14, Machine_processing_data_21,
                                                         Machine_processing_data_25, Matching_result_21)

        # print("Matching_result_21:", Matching_result_21)
        # print("t =", t)
        # print("Matching_result_all =", Matching_result_all)
        # print("fitness=", self.fitness)   # 目标函数还是最小化最大完工时间

        # self.fitness = sum(index * value for index, value in enumerate(t, start=1))   #更改目标函数了

        return self.fitness, Matching_result_all

