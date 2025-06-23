import math
import random
import numpy as np
from Decode import Decode
from Instance import Processing_time, J, M_num

# SO 未改进 标准SO
class SO():
    def __init__(self, Len_Chromo):
        self.Pop_size = 100  # 种群数量

        self.C1 = 0.5     #
        self.C2 = 0.05  # 之前是0.5，改成0.05好像更好了，也不一定
        self.C3 = 2  #

        self.food_threshold = 0.28        # 有没有食物的阈值
        self.temp_threshold = 0.6         # 温度适不适合交配的阈值
        self.model_threshold = 0.62        # 模式阈值,当产生的随机值小于模式阈值就进入战斗模式，否则就进入交配模式

        self.Max_Itertions = 125  # 最大迭代次数
        self.Len_Chromo = Len_Chromo

        self.vec_flag = [1, -1]
        self.ub = 1
        self.lb = 0

    # def pwlc_map(self, N, dim, p=0.5, epsilon=1e-10):
    #     """
    #     使用分段线性混沌映射(PWLCM)生成指定大小的矩阵。
    #
    #     参数:
    #     N: int - 矩阵的行数。
    #     dim: int - 矩阵的列数。
    #     p: float - 分段点的位置，默认为0.5。
    #     epsilon: float - 用于避免取到0和1的小量。
    #
    #     返回:
    #     result: np.ndarray - 使用PWLCM生成的矩阵。
    #     """
    #     Piecewise = np.random.rand(N, dim)
    #     for i in range(N):
    #         for j in range(1, dim):
    #             if 0 < Piecewise[i, j - 1] <= p - epsilon:
    #                 Piecewise[i, j] = Piecewise[i, j - 1] / p + epsilon
    #             elif p < Piecewise[i, j - 1] <= 0.5 - epsilon:
    #                 Piecewise[i, j] = (Piecewise[i, j - 1] - p) / (0.5 - p) + epsilon
    #             elif 0.5 < Piecewise[i, j - 1] <= 1 - p - epsilon:
    #                 Piecewise[i, j] = (1 - p - Piecewise[i, j - 1]) / (0.5 - p) + epsilon
    #             elif 1 - p < Piecewise[i, j - 1] < 1:
    #                 Piecewise[i, j] = (1 - Piecewise[i, j - 1]) / p + epsilon
    #
    #     return Piecewise

    def SO_initial(self):
        # 随机生成
        X = self.lb + np.random.random_sample((self.Pop_size, self.Len_Chromo*2)) * (self.ub - self.lb)
        # # 分段线性混沌映射(PWLCM)生成
        # X = self.pwlc_map(self.Pop_size, self.Len_Chromo*2)
        return X

    # 适应度
    def fitness(self, e, CHS, J, Processing_time, M_num, Len):
        # 种群映射转换
        CHS = e.Coding_mapping_conversion(CHS)
        Fit = []
        for i in range(len(CHS)):
            d = Decode(J, Processing_time, M_num)
            y, Matching_result_all, tn = d.decode(CHS[i], Len)
            Fit.append(y)
        return Fit

    # 标准蛇优化算法的ExplorationPhaseNoFood
    def ExplorationPhaseNoFood(self, male_number, male, male_individual_fitness, new_male, female_number, female, female_individual_fitness, new_female):
        # 先是雄性
        for i in range(male_number):
            for j in range(self.Len_Chromo*2):
                # 先取得一个随机的个体
                rand_leader_index = np.random.randint(0, male_number)
                rand_male = male[rand_leader_index, :]
                # 随机生成+或者是-,来判断当前的c2是取正还是负
                negative_or_positive = np.random.randint(0, 2)
                flag = self.vec_flag[negative_or_positive]
                # 计算Am,np.spacing(1)是为了防止进行除法运算的时候出现除0操作
                am = math.exp(
                    -(male_individual_fitness[rand_leader_index] / (male_individual_fitness[i] + np.spacing(1))))
                new_male[i, j] = rand_male[j] + flag * self.C2 * am * (
                        (self.ub - self.lb) * random.random() + self.lb)
        for i in range(female_number):
            for j in range(self.Len_Chromo*2):
                # 先取得一个随机的个体
                rand_leader_index = np.random.randint(0, female_number)
                rand_female = female[rand_leader_index, :]
                # 随机生成+或者是-,来判断当前的c2是取正还是负
                negative_or_positive = np.random.randint(0, 2)
                flag = self.vec_flag[negative_or_positive]
                # 计算Am,np.spacing(1)是为了防止进行除法运算的时候出现除0操作
                am = math.exp(-(female_individual_fitness[rand_leader_index] / (
                        female_individual_fitness[i] + np.spacing(1))))
                new_female[i, j] = rand_female[j] + flag * self.C2 * am * (
                        (self.ub - self.lb) * random.random() + self.lb)
        return new_male, new_female

    # #算法改进：将将勘探阶段的位置更新公式替换为WOA螺旋。 TODO
    # def ExplorationPhaseNoFood(self, food, male_number, male, male_individual_fitness, new_male, female_number, female,
    #                            female_individual_fitness, new_female):
    #     # 对雄性进行处理
    #     for i in range(male_number):
    #         b = 1
    #         l = 2 * random.random() - 1  # [-1,1]之间的随机数
    #         temp = np.zeros_like(male[i])
    #         for j in range(self.Len_Chromo * 2):
    #             distance2Leader = np.abs(food[j] - male[i, j])
    #             temp[j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + food[j]
    #             # 更新雄性的位置
    #         new_male[i, :] = temp
    #
    #     # 对雌性进行处理
    #     for i in range(female_number):
    #         b = 1
    #         l = 2 * random.random() - 1  # [-1,1]之间的随机数
    #         temp = np.zeros_like(female[i])
    #         for j in range(self.Len_Chromo * 2):
    #             distance2Leader = np.abs(food[j] - female[i, j])
    #             temp[j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + food[j]
    #             # 更新雄性的位置
    #         new_female[i, :] = temp
    #
    #     return new_male, new_female

    def ExplorationPhaseFoodExists(self, food, temp, male_number, male, new_male, female_number, female, new_female):
        # 更新雄性的位置
        for i in range(male_number):
            # 随机生成+或者是-,来判断当前的c2是取正还是负
            negative_or_positive = np.random.randint(0, 2)
            flag = self.vec_flag[negative_or_positive]
            for j in range(self.Len_Chromo*2):
                new_male[i, j] = food[j] + flag * self.C3 * temp * random.random() * (food[j] - male[i, j])
        # 更新雌性的位置
        for i in range(female_number):
            # 随机生成+或者是-,来判断当前的c2是取正还是负
            negative_or_positive = np.random.randint(0, 2)
            flag = self.vec_flag[negative_or_positive]
            for j in range(self.Len_Chromo*2):
                new_female[i, j] = food[j] + flag * self.C3 * temp * random.random() * (food[j] - female[i, j])
        return new_male, new_female

    def fight(self, quantity, male, male_number, male_individual_fitness, male_fitness_best_value, male_best_fitness_individual, new_male, female, female_number, female_individual_fitness, female_fitness_best_value, female_best_fitness_individual, new_female):
        # 更新雄性的位置
        for i in range(male_number):
            for j in range(self.Len_Chromo*2):
                # 先计算当前雄性的战斗的能力
                fm = math.exp(-female_fitness_best_value / (male_individual_fitness[i] + np.spacing(1)))
                new_male[i, j] = male[i, j] + self.C3 * fm * random.random() * (
                        quantity * male_best_fitness_individual[j] - male[i, j])
        # 更新雌性的位置
        for i in range(female_number):
            for j in range(self.Len_Chromo*2):
                # 先计算当前雌性的战斗的能力
                ff = math.exp(-male_fitness_best_value / (female_individual_fitness[i] + np.spacing(1)))
                new_female[i, j] = female[i, j] + self.C3 * ff * random.random() * (
                        quantity * female_best_fitness_individual[j] - female[i, j])
        return new_male, new_female

    def mating(self, quantity, male, male_number, male_individual_fitness, new_male, female, female_number, female_individual_fitness, new_female):
        # 雄性先交配
        for i in range(male_number):
            for j in range(self.Len_Chromo*2):
                # 计算当前雄性的交配的能力
                mm = math.exp(-female_individual_fitness[i] / (male_individual_fitness[i] + np.spacing(1)))
                new_male[i, j] = male[i, j] + self.C3 * random.random() * mm * (
                        quantity * female[i, j] - male[i, j])
        # 雌性先交配
        for i in range(female_number):
            for j in range(self.Len_Chromo*2):
                # 计算当前雄性的交配的能力
                mf = math.exp(-male_individual_fitness[i] / (female_individual_fitness[i] + np.spacing(1)))
                new_female[i, j] = female[i, j] + self.C3 * random.random() * mf * (
                        quantity * male[i, j] - female[i, j])
        # 产蛋
        negative_or_positive = np.random.randint(0, 2)
        egg = self.vec_flag[negative_or_positive]
        if egg == 1:
            # 拿到当前雄性种群中适应度最大的个体
            male_worst_fitness_index = np.argmax(male_individual_fitness)
            # 未改进
            new_male[male_worst_fitness_index, :] = self.lb + random.random() * (self.ub - self.lb)
            # # 分段混沌映射改进
            # new_male[male_worst_fitness_index, :] = self.pwlc_map(1, self.Len_Chromo*2)[0]
            # 拿到当前雌性种群中适应度最大的
            female_worst_fitness_index = np.argmax(female_individual_fitness)
            # 未改进
            new_female[female_worst_fitness_index, :] = self.lb + random.random() * (self.ub - self.lb)
            # # 分段混沌映射改进
            # new_female[female_worst_fitness_index, :] = self.pwlc_map(1, self.Len_Chromo*2)[0]
        return new_male, new_female

    def update(self, t, gy_best, Len, e, food, male, male_number, male_individual_fitness, male_fitness_best_value, new_male, male_best_fitness_individual, female, female_number, female_individual_fitness, female_fitness_best_value, new_female, female_best_fitness_individual):
        # 处理雄性
        for j in range(male_number):
            # 如果当前更新后的值是否在规定的范围内
            flag_low = new_male[j, :] < self.lb
            flag_high = new_male[j, :] > self.ub
            new_male[j, :] = (np.multiply(new_male[j, :], ~(flag_low + flag_high))) + np.multiply(self.ub-0.0000001,flag_high) + np.multiply(self.lb, flag_low)
            # 计算雄性种群中每一个个体的适应度（这个是被更新过位置的）
            individual = np.array(new_male[j, :])[0]
            mapped_individual = e.Individual_Coding_mapping_conversion(individual)
            d = Decode(J, Processing_time, M_num)
            y, Matching_result_all,tn = d.decode(mapped_individual, Len)


            # # LOBL strategy
            # k = (1 + (t / self.Max_Itertions) ** 0.5) ** 10
            # new_individual = (self.ub + self.lb) / 2 + (self.ub + self.lb) / (2 * k) - individual / k
            #
            # flag_low = new_individual < self.lb
            # flag_high = new_individual > self.ub
            # new_individual = (np.multiply(new_individual, ~(flag_low + flag_high))) + np.multiply(self.ub-0.0000001,flag_high) + np.multiply(self.lb, flag_low)
            # new_mapped_individual = e.Individual_Coding_mapping_conversion(new_individual)
            # d = Decode(J, Processing_time, M_num, self.k)
            # y_new, Matching_result_all = d.decode(new_mapped_individual, Len)
            #
            # if y_new < y:
            #     new_male[j, :] = new_individual
            #     y = y_new


            # 判断是否需要更改当前个体的历史最佳适应度
            if y < male_individual_fitness[j]:
                # 更新适应度
                male_individual_fitness[j] = y
                # 更新原有种群中个体的位置到新位置
                male[j, :] = new_male[j, :]
        # 得到雄性个体中的最佳适应度
        # 拿到索引
        male_current_best_fitness_index = np.argmin(male_individual_fitness)
        # 拿到值
        male_current_best_fitness = male_individual_fitness[male_current_best_fitness_index]

        # 处理雌性
        for j in range(female_number):
            # 如果当前更新后的值是否在规定的范围内
            flag_low = new_female[j, :] < self.lb
            flag_high = new_female[j, :] > self.ub
            new_female[j, :] = (np.multiply(new_female[j, :], ~(flag_low + flag_high))) + np.multiply(self.ub-0.0000001, flag_high) + np.multiply(self.lb, flag_low)
            # 计算雄性种群中每一个个体的适应度（这个是被更新过位置的）
            individual = np.array(new_female[j, :])[0]
            mapped_individual = e.Individual_Coding_mapping_conversion(individual)
            d = Decode(J, Processing_time, M_num)
            y, Matching_result_all, tn = d.decode(mapped_individual, Len)

            # # LOBL strategy
            # k = (1 + (t / self.Max_Itertions) ** 0.5) ** 10
            # new_individual = (self.ub + self.lb) / 2 + (self.ub + self.lb) / (2 * k) - individual / k
            #
            # flag_low = new_individual < self.lb
            # flag_high = new_individual > self.ub
            # new_individual = (np.multiply(new_individual, ~(flag_low + flag_high))) + np.multiply(self.ub-0.0000001,flag_high) + np.multiply(self.lb, flag_low)
            # new_mapped_individual = e.Individual_Coding_mapping_conversion(new_individual)
            # d = Decode(J, Processing_time, M_num, self.k)
            # y_new, Matching_result_all = d.decode(new_mapped_individual, Len)
            #
            # if y_new < y:
            #     new_male[j, :] = new_individual
            #     y = y_new


            # 判断是否需要更改当前个体的历史最佳适应度
            if y < female_individual_fitness[j]:
                # 更新适应度
                female_individual_fitness[j] = y
                # 更新原有种群中个体的位置到新位置
                female[j, :] = new_female[j, :]
        # 得到雄性个体中的最佳适应度
        # 拿到索引
        female_current_best_fitness_index = np.argmin(female_individual_fitness)
        # 拿到值
        female_current_best_fitness = female_individual_fitness[female_current_best_fitness_index]

        # 判断是否需要更新雄性种群的全局最佳适应度
        if male_current_best_fitness < male_fitness_best_value:
            # 更新解决方案
            male_best_fitness_individual = male[male_current_best_fitness_index, :]
            # 更新最佳适应度
            male_fitness_best_value = male_current_best_fitness
        # 判断是否需要更新雌性种群的全局最佳适应度
        if female_current_best_fitness < female_fitness_best_value:
            # 更新解决方案
            female_best_fitness_individual = female[female_current_best_fitness_index, :]
            # 更新最佳适应度
            female_fitness_best_value = female_current_best_fitness


        if male_fitness_best_value < female_fitness_best_value:
            gy_best = male_fitness_best_value
            # 更新食物的位置
            food = male_best_fitness_individual
        else:
            gy_best = female_fitness_best_value
            # 更新食物的位置
            food = female_best_fitness_individual

        return male_best_fitness_individual, female_best_fitness_individual, food, gy_best, male, male_individual_fitness, male_fitness_best_value, female, female_individual_fitness, female_fitness_best_value
