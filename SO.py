import itertools
import math
import random

import numpy as np

from Decode import Decode
from Instance_2 import *


class SO():
    def __init__(self, Len_Chromo):
        self.Pop_size = 100  # 种群数量

        self.C1 = 0.5     #
        self.C2 = 0.5  #
        self.C3 = 2  #

        self.food_threshold = 0.25        # 有没有食物的阈值
        self.temp_threshold = 0.6         # 温度适不适合交配的阈值
        self.model_threshold = 0.6        # 模式阈值,当产生的随机值小于模式阈值就进入战斗模式，否则就进入交配模式

        self.Max_Itertions = 100  # 最大迭代次数
        self.Len_Chromo = Len_Chromo

        self.vec_flag = [1, -1]
        self.ub = 1
        self.lb = 0

    def SO_initial(self):
        X = self.lb + np.random.random_sample((self.Pop_size, self.Len_Chromo*2)) * (self.ub - self.lb)
        return X

    # 适应度
    def fitness(self, e, CHS, J, Processing_time, M_num, Len):
        # 种群映射转换
        CHS = e.Coding_mapping_conversion(CHS)
        Fit = []
        for i in range(len(CHS)):
            d = Decode(J, Processing_time, M_num)
            Fit.append(d.decode(CHS[i], Len))
        return Fit

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
                new_male[i, j] = rand_male[0, j] + flag * self.C2 * am * (
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
                new_female[i, j] = rand_female[0, j] + flag * self.C2 * am * (
                        (self.ub - self.lb) * random.random() + self.lb)
        return new_male, new_female


    def ExplorationPhaseFoodExists(self, food, temp, male_number, male, new_male, female_number, female, new_female):
        # 更新雄性的位置
        for i in range(male_number):
            # 随机生成+或者是-,来判断当前的c2是取正还是负
            negative_or_positive = np.random.randint(0, 2)
            flag = self.vec_flag[negative_or_positive]
            for j in range(self.Len_Chromo*2):
                new_male[i, j] = food[0, j] + flag * self.C3 * temp * random.random() * (food[0, j] - male[i, j])
        # 更新雌性的位置
        for i in range(female_number):
            # 随机生成+或者是-,来判断当前的c2是取正还是负
            negative_or_positive = np.random.randint(0, 2)
            flag = self.vec_flag[negative_or_positive]
            for j in range(self.Len_Chromo*2):
                new_female[i, j] = food[0, j] + flag * self.C3 * temp * random.random() * (food[0, j] - female[i, j])
        return new_male, new_female

    def fight(self, quantity, male, male_number, male_individual_fitness, male_fitness_best_value, male_best_fitness_individual, new_male, female, female_number, female_individual_fitness, female_fitness_best_value, female_best_fitness_individual, new_female):
        # 更新雄性的位置
        for i in range(male_number):
            for j in range(self.Len_Chromo*2):
                # 先计算当前雄性的战斗的能力
                fm = math.exp(-female_fitness_best_value / (male_individual_fitness[i] + np.spacing(1)))
                new_male[i, j] = male[i, j] + self.C3 * fm * random.random() * (
                        quantity * male_best_fitness_individual[0, j] - male[i, j])
        # 更新雌性的位置
        for i in range(female_number):
            for j in range(self.Len_Chromo*2):
                # 先计算当前雌性的战斗的能力
                ff = math.exp(-male_fitness_best_value / (female_individual_fitness[i] + np.spacing(1)))
                new_female[i, j] = female[i, j] + self.C3 * ff * random.random() * (
                        quantity * female_best_fitness_individual[0, j] - female[i, j])
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
            new_male[male_worst_fitness_index, :] = self.lb + random.random() * (
                    self.ub - self.lb)
            # 拿到当前雌性种群中适应度最大的
            female_worst_fitness_index = np.argmax(female_individual_fitness)
            new_female[female_worst_fitness_index, :] = self.lb + random.random() * (
                    self.ub - self.lb)
        return new_male, new_female

    def update(self, gy_best, Best_fit, Len, e, food, male, male_number, male_individual_fitness, male_fitness_best_value, new_male, female, female_number, female_individual_fitness, female_fitness_best_value, new_female):
        # 处理雄性
        for j in range(male_number):
            # 如果当前更新后的值是否在规定的范围内
            flag_low = new_male[j, :] < self.lb
            flag_high = new_male[j, :] > self.ub
            new_male[j, :] = (np.multiply(new_male[j, :], ~(flag_low + flag_high))) + np.multiply(self.ub,flag_high) + np.multiply(self.lb, flag_low)
            # 计算雄性种群中每一个个体的适应度（这个是被更新过位置的）
            mapped_individual = e.Individual_Coding_mapping_conversion(new_male[j, :])
            d = Decode(J, Processing_time, M_num)
            y = d.decode(mapped_individual, Len)
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
            new_female[j, :] = (np.multiply(new_female[j, :], ~(flag_low + flag_high))) + np.multiply(self.ub, flag_high) + np.multiply(self.lb, flag_low)
            # 计算雄性种群中每一个个体的适应度（这个是被更新过位置的）
            mapped_individual = e.Individual_Coding_mapping_conversion(new_female[j, :])
            d = Decode(J, Processing_time, M_num)
            y = d.decode(mapped_individual, Len)
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
        female_current_best_fitness = male_individual_fitness[female_current_best_fitness_index]

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

        # if male_current_best_fitness < female_current_best_fitness:
        #     # Best_fit[t] = male_current_best_fitness
        #     Best_fit.append(male_current_best_fitness)
        # else:
        #     # Best_fit[t] = female_current_best_fitness
        #     Best_fit.append(female_current_best_fitness)


        # 更新全局最佳适应度（这里是非常的奇怪的，不进行判断就直接更新了，他就能确定本代的最佳一定是比上一代好！！！）
        if male_fitness_best_value < female_fitness_best_value:
            gy_best = male_fitness_best_value
            # 更新食物的位置
            food = male_best_fitness_individual
        else:
            gy_best = female_fitness_best_value
            # 更新食物的位置
            food = female_best_fitness_individual

        return food, gy_best, Best_fit, male, male_individual_fitness, male_fitness_best_value, female, female_individual_fitness, female_fitness_best_value














