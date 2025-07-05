import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy

plt.rcParams['font.sans-serif'] = 'SimHei'  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'




'''F2函数'''
def F2(X):
    Results = np.sum(np.abs(X)) + np.prod(np.abs(X))
    return Results



# Funobject = {'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'F5': F5, 'F6': F6, 'F7': F7, 'F8': F8, 'F9': F9, 'F10': F10,
#              'F11': F11, 'F12': F12, 'F13': F13, 'F14': F14, 'F15': F15, 'F16': F16, 'F17': F17,
#              'F18': F18, 'F19': F19, 'F20': F20, 'F21': F21, 'F22': F22, 'F23': F23}
# Funobject.keys()
#
# # 维度，搜索区间下界，搜索区间上界，最优值
# Fundim = {'F1': [30, -100, 100], 'F2': [30, -10, 10], 'F3': [30, -100, 100], 'F4': [30, -10, 10], 'F5': [30, -30, 30],
#           'F6': [30, -100, 100], 'F7': [30, -1.28, 1.28], 'F8': [30, -500, 500], 'F9': [30, -5.12, 5.12],
#           'F10': [30, -32, 32],
#           'F11': [30, -600, 600], 'F12': [30, -50, 50], 'F13': [30, -50, 50], 'F14': [2, -65, 65], 'F15': [4, -5, 5],
#           'F16': [2, -5, 5],
#           'F17': [2, -5, 5], 'F18': [2, -2, 2], 'F19': [3, 0, 1], 'F20': [6, 0, 1], 'F21': [4, 0, 10],
#           'F22': [4, 0, 10], 'F23': [4, 0, 10]}

def initialization(pop, ub, lb, dim):
    ''' 种群初始化函数'''
    '''
    pop:为种群数量
    dim:每个个体的维度
    ub:每个维度的变量上边界，维度为[dim,1]
    lb:为每个维度的变量下边界，维度为[dim,1]
    X:为输出的种群，维度[pop,dim]
    '''
    X = np.zeros([pop, dim])  # 声明空间
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.random() + lb[j]  # 生成[lb,ub]之间的随机数

    return X


def BorderCheck(X, ub, lb, pop, dim):
    '''边界检查函数'''
    '''
    dim:为每个个体数据的维度大小
    X:为输入数据，维度为[pop,dim]
    ub:为个体数据上边界，维度为[dim,1]
    lb:为个体数据下边界，维度为[dim,1]
    pop:为种群数量
    '''
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


def CaculateFitness(X, fun):
    '''计算种群的所有个体的适应度值'''
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


def SortFitness(Fit):
    '''适应度排序'''
    '''
    输入为适应度值
    输出为排序后的适应度值，和索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    '''根据适应度对位置进行排序'''
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def RouletteWheelSelection(P):
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


def ABC(pop, dim, lb, ub, MaxIter, fun):
    '''人工蜂群算法'''
    '''
    输入：
    pop:为种群数量
    dim:每个个体的维度
    ub:为个体上边界信息，维度为[1,dim]
    lb:为个体下边界信息，维度为[1,dim]
    fun:为适应度函数接口
    MaxIter:为最大迭代次数
    输出：
    GbestScore:最优解对应的适应度值
    GbestPositon:最优解
    Curve:迭代曲线
    '''
    L = round(0.6 * dim * pop)  # limit 参数
    C = np.zeros([pop, 1])  # 计数器，用于与limit进行比较判定接下来的操作
    nOnlooker = pop  # 引领蜂数量

    X = initialization(pop, ub, lb, dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])  # 记录最优适应度值
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])  # 记录最优位置
    Curve = np.zeros([MaxIter, 1])
    Xnew = np.zeros([pop, dim])
    fitnessNew = copy.copy(fitness)
    for t in range(MaxIter):
        '''引领蜂搜索'''
        for i in range(pop):
            k = np.random.randint(pop)  # 随机选择一个个体
            while (k == i):  # 当k=i时，再次随机选择，直到k不等于i
                k = np.random.randint(pop)
            phi = (2 * np.random.random([1, dim]) - 1)
            Xnew[i, :] = X[i, :] + phi * (X[i, :] - X[k, :])  # 公式(2.2)位置更新
        Xnew = BorderCheck(Xnew, ub, lb, pop, dim)  # 边界检查
        fitnessNew = CaculateFitness(Xnew, fun)  # 计算适应度值
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:  # 如果适应度值更优，替换原始位置
                X[i, :] = copy.copy(Xnew[i, :])
                fitness[i] = copy.copy(fitnessNew[i])
            else:
                C[i] = C[i] + 1  # 如果位置没有更新，累加器+1

        # 计算选择适应度权重
        F = np.zeros([pop, 1])
        MeanCost = np.mean(fitness)
        for i in range(pop):
            F[i] = np.exp(-fitness[i] / MeanCost)
        P = F / sum(F)  # 式（2.4）
        '''侦察蜂搜索'''
        for m in range(nOnlooker):
            i = RouletteWheelSelection(P)  # 轮盘赌测量选择个体
            k = np.random.randint(pop)  # 随机选择个体
            while (k == i):
                k = np.random.randint(pop)
            phi = (2 * np.random.random([1, dim]) - 1)
            Xnew[i, :] = X[i, :] + phi * (X[i, :] - X[k, :])  # 位置更新
        Xnew = BorderCheck(Xnew, ub, lb, pop, dim)  # 边界检查
        fitnessNew = CaculateFitness(Xnew, fun)  # 计算适应度值
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:  # 如果适应度值更优，替换原始位置
                X[i, :] = copy.copy(Xnew[i, :])
                fitness[i] = copy.copy(fitnessNew[i])
            else:
                C[i] = C[i] + 1  # 如果位置没有更新，累加器+1
        '''判断limit条件，并进行更新'''
        for i in range(pop):
            if C[i] >= L:
                for j in range(dim):
                    X[i, j] = np.random.random() * (ub[j] - lb[j]) + lb[j]
                    C[i] = 0

        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve



# 设置参数
pop = 30  # 种群数量
MaxIter = 500  # 最大迭代次数
dim = 30  # 维度
lb = 0 * np.ones([dim, 1])  # 下边界
ub = 1 * np.ones([dim, 1])  # 上边界
# 选择适应度函数
fobj = F2
# 原始算法
GbestScore, GbestPositon, Curve = ABC(pop, dim, lb, ub, MaxIter, fobj)
# 改进算法

print('------原始算法结果--------------')
print('最优适应度值：', GbestScore)
print('最优解：', GbestPositon)



#绘制适应度曲线
plt.figure(figsize=(6,2.7),dpi=128)
plt.semilogy(Curve,'b-',linewidth=2)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('ABC',fontsize='large')
plt.legend(['ABC'], loc='upper right')
plt.show()
