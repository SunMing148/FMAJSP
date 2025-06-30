import numpy as np
import matplotlib.pyplot as plt
import math


# ====================== 基准测试函数 ======================
def F1(x):
    return np.sum(x ** 2)


def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def F3(x):
    dim = len(x)
    o = 0.0
    for i in range(dim):
        o += np.sum(x[:i + 1]) ** 2
    return o


def F4(x):
    return np.max(np.abs(x))


def F5(x):
    dim = len(x)
    o = np.sum(100 * (x[1:dim] - x[:dim - 1] ** 2) ** 2 + (x[:dim - 1] - 1) ** 2)
    return o


def F6(x):
    return np.sum((np.abs(x) + 0.5) ** 2)


def F7(x):
    dim = len(x)
    weights = np.arange(1, dim + 1)
    return np.sum(weights * (x ** 4)) + np.random.rand()


def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    dim = len(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim


def F10(x):
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim)) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)


def F11(x):
    dim = len(x)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1


def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))


def F12(x):
    dim = len(x)
    part1 = 10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4))) ** 2
    part2 = np.sum((((x[:dim - 1] + 1) / 4) ** 2) *
                   (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4))) ** 2))
    part3 = ((x[dim - 1] + 1) / 4) ** 2
    return (np.pi / dim) * (part1 + part2 + part3) + np.sum(Ufun(x, 10, 100, 4))


def F13(x):
    dim = len(x)
    part1 = (np.sin(3 * np.pi * x[0])) ** 2
    part2 = np.sum((x[:dim - 1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1:])) ** 2))
    part3 = ((x[dim - 1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[dim - 1])) ** 2)
    return 0.1 * (part1 + part2 + part3) + np.sum(Ufun(x, 5, 100, 4))




def F16(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + x[0] ** 6 / 3 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


def F17(x):
    return (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + (5 / np.pi) * x[0] - 6) ** 2 + 10 * (
                1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10


def F18(x):
    part1 = 1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
    part2 = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (
                18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
    return part1 * part2


def F19(x):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array(
        [[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    o = 0.0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i] * (x - pH[i]) ** 2))
    return o


def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    o = 0.0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i] * (x - pH[i]) ** 2))
    return o


def F21(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0.0
    for i in range(5):
        o -= ((x - aSH[i]).dot(x - aSH[i]) + cSH[i]) ** (-1)
    return o


def F22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0.0
    for i in range(7):
        o -= ((x - aSH[i]).dot(x - aSH[i]) + cSH[i]) ** (-1)
    return o


def F23(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0.0
    for i in range(10):
        o -= ((x - aSH[i]).dot(x - aSH[i]) + cSH[i]) ** (-1)
    return o


def Get_Functions_details(F):
    func_dict = {
        'F1': {'fobj': F1, 'lb': -100, 'ub': 100, 'dim': 30},
        'F2': {'fobj': F2, 'lb': -10, 'ub': 10, 'dim': 30},
        'F3': {'fobj': F3, 'lb': -100, 'ub': 100, 'dim': 30},
        'F4': {'fobj': F4, 'lb': -100, 'ub': 100, 'dim': 30},
        'F5': {'fobj': F5, 'lb': -30, 'ub': 30, 'dim': 30},
        'F6': {'fobj': F6, 'lb': -100, 'ub': 100, 'dim': 30},
        'F7': {'fobj': F7, 'lb': -1.28, 'ub': 1.28, 'dim': 30},
        'F8': {'fobj': F8, 'lb': -500, 'ub': 500, 'dim': 30},
        'F9': {'fobj': F9, 'lb': -5.12, 'ub': 5.12, 'dim': 30},
        'F10': {'fobj': F10, 'lb': -32, 'ub': 32, 'dim': 30},
        'F11': {'fobj': F11, 'lb': -600, 'ub': 600, 'dim': 30},
        'F12': {'fobj': F12, 'lb': -50, 'ub': 50, 'dim': 30},
        'F13': {'fobj': F13, 'lb': -50, 'ub': 50, 'dim': 30},
        # 'F14': {'fobj': F14, 'lb': -65.536, 'ub': 65.536, 'dim': 2},
        # 'F15': {'fobj': F15, 'lb': -5, 'ub': 5, 'dim': 4},
        'F16': {'fobj': F16, 'lb': -5, 'ub': 5, 'dim': 2},
        'F17': {'fobj': F17, 'lb': np.array([-5, 0]), 'ub': np.array([10, 15]), 'dim': 2},
        'F18': {'fobj': F18, 'lb': -2, 'ub': 2, 'dim': 2},
        'F19': {'fobj': F19, 'lb': 0, 'ub': 1, 'dim': 3},
        'F20': {'fobj': F20, 'lb': 0, 'ub': 1, 'dim': 6},
        'F21': {'fobj': F21, 'lb': 0, 'ub': 10, 'dim': 4},
        'F22': {'fobj': F22, 'lb': 0, 'ub': 10, 'dim': 4},
        'F23': {'fobj': F23, 'lb': 0, 'ub': 10, 'dim': 4}
    }
    details = func_dict[F]
    return details['lb'], details['ub'], details['dim'], details['fobj']


# ====================== 初始化函数 ======================
def initialization(SearchAgents_no, dim, ub, lb):
    if isinstance(ub, (int, float)):
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i] if isinstance(ub, np.ndarray) else ub
            lb_i = lb[i] if isinstance(lb, np.ndarray) else lb
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i
    return Positions


# ====================== WOA 主算法 ======================
def WOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    # 初始化领导者的位置和分数
    Leader_pos = np.zeros(dim)
    Leader_score = float('inf')  # 最小化问题

    # 初始化搜索代理的位置
    Positions = initialization(SearchAgents_no, dim, ub, lb)

    Convergence_curve = np.zeros(Max_iter)

    # 主循环
    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            # 确保搜索代理在搜索空间内
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)

            # 计算适应度
            fitness = fobj(Positions[i, :])

            # 更新领导者
            if fitness < Leader_score:
                Leader_score = fitness
                Leader_pos = Positions[i, :].copy()

        # 更新 a
        a = 2 - t * (2 / Max_iter)  # 线性从2降到0
        a2 = -1 + t * (-1 / Max_iter)  # 线性从-1降到-2

        # 更新搜索代理的位置
        for i in range(SearchAgents_no):
            r1 = np.random.rand()
            r2 = np.random.rand()

            A = 2 * a * r1 - a  # 计算A
            C = 2 * r2  # 计算C

            b = 1  # 定义b
            l = (a2 - 1) * np.random.rand() + 1  # 计算l

            p = np.random.rand()

            for j in range(dim):
                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = np.random.randint(0, SearchAgents_no)
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand
                    else:
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader
                else:
                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    Positions[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + Leader_pos[j]

        Convergence_curve[t] = Leader_score
        print(f'Iteration {t + 1}: Best Score = {Leader_score}')

    return Leader_score, Leader_pos, Convergence_curve


# ====================== 绘图函数 ======================
def func_plot(func_name):
    lb, ub, dim, fobj = Get_Functions_details(func_name)

    # 根据函数名设置绘图范围
    ranges = {
        'F1': (-100, 100, 2), 'F2': (-10, 10, 2), 'F3': (-100, 100, 2),
        'F4': (-100, 100, 2), 'F5': (-200, 200, 2), 'F6': (-100, 100, 2),
        'F7': (-1, 1, 0.03), 'F8': (-500, 500, 10), 'F9': (-5, 5, 0.1),
        'F10': (-20, 20, 0.5), 'F11': (-500, 500, 10), 'F12': (-10, 10, 0.1),
        'F13': (-5, 5, 0.08), 'F14': (-100, 100, 2), 'F15': (-5, 5, 0.1),
        'F16': (-1, 1, 0.01), 'F17': (-5, 5, 0.1), 'F18': (-5, 5, 0.06),
        'F19': (-5, 5, 0.1), 'F20': (-5, 5, 0.1), 'F21': (-5, 5, 0.1),
        'F22': (-5, 5, 0.1), 'F23': (-5, 5, 0.1)
    }

    start, end, step = ranges[func_name]
    x = np.arange(start, end + step, step)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # 计算函数值
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if func_name in ['F15', 'F19', 'F20', 'F21', 'F22', 'F23']:
                if func_name == 'F15':
                    Z[i, j] = fobj(np.array([X[i, j], Y[i, j], 0, 0]))
                elif func_name == 'F19':
                    Z[i, j] = fobj(np.array([X[i, j], Y[i, j], 0]))
                elif func_name == 'F20':
                    Z[i, j] = fobj(np.array([X[i, j], Y[i, j], 0, 0, 0, 0]))
                else:  # F21, F22, F23
                    Z[i, j] = fobj(np.array([X[i, j], Y[i, j], 0, 0]))
            else:
                Z[i, j] = fobj(np.array([X[i, j], Y[i, j]]))

    # 绘制3D图形
    fig = plt.figure(figsize=(12, 5))

    # 参数空间
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax1.set_title('Parameter space')
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('x_2')
    ax1.set_zlabel(f'{func_name}(x_1, x_2)')

    # 等高线
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax2)
    ax2.set_title('Contour plot')
    ax2.set_xlabel('x_1')
    ax2.set_ylabel('x_2')

    plt.tight_layout()
    plt.show()


# ====================== 主函数 ======================
if __name__ == "__main__":
    SearchAgents_no = 30  # 搜索代理数量
    Function_name = 'F1'  # 测试函数名称
    Max_iteration = 500  # 最大迭代次数

    # 获取函数详情
    lb, ub, dim, fobj = Get_Functions_details(Function_name)

    # 运行WOA算法
    Best_score, Best_pos, Convergence_curve = WOA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)

    # 输出结果
    print(f'\nBest solution: {Best_pos}')
    print(f'Best score: {Best_score}')

    # 绘制函数空间
    if dim == 2 or Function_name in ['F15', 'F19', 'F20', 'F21', 'F22', 'F23']:
        try:
            func_plot(Function_name)
        except Exception as e:
            print(f"Could not plot function: {e}")

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(Convergence_curve, 'r', linewidth=2)
    plt.title('Convergence curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best score')
    plt.grid(True)
    plt.show()
