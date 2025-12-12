import numpy as np
import matplotlib.pyplot as plt


def Friedman(n, k, data_matrix):
    '''
    Friedman 检验
    :param n:数据集个数
    :param k: 算法种数
    :param data_matrix:排序矩阵
    :return:T1
    '''

    # 计算每个算法的平均序值
    row, col = data_matrix.shape  # 获取矩阵的行和列
    xuzhi_mean = list()
    for i in range(col):  # 计算平均序值
        xuzhi_mean.append(data_matrix[:, i].mean())  # xuzhi_mean = [1.0, 2.125, 2.875] list列表形式
    sum_mean = np.array(xuzhi_mean)  # 转成 numpy.ndarray 格式方便运算

    sum_ri2_mean = (sum_mean ** 2).sum()  # 整个矩阵内的元素逐个平方后，得到的值相加起来
    result_Tx2 = (12 * n) * (sum_ri2_mean - ((k * ((k + 1) ** 2)) / 4)) / (k * (k + 1))  # P42页的公式
    result_Tf = (n - 1) * result_Tx2 / (n * (k - 1) - result_Tx2)  # P42页的公式
    return xuzhi_mean, result_Tx2, result_Tf


def nemenyi(n, k, q):
    '''
    Nemenyi 后续检验
    :param n:数据集个数
    :param k:算法种数
    :param q:直接查书上2.7的表
    :return:
    '''
    cd = q * (np.sqrt((k * (k + 1) / (6 * n))))
    return cd


data = np.array([[1, 2.5, 4.5, 4.5, 2.5],
                 [1, 2.5, 4.5, 4.5, 2.5],
                 [1, 2.5, 4.5, 4.5, 2.5],
                 [1, 2, 4.5, 4.5, 3],
                 [1, 2, 5, 3.5, 3.5],
                 [1.5, 3, 4.5, 4.5, 1.5],
                 [1, 4, 4, 4, 2],
                 [1.5, 3, 4.5, 4.5, 1.5],
                 [1, 2.5, 4.5, 4.5, 2.5],
                 [1, 2.5, 4.5, 4.5, 2.5],
                 [1.5, 3.5, 5, 3.5, 1.5],
                 [1, 3, 4.5, 4.5, 2],
                 [1.5, 4, 4, 4, 1.5],
                 [1, 3, 4.5, 4.5, 2],
                 [1.5, 4, 4, 4, 1.5]])

xuzhi_mean, result_Tx2, result_Tf = Friedman(15, 5, data)
cd = nemenyi(15, 5, 2.728)
print('xuzhi_mean={}'.format(xuzhi_mean))
print('tx2={}'.format(result_Tx2))
print('tf={}'.format(result_Tf))
print('CD={}'.format(cd))

# 画出CD图
row, col = data.shape  # 获取矩阵的行和列
xuzhi_mean = list()
for i in range(col):  # 计算平均序值
    xuzhi_mean.append(data[:, i].mean())  # xuzhi_mean = [1.0, 2.125, 2.875] list列表形式
sum_mean = np.array(xuzhi_mean)
# 这一句可以表示上面sum_mean： rank_x = list(map(lambda x: np.mean(x), data.T))  # 均值 [1.0, 2.125, 2.875]
# name_y = ["ISOA", "SOA", "MWOA"]
name_y = ["ISOA", "SOA", "MWOA", "JCGA", "IABC"]

# 散点左右的位置
min_ = sum_mean - cd / 2
max_ = sum_mean + cd / 2
# 因为想要从高出开始画，所以数组反转一下
name_y.reverse()
sum_mean = list(sum_mean)
sum_mean.reverse()
max_ = list(max_)
max_.reverse()
min_ = list(min_)
min_.reverse()
# 开始画图
# plt.title("Friedman")
plt.xlabel("Rank")  # 设置横轴标题为Rank
plt.scatter(sum_mean, name_y, color='black')  # 绘制散点图

# 为每条线段指定不同颜色
colors = ['red', 'blue', 'green', 'orange', 'purple']  # 定义颜色列表
colors.reverse()
for i, (y, xmin, xmax, color) in enumerate(zip(name_y, min_, max_, colors)):
    plt.hlines(y, xmin, xmax, colors=color)

# 计算并设置y轴范围，使坐标更紧凑
y_positions = list(range(len(name_y)))  # y轴位置
plt.ylim(min(y_positions)-1, max(y_positions)+1)  # 紧凑显示y轴

plt.tight_layout()  # 自动调整图像布局，使显示更紧凑
plt.savefig("Friedman_test.png", dpi=300)  # 高分辨率保存
plt.show()
