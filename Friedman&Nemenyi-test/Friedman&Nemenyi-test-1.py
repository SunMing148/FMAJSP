import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from itertools import combinations

# 读取 Excel 文件
file_path = '2.xlsx'
df = pd.read_excel(file_path, sheet_name='sheet1', header=None)

# 提取前五行作为 a1~a5
a1 = df.iloc[0].dropna().to_numpy()
a2 = df.iloc[1].dropna().to_numpy()
a3 = df.iloc[2].dropna().to_numpy()
a4 = df.iloc[3].dropna().to_numpy()
a5 = df.iloc[4].dropna().to_numpy()

algorithms = [a1, a2, a3, a4, a5]
names = ['ISOA', 'SOA', 'MWOA', 'JCGA', 'IABC']

# 正态性检验和统计分析
for i, data in enumerate(algorithms):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    shapiro_stat, shapiro_p = stats.shapiro(data)
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(mean, std))

    print(f"{names[i]}")
    print(f"  Mean: {mean:.3f}")
    print(f"  Std Dev: {std:.3f}")
    print(f"  Skewness: {skewness:.3f}")
    print(f"  Kurtosis: {kurtosis:.3f}")
    print(f"  Shapiro-Wilk Test: stat={shapiro_stat:.3f}, p={shapiro_p:.4f}")
    print(f"  Kolmogorov-Smirnov Test: stat={ks_stat:.3f}, p={ks_p:.4f}")
    print("-" * 60)


# 绘制每个算法单独的直方图 + 正态曲线，并保存为 PNG 文件
for i, data in enumerate(algorithms):
    plt.figure(figsize=(6, 4))  # 单图尺寸

    sns.histplot(data, kde=False, stat='density', bins=20, color='skyblue', edgecolor='black')

    # 正态分布拟合曲线
    mu, sigma = np.mean(data), np.std(data, ddof=1)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r--')

    # 图标题和标签
    plt.title(names[i], fontsize=14)
    plt.xlabel("Best Fitness")
    plt.ylabel("Density")
    # plt.legend(['Normal PDF'])

    # 布局与保存
    plt.tight_layout()
    filename = f"{names[i]}.png"
    plt.savefig(filename, dpi=300)  # 高分辨率保存
    plt.close()  # 关闭图形窗口，避免内存占用


# 将五个算法数据堆成 (120, 5) 的二维数组：每行是一次实验的五个算法结果
data_matrix = np.vstack(algorithms).T  # shape: (120, 5)

# Friedman 检验
friedman_stat, friedman_p = friedmanchisquare(*data_matrix.T)

# 中位数与标准差
medians = [np.median(a) for a in algorithms]
stds = [np.std(a, ddof=1) for a in algorithms]

# 计算效应量：Cohen's f（对应 Friedman 检验的一种估计方式）
# 方法来源：Kendall's W = chi2 / (n * (k - 1))
# 然后 f = sqrt(W / (1 - W)) ；注意 W 需 <1 才可算
n, k = data_matrix.shape  # n: 重复次数，k: 组数
kendalls_w = friedman_stat / (n * (k - 1))

if 0 <= kendalls_w < 1:
    cohens_f = np.sqrt(kendalls_w / (1 - kendalls_w))
else:
    cohens_f = np.nan  # W超界时无法计算

# 输出结果
print("\n========== Friedman 检验结果 ==========")
for i, name in enumerate(names):
    print(f"{name}: Median = {medians[i]:.3f}, Std = {stds[i]:.3f}")

print(f"\nFriedman Test Statistic = {friedman_stat:.4f}")
print(f"P-value = {friedman_p:.6f}")
print(f"Kendall's W = {kendalls_w:.4f}")
print(f"Cohen's f = {cohens_f:.4f}" if not np.isnan(cohens_f) else "Cohen's f 无法计算（W ≥ 1）")


# # Nemenyi
#
# # 假设已有5个一维数组 a1, a2, a3, a4, a5
# # 构建行为样本，列为算法的矩阵
# data_matrix = np.array([a1, a2, a3, a4, a5]).T  # shape: (120, 5)
#
# # Nemenyi 多重比较检验
# nemenyi_result = sp.posthoc_nemenyi_friedman(data_matrix)
#
# # 输出检验结果矩阵（P值）
# print("Nemenyi 多重比较 P 值矩阵：")
# print(nemenyi_result)
#
# # 构建长格式数据 DataFrame
# df_long = pd.DataFrame({
#     'Value': np.concatenate([a1, a2, a3, a4, a5]),
#     'Algorithm': np.repeat(['ISOA', 'SOA', 'MWOA', 'JCGA', 'IABC'], len(a1))
# })
#
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Algorithm', y='Value', data=df_long, palette='Set2', width=0.6, showmeans=True,
#             meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 6})
#
# # 添加显著性星号（Wilcoxon 成对比较）
# pairs = list(combinations(df_long['Algorithm'].unique(), 2))
# y_offset = 5
# y_max = df_long['Value'].max()
# y = y_max + y_offset
#
# for (alg1, alg2) in pairs:
#     d1 = df_long[df_long['Algorithm'] == alg1]['Value']
#     d2 = df_long[df_long['Algorithm'] == alg2]['Value']
#     stat, p = stats.wilcoxon(d1, d2)
#     if p < 0.05:
#         x1, x2 = df_long['Algorithm'].unique().tolist().index(alg1), df_long['Algorithm'].unique().tolist().index(alg2)
#         plt.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], c='k')
#         plt.text((x1 + x2)/2, y + 1.5, "*", ha='center', fontsize=12)
#         y += y_offset
#
# plt.title("Algorithm Comparison (Wilcoxon Significance)")
# plt.ylabel("Best Fitness")
# plt.tight_layout()
# plt.savefig("algorithm_comparison.png", dpi=300)
# plt.show()