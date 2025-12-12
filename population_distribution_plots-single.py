import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def load_population_history(filename: str):
    """
    从文件加载种群历史数据
    """
    data = np.load(filename, allow_pickle=True)
    return data['populations']


def visualize_multiple_algorithms(algorithm_data, output_dir):
    """
    对多个优化算法的种群迭代历史进行PCA降维并在同一张图中可视化

    参数:
    algorithm_data (dict): 包含算法名称和对应种群历史的字典
    output_dir (str): 图像保存目录
    """
    # 创建保存图像的目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有算法的迭代次数（假设所有算法迭代次数相同）
    first_algorithm = next(iter(algorithm_data.values()))
    n_iterations = len(first_algorithm)

    # 定义不同算法的样式 - 颜色和标记
    styles = {
        'MISOA': {'color': 'blue', 'marker': 'o', 'label': 'MISOA'},
        'SOA': {'color': 'red', 'marker': 'x', 'label': 'SOA'},
        'SOA-PWLCM': {'color': 'green', 'marker': '^', 'label': 'SOA-PWLCM'}
    }

    plt.figure(figsize=(10, 8))

    # i = 75  # 第一章初稿对应：0/50/75
    i = 200  # 第一章二稿对应：0/100/200

    # 为每个算法处理并绘制当前迭代数据
    for algo_name, population_history in algorithm_data.items():
        # 获取当前迭代的种群
        # population = population_history[i]

        if algo_name == 'SOA':
            population = population_history[105]
        elif algo_name == 'MISOA':
            population = population_history[3]
        elif algo_name == 'SOA-PWLCM':
            population = population_history[105]

        # 创建PCA模型，降维到二维（每个迭代单独做PCA以保证可比性）
        pca = PCA(n_components=2, random_state=42)  # 固定随机种子
        population_2d = pca.fit_transform(population)

        # 绘制散点图，使用预定义的样式
        style = styles[algo_name]
        plt.scatter(
            population_2d[:, 0],
            population_2d[:, 1],
            c=style['color'],
            marker=style['marker'],
            alpha=0.6,
            s=30,
            label=style['label']
        )

    # 设置图表标题和坐标轴标签，增加字体大小
    plt.title(f'L(14,7,4,1)×P(7,9,5)×210 - Iteration {i}', fontsize=20)  # 标题变大
    plt.xlabel('Principal Component 1', fontsize=18)  # 横轴标签变大
    plt.ylabel('Principal Component 2', fontsize=18)  # 纵轴标签变大

    # 添加图例（增大字体）和网格线
    plt.legend(fontsize=16)  # 图例变大
    plt.grid(True, linestyle='--', alpha=0.7)

    # # 可以考虑调整刻度标签大小
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    # 保存图像
    plt.savefig(os.path.join(output_dir, f'L(14,7,4,1)×P(7,9,5)×210_iteration_{i}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存迭代 {i} 的对比图: {os.path.join(output_dir, f'iteration_{i}.png')}")


# 使用示例
if __name__ == "__main__":
    # 定义三个算法的数据路径

    # Instance = "Instance_L(14,7,4)_P(1,1,1)"
    # Instance = "Instance_L(14,7,4)_P(2,2,2)"
    # Instance = "Instance_L(14,7,4)_P(3,4,2)"
    # Instance = "Instance_L(14,7,4)_P(5,7,3)"
    Instance = "Instance_L(14,7,4)_P(7,9,5)"

    algorithm_files = {
        'SOA': 'population_history/SOA_' + Instance + '_run_1_population_history.npz',
        'MISOA': 'population_history/ISOA_' + Instance + '_run_1_population_history.npz',
        'SOA-PWLCM': 'population_history/ISOA-1_' + Instance + '_run_1_population_history.npz'
    }

    # 加载所有算法的数据
    algorithm_data = {}
    for name, file_path in algorithm_files.items():
        print(f"加载 {name} 的数据...")
        algorithm_data[name] = load_population_history(file_path)

    # 可视化多个算法的对比
    output_dir = 'population_distribution_comparison_plots_single/' + Instance
    visualize_multiple_algorithms(algorithm_data, output_dir)
