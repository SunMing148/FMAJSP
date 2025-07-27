# 先运行update1,再运行update2，再去删每个矩阵的第一行末尾的]
# 用于生成随机数据集
import os
import re
import random
import ast
from tqdm import tqdm


def find_matching_bracket(s, start_idx):
    """找到与起始括号匹配的结束括号位置"""
    depth = 0
    for i in range(start_idx, len(s)):
        if s[i] == '[':
            depth += 1
        elif s[i] == ']':
            depth -= 1
            if depth == 0:
                return i
    return -1


def process_line(line, matrix_name, row_idx, config):
    """处理单行数据，替换非9999的值"""
    start_idx = line.find('[')
    if start_idx == -1:
        return line

    end_idx = find_matching_bracket(line, start_idx)
    if end_idx == -1:
        return line

    # 提取列表字符串
    list_str = line[start_idx:end_idx + 1]

    try:
        # 安全解析列表
        row_data = ast.literal_eval(list_str)
    except (SyntaxError, ValueError):
        return line

    # 检查是否是嵌套列表（处理错误格式）
    while isinstance(row_data, list) and len(row_data) == 1 and isinstance(row_data[0], list):
        row_data = row_data[0]

    # 获取配置
    key = f"{matrix_name}_row{row_idx}"
    if key not in config:
        return line

    low, high = config[key]

    # 查找非9999的值
    non_9999 = None
    for value in row_data:
        if value != 9999:
            non_9999 = value
            break

    # 如果找到非9999值，则替换整行
    if non_9999 is not None:
        # 生成新的随机值
        new_val = round(random.uniform(low, high), 2)
        if new_val.is_integer():
            new_val = int(new_val)

        # 创建新行数据
        new_row = []
        for value in row_data:
            if value == 9999:
                new_row.append(9999)
            else:
                new_row.append(new_val)

        # 生成新的列表字符串，保持原始格式
        formatted_elements = []
        for value in new_row:
            if value == 9999:
                s = '9999'
            else:
                s = f"{value:.2f}" if isinstance(value, float) else str(value)
                if '.' in s:
                    s = s.rjust(6)
                else:
                    s = s.rjust(5)
            formatted_elements.append(s)

        new_list_str = '[' + ', '.join(formatted_elements) + ']'
        new_line = line[:start_idx] + new_list_str + line[end_idx + 1:]
        return new_line

    return line


def process_file(file_path, config):
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 定义要处理的矩阵及其行数
    matrix_info = {
        'A_PT': 10, 'B_PT': 10, 'C_PT': 10,
        'D1_PT': 3, 'D2_PT': 3, 'D3_PT': 1,
        'E1_PT': 3, 'E2_PT': 3, 'E3_PT': 1,
        'F_PT': 4, 'G_PT': 4
    }

    new_lines = []
    current_matrix = None
    rows_remaining = 0
    row_idx = 0

    for line in lines:
        if current_matrix is None:
            # 检查是否开始新矩阵
            for matrix_name in matrix_info:
                if re.match(rf"\s*{matrix_name}\s*=", line):
                    current_matrix = matrix_name
                    rows_remaining = matrix_info[matrix_name]
                    row_idx = 0
                    break

            if current_matrix:
                # 处理矩阵定义行
                new_line = process_line(line, current_matrix, row_idx, config)
                new_lines.append(new_line)
                row_idx += 1
                rows_remaining -= 1

                if rows_remaining == 0:
                    current_matrix = None
            else:
                new_lines.append(line)
        else:
            # 处理矩阵数据行
            if rows_remaining > 0:
                new_line = process_line(line, current_matrix, row_idx, config)
                new_lines.append(new_line)
                row_idx += 1
                rows_remaining -= 1

                if rows_remaining == 0:
                    current_matrix = None
            else:
                new_lines.append(line)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def process_folder(folder_path, config):
    """处理文件夹中的所有文件"""
    files = [f for f in os.listdir(folder_path)
             if f.endswith('.py') and f.startswith('Instance_')]

    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(folder_path, filename)
        process_file(file_path, config)


if __name__ == "__main__":
    # 配置随机数范围 - 需要为每个矩阵的每一行设置
    config = {
        # A_PT 矩阵配置 (10行)
        "A_PT_row0": (7.0, 9.25),
        "A_PT_row1": (0.65, 0.9),
        "A_PT_row2": (1, 1.25),
        "A_PT_row3": (0.575, 0.75),
        "A_PT_row4": (2.5, 3),
        "A_PT_row5": (0.8, 0.9),
        "A_PT_row6": (0.9, 1.0),
        "A_PT_row7": (1, 1.2),
        "A_PT_row8": (1, 1.4),
        "A_PT_row9": (1.5, 2.5),

        # B_PT 矩阵配置 (10行)
        "B_PT_row0": (7.0, 9.25),
        "B_PT_row1": (0.65, 0.9),
        "B_PT_row2": (1, 1.25),
        "B_PT_row3": (0.575, 0.75),
        "B_PT_row4": (2.5, 3),
        "B_PT_row5": (0.8, 0.9),
        "B_PT_row6": (0.9, 1.0),
        "B_PT_row7": (1, 1.2),
        "B_PT_row8": (1, 1.4),
        "B_PT_row9": (1.5, 2.5),

        # C_PT 矩阵配置 (10行)
        "C_PT_row0": (7.0, 9.25),
        "C_PT_row1": (0.65, 0.9),
        "C_PT_row2": (1, 1.25),
        "C_PT_row3": (0.575, 0.75),
        "C_PT_row4": (2.5, 3),
        "C_PT_row5": (0.8, 0.9),
        "C_PT_row6": (0.9, 1.0),
        "C_PT_row7": (1, 1.2),
        "C_PT_row8": (1, 1.4),
        "C_PT_row9": (1.5, 2.5),

        # D1_PT 矩阵配置 (3行)
        "D1_PT_row0": (6, 7),
        "D1_PT_row1": (6.5, 7.5),
        "D1_PT_row2": (3, 3),

        # D2_PT 矩阵配置 (3行)
        "D2_PT_row0": (6, 8),
        "D2_PT_row1": (1.5, 2.5),
        "D2_PT_row2": (7, 9),

        # D3_PT 矩阵配置 (1行)
        "D3_PT_row0": (5, 6),

        # E1_PT 矩阵配置 (3行)
        "E1_PT_row0": (6, 7),
        "E1_PT_row1": (6.5, 7.5),
        "E1_PT_row2": (3, 3),

        # E2_PT 矩阵配置 (3行)
        "E2_PT_row0": (6, 8),
        "E2_PT_row1": (1.5, 2.5),
        "E2_PT_row2": (7, 9),

        # E3_PT 矩阵配置 (1行)
        "E3_PT_row0": (5, 6),

        # F_PT 矩阵配置 (4行)
        "F_PT_row0": (2.5, 3.5),
        "F_PT_row1": (1, 1.25),
        "F_PT_row2": (2, 3),
        "F_PT_row3": (2, 2),

        # G_PT 矩阵配置 (4行)
        "G_PT_row0": (2.5, 3.5),
        "G_PT_row1": (1, 1.25),
        "G_PT_row2": (2, 3),
        "G_PT_row3": (2, 2),

    }

    # 设置文件夹路径
    folder_path = r'dataset/randomDataset/small/SET5'
    process_folder(folder_path, config)
    print("所有文件处理完成！")
