# 先运行update1,再运行update2，再去删每个矩阵的第一行末尾的]
import os
import random
import re

def replace_matrix_values(matrix_str, ranges):
    lines = matrix_str.strip()[1:-1].split('],')  # 去除最外层 []
    new_lines = []

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 提取数值
        values = list(map(lambda x: x.strip(), re.findall(r'\d+\.?\d*|9999', line)))

        # 获取当前行的随机范围
        low, high = ranges[idx]
        unique_values = set(values)

        if '9999' in unique_values:
            unique_values.remove('9999')

        if not unique_values:
            new_line = line
        else:
            replace_value = round(random.uniform(low, high), 2)
            new_values = [str(replace_value) if v != '9999' else v for v in values]
            new_line = '[' + ', '.join(new_values) + ']'

        new_lines.append(new_line)

    return '[{}]'.format(',\n '.join(new_lines))

def process_file(file_path, replace_ranges):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for matrix_name in replace_ranges:
        matrix_pattern = rf'{matrix_name}\s*=\s*(\[[\s\S]*?\])'
        match = re.search(matrix_pattern, content)

        if match:
            matrix_str = match.group(1)
            new_matrix_str = replace_matrix_values(matrix_str, replace_ranges[matrix_name])

            # 替换原文本
            content = content.replace(matrix_str, new_matrix_str, 1)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Processed {file_path}")

def batch_process_folder(folder_path, replace_ranges):
    for filename in os.listdir(folder_path):
        if filename.startswith("Instance_L") and filename.endswith(".py"):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path, replace_ranges)

# 示例调用
if __name__ == "__main__":
    folder_path = r'dataset/randomDataset/small/SET5'

    # 指定每行的替换区间，按照行顺序给出
    replace_ranges = {
        'A_PT': [(7, 9.25), (0.65, 0.9), (1, 1.25), (0.575, 0.75), (2.5, 3),
                 (0.8, 0.9), (0.9, 1.0), (1, 1.2), (1, 1.4), (1.5, 2.5)],
        'B_PT': [(7, 9.25), (0.65, 0.9), (1, 1.25), (0.575, 0.75), (2.5, 3),
                 (0.8, 0.9), (0.9, 1.0), (1, 1.2), (1, 1.4), (1.5, 2.5)],
        'C_PT': [(7, 9.25), (0.65, 0.9), (1, 1.25), (0.575, 0.75), (2.5, 3),
                 (0.8, 0.9), (0.9, 1.0), (1, 1.2), (1, 1.4), (1.5, 2.5)],
        'D1_PT': [(6, 7), (6.5, 7.5), (3, 3)],
        'D2_PT': [(6, 8), (1.5, 2.5), (7, 9)],
        'D3_PT': [(5, 6)],
        'E1_PT': [(6, 7), (6.5, 7.5), (3, 3)],
        'E2_PT': [(6, 8), (1.5, 2.5), (7, 9)],
        'E3_PT': [(5, 6)],
        'F_PT': [(2.5, 3.5), (1, 1.25), (2, 3), (2, 2)],
        'G_PT': [(2.5, 3.5), (1, 1.25), (2, 3), (2, 2)],
    }

    batch_process_folder(folder_path, replace_ranges)
