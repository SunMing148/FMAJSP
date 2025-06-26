import os
import pandas as pd
import numpy as np
import re

# 定义数据集路径
RANDOM_DATASET_DIR = "dataset/randomDataset/"
EXCEL_FILE_PATH = "随机数据集 - 副本.xlsx"  # Excel文件路径，根据实际情况修改

# 定义数据集规模与sheet名的映射
SCALE_SHEET_MAP = {
    "small": "small规模随机数据集",
    "middle": "middle规模随机数据集",
    "big": "big规模随机数据集"
}

# 定义文件模式正则表达式
INSTANCE_FILE_PATTERN = re.compile(r"Instance_L\((\d+),(\d+),(\d+)\)_P\((\d+),(\d+),(\d+)\).py")


def read_excel_data(excel_file_path):
    """
    读取Excel文件中的数据

    Args:
        excel_file_path (str): Excel文件路径

    Returns:
        dict: 包含三个sheet数据的字典，每个sheet对应一个规模的数据
    """
    excel_data = {}

    # 读取每个sheet的数据
    for scale, sheet_name in SCALE_SHEET_MAP.items():
        try:
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            excel_data[scale] = df
        except Exception as e:
            print(f"Error reading sheet {sheet_name}: {e}")
            excel_data[scale] = pd.DataFrame()

    return excel_data


def parse_instance_file_name(file_name):
    """
    解析Instance文件名，提取L和P的值

    Args:
        file_name (str): Instance文件名

    Returns:
        tuple: (L1, L2, L3, P1, P2, P3) 或 None（如果文件名格式不匹配）
    """
    match = INSTANCE_FILE_PATTERN.match(file_name)
    if match:
        L1, L2, L3, P1, P2, P3 = map(int, match.groups())
        return (L1, L2, L3, P1, P2, P3)
    return None


def find_matching_row(excel_df, set_num, L_values, P_values):
    """
    在Excel数据框中查找匹配的行

    Args:
        excel_df (pd.DataFrame): Excel数据框
        set_num (int): SET编号
        L_values (tuple): L值元组 (L1, L2, L3)
        P_values (tuple): P值元组 (P1, P2, P3)

    Returns:
        pd.Series: 匹配的行或 None
    """
    # 创建匹配的L(P)格式字符串
    LP_str = f"L({L_values[0]},{L_values[1]},{L_values[2]})*P({P_values[0]},{P_values[1]},{P_values[2]})"

    # 查找匹配的行
    matching_rows = excel_df[
        (excel_df["SET"] == set_num) &
        (excel_df["L(L1,L2,L3)*P(MTZ1,MTZ2,MTZ3)"] == LP_str)
        ]

    if not matching_rows.empty:
        return matching_rows.iloc[0]
    return None


def extract_processing_time(row):
    """
    从Excel行中提取工序加工时间数据

    Args:
        row (pd.Series): Excel数据行

    Returns:
        dict: 包含A、B、C、D、E、F、G的加工时间数据
    """
    # 示例：提取三个表格的数据并转换为字典
    # 根据实际的列名和数据结构进行调整

    # 假设工序加工时间数据包含三个表格，分别命名为 'table1', 'table2', 'table3'
    # 实际应用中需要根据你的Excel文件结构调整
    table1 = row.get('工序加工时间数据1', None)
    table2 = row.get('工序加工时间数据2', None)
    table3 = row.get('工序加工时间数据3', None)

    # 将表格数据转换为字典格式
    processing_time = {
        'A': table1,
        'B': table2,
        'C': table3
        # 添加其他需要的键值对
    }

    return processing_time


def update_matrices(instance_file_path, processing_time_data):
    """
    更新Instance文件中的矩阵数据

    Args:
        instance_file_path (str): Instance文件路径
        processing_time_data (dict): 包含A、B、C、D、E、F、G的加工时间数据
    """
    # 示例：更新Instance文件中的矩阵数据
    # 实际应用中需要根据你的Instance文件结构调整

    try:
        # 读取Instance文件内容
        with open(instance_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 在这里实现更新矩阵逻辑
        # 示例：简单打印，实际应用中替换为矩阵更新代码
        print(f"Updating file: {instance_file_path}")
        print("Processing time data:", processing_time_data)

        # 实际更新代码示例（需要根据实际文件结构调整）
        # for matrix_name, data in processing_time_data.items():
        #     if data is not None:
        #         # 找到对应的矩阵并更新
        #         # 这里需要实现具体的矩阵查找和更新逻辑
        #         pass

        # 写回更新后的内容
        # with open(instance_file_path, 'w', encoding='utf-8') as file:
        #     file.writelines(lines)

    except Exception as e:
        print(f"Error updating file {instance_file_path}: {e}")


def process_dataset():
    """
    处理整个数据集
    """
    # 读取Excel数据
    excel_data = read_excel_data(EXCEL_FILE_PATH)

    # 遍历每个规模
    for scale, excel_df in excel_data.items():
        scale_dir = os.path.join(RANDOM_DATASET_DIR, scale)

        # 检查目录是否存在
        if not os.path.exists(scale_dir):
            print(f"Directory not found: {scale_dir}")
            continue

        # 遍历每个SET目录
        for set_num in range(1, 6):  # SET1到SET5
            set_dir = os.path.join(scale_dir, f"SET{set_num}")

            # 检查目录是否存在
            if not os.path.exists(set_dir):
                print(f"Directory not found: {set_dir}")
                continue

            # 遍历SET目录下的每个Instance文件
            for file_name in os.listdir(set_dir):
                if file_name.startswith("Instance_L") and file_name.endswith(".py"):
                    file_path = os.path.join(set_dir, file_name)

                    # 解析文件名
                    parsed_result = parse_instance_file_name(file_name)
                    if parsed_result:
                        L_values = parsed_result[:3]
                        P_values = parsed_result[3:]

                        # 查找匹配的Excel行
                        matching_row = find_matching_row(excel_df, set_num, L_values, P_values)
                        if matching_row is not None:
                            # 提取工序加工时间数据
                            processing_time_data = extract_processing_time(matching_row)

                            # 更新Instance文件中的矩阵数据
                            update_matrices(file_path, processing_time_data)
                            print(f"Updated: {file_path}")
                        else:
                            print(f"No matching row found for: {file_name}")
                    else:
                        print(f"Invalid file name format: {file_name}")


if __name__ == "__main__":
    process_dataset()