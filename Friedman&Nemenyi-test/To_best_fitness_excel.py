import pandas as pd


def transfer_excel_data():
    try:
        # 创建包含120列的DataFrame (A-DL)
        target_df = pd.DataFrame(columns=[chr(65 + i) for i in range(120)])

        # 定义要处理的源sheet和对应的列
        sheets_and_columns = [
            ('不同case实验数据小small规模随机数据集', 3),  # D列
            ('不同case实验数据中middle规模随机数据集', 3),  # D列
            ('不同case实验数据大big规模随机数据集', 3)  # D列
        ]

        # 定义要处理的行组 (D, L, T, AB, AJ)
        row_groups = [
            {'name': 'D', 'col_index': 3},  # D列
            {'name': 'L', 'col_index': 11},  # L列
            {'name': 'T', 'col_index': 19},  # T列
            {'name': 'AB', 'col_index': 27},  # AB列
            {'name': 'AJ', 'col_index': 35}  # AJ列
        ]

        # 处理每一行组
        for row_idx, row_group in enumerate(row_groups):
            source_values = []

            # 处理每个sheet
            for sheet_name, base_col in sheets_and_columns:
                # 读取源Excel文件中的指定sheet
                df = pd.read_excel('1.xlsx', sheet_name=sheet_name)

                # 获取当前列组的列索引
                col_index = row_group['col_index']

                # 获取指定单元格的值 (15, 30, ..., 600行)
                for i in range(14, 600, 15):  # 行索引从0开始，所以15行对应索引14
                    if i < len(df) and col_index < len(df.columns):
                        source_values.append(df.iloc[i, col_index])
                    else:
                        source_values.append(None)

            # 将值写入目标DataFrame的对应行
            target_df.loc[row_idx] = source_values

        # 写入到新的Excel文件
        target_df.to_excel('2.xlsx', sheet_name='sheet1', index=False, header=False)

        print("数据转移完成！")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    transfer_excel_data()