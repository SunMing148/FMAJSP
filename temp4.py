"""
temp4 -
Author:Sun_M
Date:2025/3/2
"""
import copy

F = [24]
temp_F=copy.deepcopy(F)
for i in range(1, 3):  # 从1到k-1，逐次添加新元素
    new_values = [x + 27 * i for x in temp_F]  # 计算新的元素值
    F.extend(new_values)  # 将新元素添加到数组中

print(F)