def update_job_groups(k):
    job_groups = {
        'Reds': [1, 2, 3],                        
        'Blues': [10, 11, 12],                    
        'Greens': [19, 20, 21],                   
        'Oranges': [4, 5, 6, 13, 14, 15],         
        'Purples': [22, 23, 24],                  
        'YlOrBr': [7, 8, 9, 16, 17, 18],          
        'PuBu': [25, 26, 27]                      
    }

    # # 如果k为1，直接返回原始job_groups
    # if k == 1:
    #     return job_groups

    # 遍历job_groups中的每个键值对
    for key, values in job_groups.items():
        original_values = values[:]  # 保存原始数组的副本
        for i in range(1, k):  # 从1到k-1，逐次添加新元素
            new_values = [x + 27 * i for x in original_values]  # 计算新的元素值
            job_groups[key].extend(new_values)  # 将新元素添加到数组中

    return job_groups

# 示例调用
k = 1
updated_job_groups = update_job_groups(k)
print(updated_job_groups)