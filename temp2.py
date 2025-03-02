def find_earliest_completion(machine_data, component_list):
    """
    查找特定组件列表中最早完成加工的时间及对应的组件编号。
    :param machine_data: 机器的加工数据（如Machine_processing_data_17）。
    :param component_list: 组件列表（如D_component1）。
    :return: 最早完成时间及对应的组件编号。
    """
    earliest_time = float('inf')
    component_id = None
    for item in machine_data:
        if item[0] in component_list and item[1] < earliest_time:
            earliest_time = item[1]
            component_id = item[0]
    return earliest_time, component_id

def process(D, F, D_component1, D_component2, F_component1, F_component2, Machine_processing_data_17, Machine_processing_data_20):
    """
    根据给定的数据，确定最佳加工顺序及其配套关系。
    """
    result_D = []
    result_F = []
    current_time = 0
    
    while D or F:
        # 对于D类型成品
        time_d_comp1, comp1_id_d = find_earliest_completion(Machine_processing_data_17, D_component1)
        time_d_comp2, comp2_id_d = find_earliest_completion(Machine_processing_data_20, D_component2)
        earliest_time_d = max(time_d_comp1, time_d_comp2)
        
        # 对于F类型成品
        time_f_comp1, comp1_id_f = find_earliest_completion(Machine_processing_data_17, F_component1)
        time_f_comp2, comp2_id_f = find_earliest_completion(Machine_processing_data_20, F_component2)
        earliest_time_f = max(time_f_comp1, time_f_comp2)
        
        if (not D) or (F and earliest_time_f < earliest_time_d):
            # 加工F类型成品
            if F:
                finished_item = F.pop(0)
                result_F.append((finished_item, comp1_id_f, comp2_id_f))
                current_time = max(current_time, earliest_time_f) + 1
        else:
            # 加工D类型成品
            if D:
                finished_item = D.pop(0)
                result_D.append((finished_item, comp1_id_d, comp2_id_d))
                current_time = max(current_time, earliest_time_d) + 1
                
    print("D成品加工顺序及配套关系:", result_D)
    print("F成品加工顺序及配套关系:", result_F)

# 示例数据
D=[6,15,33,42]
F=[24,51]
D_component1=[4,13,31,40]
D_component2=[5,14,32,41]
F_component1=[22,49]
F_component2=[23,50]
Machine_processing_data_17=[[40, 4.3], [4, 5.6], [49, 7.1], [22, 8.6], [13, 9.9], [31, 11.2]]
Machine_processing_data_20=[[14, 4.1], [32, 5.5], [5, 6.9], [41, 8.3], [50, 10.3], [23, 12.1]]

process(D, F, D_component1, D_component2, F_component1, F_component2, Machine_processing_data_17, Machine_processing_data_20)