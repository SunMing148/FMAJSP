"""
Processing_time：工件各工序对应各机器加工时间矩阵
J：各工件对应的工序数字典
M_num：加工机器数
O_num：加工工序数
J_num：工件个数
"""

      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14        工序
A_PT  =  [[7.13, 7.13, 7.13, 7.13, 7.13, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999,   0.86,   0.86, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.04,   1.04, 9999, 9999, 9999, 9999, 9999],      # 3
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.69, 9999, 9999, 9999, 9999],      # 4
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   2.86, 9999, 9999, 9999],      # 5
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.85, 9999, 9999],      # 6
          [9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.93,   0.93, 9999, 9999, 9999, 9999, 9999],      # 7
          [9999, 9999, 9999, 9999, 9999,   1.03,   1.03, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 8
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.19, 9999],      # 9
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.83]]      # 10

     #  机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14        工序
B_PT  =  [[8.53, 8.53, 8.53, 8.53, 8.53, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999,   0.69,   0.69, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.02,   1.02, 9999, 9999, 9999, 9999, 9999],      # 3
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.71, 9999, 9999, 9999, 9999],      # 4
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   2.70, 9999, 9999, 9999],      # 5
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.81, 9999, 9999],      # 6
          [9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.96,   0.96, 9999, 9999, 9999, 9999, 9999],      # 7
          [9999, 9999, 9999, 9999, 9999,   1.15,   1.15, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 8
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.16, 9999],      # 9
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.81]]      # 10

      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14        工序
C_PT  =  [[8.11, 8.11, 8.11, 8.11, 8.11, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999,   0.84,   0.84, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.13,   1.13, 9999, 9999, 9999, 9999, 9999],      # 3
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.63, 9999, 9999, 9999, 9999],      # 4
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   2.70, 9999, 9999, 9999],      # 5
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.86, 9999, 9999],      # 6
          [9999, 9999, 9999, 9999, 9999, 9999, 9999,   0.99,   0.99, 9999, 9999, 9999, 9999, 9999],      # 7
          [9999, 9999, 9999, 9999, 9999,   1.15,   1.15, 9999, 9999, 9999, 9999, 9999, 9999, 9999],      # 8
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.29, 9999],      # 9
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.66]]      # 10


      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18        工序
D1_PT  = [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 6.45, 6.45, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   7.33, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,     3]]      # 3

      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22        工序
D2_PT  = [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 6.44, 6.44, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   2.09, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   8.88]]      # 3

      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23       工序
D3_PT  = [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 5.21]]      # 1


      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18     工序
E1_PT  = [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 6.28, 6.28, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   6.98, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,     3]]      # 3

      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22       工序
E2_PT  = [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 7.63, 7.63, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.80, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   8.79]]      # 3

      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23        工序
E3_PT  = [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 5.12]]      # 1



      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29        工序
F_PT  =  [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 2.68, 2.68, 9999, 9999, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.16,   1.16, 9999, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   2.47, 9999],      # 3
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,     2]]      # 4

      # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29        工序
G_PT  =  [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 2.99, 2.99, 9999, 9999, 9999, 9999],      # 1
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   1.20,   1.20, 9999, 9999],      # 2
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,   2.77, 9999],      # 3
          [9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999,     2]]      # 4

       # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30       工序
M1_PT  =  [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 10.4]]      #1

       # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30      工序
M2_PT  =  [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 12.5]]      #1

       # 机器   1     2     3     4     5    6     7      8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30       工序
M3_PT  =  [[9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 9999, 15.2]]      #1

# 特殊机器ID号 代码中从0开始编号，因此要机器编号-1
Special_Machine_ID = {
    "L1_last_machine_ID": 14 - 1,               # 互感器产线最后一个工序的机器ID号
    "L2_last_machine_ID": 23 - 1,               # 机构产线最后一个工序（装配工序）的机器ID号
    "L2_pre_assembly_machine_ID_low": 18 - 1,   # 机构产线最后装配工序的前置工序（编号较低的，对应“机构零件铆压”工序）的机器ID号
    "L2_pre_assembly_machine_ID_high": 22 - 1,  # 机构产线最后装配工序的前置工序（编号较高的，对应“机构主轴装配”工序）的机器ID号
    "L3_last_machine_ID": 29 - 1,               # 灭弧室产线最后一个工序的机器ID号
    "L4_last_machine_ID": 30 - 1 # 总装产线最后一个工序的机器ID号
}

k1 = 5 #生产MTZ1的数量
k2 = 6 #生产MTZ2的数量
k3 = 6 #生产MTZ3的数量

M_num = 30   # 总机器数量 = L1 + L2 + L3 + L4   L4固定为1 因此值为前三条产线机器数量加一

def generate_dict(n):
    pattern = [10, 10, 10, 3, 3, 1, 4, 4, 4, 1]
    result = {}
    for cycle in range(n):
        for i, value in enumerate(pattern, start=1 + cycle * len(pattern)):
            result[i] = value
    return result

MTZ1_PT = [A_PT]*3 + [D1_PT] + [D2_PT] + [D3_PT] + [F_PT]*3 + [M1_PT]
MTZ2_PT = [B_PT]*3 + [D1_PT] + [D2_PT] + [D3_PT] + [G_PT]*3 + [M2_PT]
MTZ3_PT = [C_PT]*3 + [E1_PT] + [E2_PT] + [E3_PT] + [G_PT]*3 + [M3_PT]

Processing_time = MTZ1_PT*k1 + MTZ2_PT*k2 + MTZ3_PT*k3

kn = k1 + k2 + k3  #生产MTZ型产品总数量
J = generate_dict(kn)

# J = {
    # 1: 10,
    # 2: 10,
    # 3: 10,
    # 4: 3,
    # 5: 3,
    # 6: 1,
    # 7: 4,
    # 8: 4,
    # 9: 4,
    # 10: 1,
# }

O_num = (10 * 3 + 3 + 3 + 1 + 4 * 3 + 1 * 1) * kn
J_num = (3 + 1 + 1 + 1 + 3 + 1) * kn

Ap = [5 + i * 10 for i in range(kn)]  # 机器21装配工序 （第一次装配工序） 对应的Job编号（编号减1）
Ap.extend([9 + i * 10 for i in range(kn)])  # 机器26装配工序 （总装工序） 对应的Job编号（编号减1） 编号kn次是因为每个产品都有这一步

A = [1 + i * 10 for i in range(k1)]  # MTZ1的A部件编号
A.extend([2 + i * 10 for i in range(k1)])  # MTZ1的A部件编号
A.extend([3 + i * 10 for i in range(k1)])  # MTZ1的A部件编号

B = [(1 + 10 * k1) + i * 10 for i in range(k2)]  # MTZ2的B部件编号
B.extend([(2 + 10 * k1) + i * 10 for i in range(k2)])  # MTZ2的B部件编号
B.extend([(3 + 10 * k1) + i * 10 for i in range(k2)])  # MTZ2的B部件编号

C = [(1 + 10 * (k1 + k2)) + i * 10 for i in range(k3)]  # MTZ3的C部件编号
C.extend([(2 + 10 * (k1 + k2)) + i * 10 for i in range(k3)])  # MTZ3的C部件编号
C.extend([(3 + 10 * (k1 + k2)) + i * 10 for i in range(k3)])  # MTZ3的C部件编号

D = [6 + i * 10 for i in range(k1)]  # MTZ1的D部件编号
D.extend([(6 + k1 * 10) + i * 10 for i in range(k2)])  # MTZ2的D部件编号 理论上可合写为self.D = [6 + i * 10 for i in range(k1+k2)]

D_component1 = [4 + i * 10 for i in range(k1)]  # MTZ1的D_component1部件编号
D_component1.extend([(4 + k1 * 10) + i * 10 for i in range(k2)])  # MTZ2的D_component1部件编号 理论上可合写为self.D_component1 = [4 + i * 10 for i in range(k1+k2)]

D_component2 = [5 + i * 10 for i in range(k1)]  # MTZ1的D_component1部件编号
D_component2.extend([(5 + k1 * 10) + i * 10 for i in range(k2)])  # MTZ2的D_component1部件编号 理论上可合写为self.D_component2 = [5 + i * 10 for i in range(k1+k2)]

E = [(6 + (k1 + k2) * 10) + i * 10 for i in range(k3)]  # MTZ3的E部件编号
E_component1 = [(4 + (k1 + k2) * 10) + i * 10 for i in range(k3)]  # MTZ3的E_component1部件编号
E_component2 = [(5 + (k1 + k2) * 10) + i * 10 for i in range(k3)]  # MTZ3的E_component2部件编号


F = [7 + i * 10 for i in range(k1)]  # MTZ1的F部件编号
F.extend([8 + i * 10 for i in range(k1)])  # MTZ1的F部件编号
F.extend([9 + i * 10 for i in range(k1)])  # MTZ1的F部件编号

G = [(7 + 10 * k1) + i * 10 for i in range(k2)]  # MTZ2的G部件编号
G.extend([(8 + 10 * k1) + i * 10 for i in range(k2)])  # MTZ2的G部件编号
G.extend([(9 + 10 * k1) + i * 10 for i in range(k2)])  # MTZ2的G部件编号
G.extend([(7 + 10 * (k1 + k2)) + i * 10 for i in range(k3)])  # MTZ3的G部件编号
G.extend([(8 + 10 * (k1 + k2)) + i * 10 for i in range(k3)])  # MTZ3的G部件编号
G.extend([(9 + 10 * (k1 + k2)) + i * 10 for i in range(k3)])  # MTZ3的G部件编号

M1 = [10 + i * 10 for i in range(k1)]  # MTZ1的M1部件编号
M2 = [(10 + 10 * k1) + i * 10 for i in range(k2)]  # MTZ2的M2部件编号
M3 = [(10 + 10 * (k1 + k2)) + i * 10 for i in range(k3)]  # MTZ3的G部件编号\

Job_serial_number = {
    "Ap": Ap,
    "A": A,
    "B": B,
    "C": C,
    "D": D,
    "D_component1": D_component1,
    "D_component2": D_component2,
    "E": E,
    "E_component1": E_component1,
    "E_component2": E_component2,
    "F": F,
    "G": G,
    "M1": M1,
    "M2": M2,
    "M3": M3
}
