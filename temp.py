def process_job(job_num, OS_Mechanism_Assembly, a, b, c, d, JM, Processing_time, Machines, Jobs, fitness):
    O_num = Jobs[job_num].Current_Processed()
    Machine = JM[job_num][O_num]
    P_t = Processing_time[job_num][O_num][Machine]
    Machine_end_time = Machines[Machine].End_time

    #有问题！！！
    # if job_num == OS_Mechanism_Assembly[0]:
    #     start = max(Machine_end_time, min(a[b.index([job_num, 3])], a[b.index([OS_Mechanism_Assembly[1], 3])]),
    #                 min(c[d.index([OS_Mechanism_Assembly[1] - 1, 3])], c[d.index([OS_Mechanism_Assembly[2] - 1, 3])]))
    # elif job_num == OS_Mechanism_Assembly[1]:
    #     start = max(Machine_end_time, max(max(a[b.index([OS_Mechanism_Assembly[0], 3])], a[b.index([OS_Mechanism_Assembly[1], 3])]),
    #                                       max(c[d.index([OS_Mechanism_Assembly[0] - 1, 3])], c[d.index([OS_Mechanism_Assembly[2] - 1, 3])])))
    # else:
    #     start = max(Machine_end_time, a[b.index([job_num, 3])], c[d.index([job_num - 1, 3])])

    End_work_time = start + P_t
    Jobs[job_num]._Input(start, End_work_time, Machine)
    if End_work_time > fitness:
        fitness = End_work_time
    Machines[Machine]._Input(job_num, start, P_t, O_num)
    return fitness

a = copy.deepcopy(self.Machines[20 - 1].O_end)
b = copy.deepcopy(self.Machines[20 - 1].assigned_task)
c = copy.deepcopy(self.Machines[17 - 1].O_end)
d = copy.deepcopy(self.Machines[17 - 1].assigned_task)

OS_Mechanism_Assembly = [num for num in OS if num in {5, 14, 23}]   # 获得OS中的5，14，23这三个，并保留其在OS中的顺序

fitness = self.fitness
for job_num in OS_Mechanism_Assembly:
    fitness = process_job(job_num, OS_Mechanism_Assembly, a, b, c, d, JM, self.Processing_time, self.Machines, self.Jobs, fitness)

self.fitness = fitness