import random
import numpy as np

Total_Machine = [10, 20, 30, 40, 50]  # 全部机器
Initial_Job_num = 20  # 初始工件个数
Job_insert = [50, 100, 200]  # 工件新到达个数
DDT = [0.5, 1.0, 1.5]  # 工件紧急程度
E_ave = [50, 100, 200]  # 指数分布


def Instance_Generator(M_num, E_ave, New_insert, DDT):
    '''
    生成了有关工件、机器和调度的详细信息，包括处理时间、工件到达时间和交付时间
    :param M_num: 机器数量
    :param E_ave: 新工件到达时间的指数分布的平均值
    :param New_insert: 新插入的工件数量
    :param DDT: 交付时间因子
    :return:
        Processing_time: 各工件的处理时间
        A1: 初始工件和新工件的到达时间
        D1: 初始工件和新工件的交付时间
        M_num: 机器数量
        Op_num: 每个工件的工序数量
        J: 工件字典
        O_num: 总的工序数量
        J_num: 总的工件数量
    '''
    Initial_Job_num = 5     # 初始工件数
    Op_num = [random.randint(1, 5) for _ in range(New_insert + Initial_Job_num)]    # 每个工件的工序数
    Processing_time = []    # 三维列表(工件数*工序数*可选设备)，-1表示该机器不能处理该工序，其他值表示处理时间
    # 生成每个工件的处理时间
    for i in range(Initial_Job_num + New_insert):
        Job_i = []
        for j in range(Op_num[i]):
            k = random.randint(1, M_num - 2)
            T = list(range(M_num))
            random.shuffle(T)
            T = T[:k + 1]
            O_i = list(np.ones(M_num) * (-1))
            for M_i in range(len(O_i)):
                if M_i in T:
                    O_i[M_i] = random.randint(1, 50)
            Job_i.append(O_i)
        Processing_time.append(Job_i)

    # 生成到达时间
    A1 = [0] * Initial_Job_num
    A = np.random.exponential(E_ave, size=New_insert)
    A = [int(a) for a in A]  # 转换为整数
    A1.extend(A)  # 初始工件的到达时间为0，新工件的到达时间随机生成

    # 计算每个工件所有工序的平均处理时间的总和
    T_ijave = []
    for i in range(Initial_Job_num + New_insert):
        Tad = []
        for j in range(Op_num[i]):
            T_ijk = [k for k in Processing_time[i][j] if k != -1]
            Tad.append(sum(T_ijk) / len(T_ijk))
        T_ijave.append(sum(Tad))
    # 生成交付时间
    D1 = [int(T_ijave[i] * DDT) for i in range(Initial_Job_num)]
    D = [int(A1[i] + T_ijave[i] * DDT) for i in range(Initial_Job_num, Initial_Job_num + New_insert)]
    D1.extend(D)
    O_num = sum(Op_num)     # 计算总工序数量
    J = dict(enumerate(Op_num))     # 工件字典
    J_num = Initial_Job_num + New_insert    # 总工件数量

    return Processing_time, A1, D1, M_num, Op_num, J, O_num, J_num


Processing_time, A, D, M_num, Op_num, J, O_num, J_num = Instance_Generator(10, 50, 10, 0.5)
print('Processing_time, A, D, M_num, Op_num, J, O_num, J_num:\n',Processing_time, A, D, M_num, Op_num, J, O_num, J_num)
# print(Processing_time)
# print(A)
# print(D)
# print(M_num)
# print(Op_num)
# print(J)
# print(O_num)
# print(J_num)
