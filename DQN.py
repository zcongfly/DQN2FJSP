import numpy as np
import os
import random
from collections import deque
from tensorflow.keras import layers, models
from Job_Shop import Situation
from tensorflow.keras.optimizers import Adam
from Instance_Generator import Processing_time, A, D, M_num, Op_num, J, O_num, J_num
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""     # -1：禁用GPU，空字符：自动分配GPU，0等编号：使用特定编号的GPU

class DQN:
    def __init__(self, ):
        self.Hid_Size = 30

        # ------------Hidden layer=5   30 nodes each layer--------------
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(7,)))     # 输入层维度为7
        model.add(layers.Dense(self.Hid_Size, name='l1'))    # 5个隐藏层，每个隐藏层有30个节点
        model.add(layers.Dense(self.Hid_Size, name='l2'))
        model.add(layers.Dense(self.Hid_Size, name='l3'))
        model.add(layers.Dense(self.Hid_Size, name='l4'))
        model.add(layers.Dense(self.Hid_Size, name='l5'))
        model.add(layers.Dense(6, name='l6'))   # 输出层，输出维度为6，代表6个可选择的动作
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))
        # model.summary()
        self.model = model  # 主网络，用于当前的策略学习和动作选择

        # ------------Q-network Parameters-------------
        self.act_dim = [1, 2, 3, 4, 5, 6]  # 神经网络的输出节点
        self.obs_n = [0, 0, 0, 0, 0, 0, 0]  # 神经网路的输入节点
        self.gama = 0.95  # γ经验折损率
        # self.lr = 0.001  # 学习率
        self.global_step = 0    # 全局步数计数器
        self.update_target_steps = 200  # 更新目标函数的步长，表示每200步更新一次目标网络
        self.target_model = self.model  # 目标网络，初始化为主网络，用于计算目标 Q 值，这个网络的权重会定期从主网络复制过来，用于稳定训练过程

        # -------------------Agent-------------------
        self.e_greedy = 0.6     # ε-贪心策略中的初始探索率（ε）
        self.e_greedy_decrement = 0.0001    # ε的递减值，每次选择动作后减少0.0001
        self.L = 40  # 训练的回合数，表示总共进行40次训练回合

        # ---------------Replay Buffer---------------
        self.buffer = deque(maxlen=2000)    # 经验回放缓冲区，用于存储过去的经验，最大容量为2000。
        self.Batch_size = 10  # 每次进行梯度下降的样本批量大小，设置为10

    def replace_target(self):
        # 将主网络的每一层的权重更新到目标网络
        self.target_model.get_layer(name='l1').set_weights(self.model.get_layer(name='l1').get_weights())
        self.target_model.get_layer(name='l2').set_weights(self.model.get_layer(name='l2').get_weights())
        self.target_model.get_layer(name='l3').set_weights(self.model.get_layer(name='l3').get_weights())
        self.target_model.get_layer(name='l4').set_weights(self.model.get_layer(name='l4').get_weights())
        self.target_model.get_layer(name='l5').set_weights(self.model.get_layer(name='l5').get_weights())
        self.target_model.get_layer(name='l6').set_weights(self.model.get_layer(name='l6').get_weights())

    def replay(self):
        # 从经验回放缓冲区中抽取小批量样本，并利用这些样本来更新主网络的权重
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()
        # replay the history and train the model
        minibatch = random.sample(self.buffer, self.Batch_size)  # 从经验回放缓冲区self.buffer中随机抽取一个小批量的样本（大小为 self.Batch_size）
        # 计算目标Q值并更新主网络
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:    # 如果 done 为 False，表示状态不是终止状态，目标 Q 值是 reward 加上未来奖励的折扣值
                target = (reward + self.gama *
                          np.argmax(self.target_model.predict(next_state)))
            # 如果 done 为 True，表示该状态是终止状态，目标 Q 值即为奖励 reward
            target_f = self.model.predict(state)    # 计算当前状态 state 的 Q 值。
            target_f[0][action] = target    # 更新 target_f 中对应 action 的 Q 值为计算得到的 target
            self.model.fit(state, target_f, epochs=1, verbose=0)    # 用更新后的 target_f 进行主网络的训练
        self.global_step += 1

    def Select_action(self, obs):
        # 根据当前的观察（状态）选择一个动作。它实现了 ε-greedy 策略，这是强化学习中常用的一种平衡探索和利用的方法
        # obs=np.expand_dims(obs,0)
        if random.random() < self.e_greedy: # 探索: 当生成的随机数小于 ε（self.e_greedy），随机选择一个动作
            act = random.randint(0, 5)
        else:   # 利用: 当随机数大于或等于 ε，根据当前模型预测的 Q 值选择最佳动作
            act = np.argmax(self.model.predict(obs))
        self.e_greedy = max(0.01, self.e_greedy - self.e_greedy_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低。确保 ε 不会降到低于 0.01，避免完全依赖模型，保持一定程度的探索。
        return act

    def _append(self, exp):
        self.buffer.append(exp)

    def main(self, J_num, M_num, O_num, J, Processing_time, D, A):
        """
        :param J_num: 作业数量
        :param M_num: 机器数量
        :param O_num: 工序数量
        :param J: 作业相关信息
        :param Processing_time: 加工时间
        :param D: 任务截止时间
        :param A: 其他相关参数
        :return:
        """
        k = 0   # 用于记录训练步数
        x = []  # 用于存储每次训练的轮次
        Total_tard = [] # 用于记录每轮训练后的总迟到时间
        TR = []     # 每轮训练后的总奖励
        # 开始训练
        for i in range(self.L):
            Total_reward = 0    # 累加当前训练轮的总奖励
            x.append(i + 1)
            print('-----------------------开始第', i + 1, '次训练------------------------------')
            obs = [0 for _ in range(7)]     # 初始化 obs（观察）为全零的数组，并调整形状以符合模型输入
            obs = np.expand_dims(obs, 0)
            done = False
            Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A)  # 当前的调度环境
            # 环境交互
            for j in range(O_num):  # 对于每个工序
                k += 1
                # print(obs)
                at = self.Select_action(obs)    # 选择一个动作（at），根据动作执行相应的调度规则
                # print(at)
                if at == 0:
                    at_trans = Sit.rule1()
                if at == 1:
                    at_trans = Sit.rule2()
                if at == 2:
                    at_trans = Sit.rule3()
                if at == 3:
                    at_trans = Sit.rule4()
                if at == 4:
                    at_trans = Sit.rule5()
                if at == 5:
                    at_trans = Sit.rule6()
                # at_trans=self.act[at]
                print(u'这是第', j, u'道工序>>', u'执行action:', at, ' ', u'将工件', at_trans[0], u'安排到机器', at_trans[1])
                Sit.scheduling(at_trans)    # 更新环境状态 Sit
                obs_t = Sit.Features()  # 并获取新的观察 obs_t。
                if j == O_num - 1:
                    done = True
                # obs = obs_t
                obs_t = np.expand_dims(obs_t, 0)
                # obs = np.expand_dims(obs, 0)
                # print(obs,obs_t)
                r_t = Sit.reward(obs[0][6], obs[0][5], obs_t[0][6], obs_t[0][5], obs[0][0], obs_t[0][0])    # 计算奖励 r_t
                self._append((obs, at, r_t, obs_t, done))   # 并将 (obs, at, r_t, obs_t, done) 经验存储到经验回放缓冲区。
                if k > self.Batch_size:     # 当累积的步数 k 大于 Batch_size 时，调用 replay 函数进行模型训练。
                    # batch_obs, batch_action, batch_reward, batch_next_obs,done= self.sample()
                    self.replay()
                Total_reward += r_t
                obs = obs_t
            # 评估性能
            # 计算每个作业的完成时间，并与截止时间 D 比较，计算总迟到时间 total_tadiness。
            total_tadiness = 0
            Job = Sit.Jobs
            End = []
            for Ji in range(len(Job)):
                End.append(max(Job[Ji].End))
                if max(Job[Ji].End) > D[Ji]:
                    total_tadiness += abs(max(Job[Ji].End) - D[Ji])
            print('<<<<<<<<<-----------------total_tardiness:', total_tadiness, '------------------->>>>>>>>>>')
            Total_tard.append(total_tadiness)
            print('<<<<<<<<<-----------------reward:', Total_reward, '------------------->>>>>>>>>>')
            TR.append(Total_reward)
            # plt.plot(K,End,color='y')
            # plt.plot(K,D,color='r')
            # plt.show()
        plt.plot(x, Total_tard)
        plt.show()
        return Total_reward


d = DQN()
d.main(J_num, M_num, O_num, J, Processing_time, D, A)

