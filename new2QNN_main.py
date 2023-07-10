#!/usr/bin/python
# -*- coding: UTF-8 -*-

from new2QNN_agents import MasterAgent, ControllerAgent, ComputeAgent
from dataset import data
import time
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import numpy as np
from new2QNN_tools import rolling_reward, check_top_ten, epsilon_greedy
import os
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class new2Qnn:
    def __init__(self,
                 name,
                 device,
                 T=10,
                 sampleBlock_num=64,
                 batch_size=64,
                 evaluate_best_model=True):
        """
        :param T: maximum layer num in a network
        """
        self.name = name
        self.device = device
        datasets = data()
        X_train, X_test, y_train, y_test = datasets.group_datasets()  # train: test = 4 : 1
        train_dataset = TensorDataset(X_train, y_train)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

        self.X_test = X_test
        self.y_test = y_test
        self.T = T   # max layer number of the network
        self.evaluate_best_model = evaluate_best_model
        self.masterAgent = MasterAgent(T=T)
        self.sampleBlock_num = sampleBlock_num  # q learning replay update
        self.batch_size = batch_size
        self.replay_memory = []
        self.sampled_network_memory = []

    def hnas(self):
        step = 0
        rt_list = []
        failure_network_num = 0
        acc_top_ten = []
        dist1, dist2, dist3, dist4, dist5, dist6 = 0, 0, 0, 0, 0, 0
        dist7, dist8, dist9, dist10 = 0, 0, 0, 0
        step1 = 400
        step2 = 1600

        while step < step1:
            eps = epsilon_greedy(step, 0.95, 0.05, 100)
            is_training = True
            S, U = self.masterAgent.sample_new_network1(epsilon=eps)
            if S in self.sampled_network_memory:
                index = self.sampled_network_memory.index(S)
                self.replay_memory.pop(index)
                self.sampled_network_memory.pop(index)
                is_training = False
            network_model = ControllerAgent(T=self.T, S=S, bs=self.batch_size).to(self.device)
            if network_model.network_built_success == False:
                failure_network_num += 1
                self.masterAgent.update_q_values(S=S, U=U, accuracy=0)
                rt_list.append(0)
                step += 1
                continue
            optimizer = optim.Adam(network_model.parameters(), lr=0.001)
            compute_agent = ComputeAgent(is_training=is_training, device=self.device, opt=optimizer,
                                         train_data=self.train_loader, test_data=(self.X_test, self.y_test),
                                         network_model=network_model, minibatch_size=self.batch_size, epoch=10,
                                         model_name="model_{0}.h5".format(S))
            accuracy = compute_agent.train_network(save_dir=self.name+'saved_models')

            self.replay_memory.append((S, U, accuracy))
            self.sampled_network_memory.append(S)

            # experience replay
            for memory in range(0, np.min((len(self.replay_memory),
                                          self.sampleBlock_num))):
                choiceIndex = np.random.choice(range(0, len(self.replay_memory)))
                s_sample, u_sample, accuracy_sample = self.replay_memory[choiceIndex]
                self.masterAgent.update_q_values(S=s_sample, U=u_sample, accuracy=accuracy_sample)
            rt_list.append(accuracy)
            step += 1

            if 0 <= accuracy < 0.5:
                dist1 = dist1 + 1
            if 0.5 <= accuracy < 0.55:
                dist2 = dist2 + 1
            if 0.55 <= accuracy < 0.6:
                dist3 = dist3 + 1
            if 0.6 <= accuracy < 0.65:
                dist4 = dist4 + 1
            if 0.65 <= accuracy < 0.7:
                dist5 = dist5 + 1
            if 0.7 <= accuracy < 0.75:
                dist6 = dist6 + 1
            if 0.75 <= accuracy < 0.8:
                dist7 = dist7 + 1
            if 0.8 <= accuracy < 0.85:
                dist8 = dist8 + 1
            if 0.85 <= accuracy < 0.9:
                dist9 = dist9 + 1
            if 0.9 <= accuracy < 1:
                dist10 = dist10 + 1
            if step == 199:
                print('first200: dist1:{0},dist2:{1},dist3:{2},dist4:{3},dist5:{4},dist6:{5},'
                      'dist7:{6},dist8:{7},dist9:{8},dist10:{9},'
                      .format(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10))

        step = 0
        dist1, dist2, dist3, dist4, dist5, dist6 = 0, 0, 0, 0, 0, 0
        dist7, dist8, dist9, dist10 = 0, 0, 0, 0
        while step < step2:
            eps = epsilon_greedy(step, 0.95, 0.05, 300)
            is_training = True
            S, U = self.masterAgent.sample_new_network2(epsilon_par=eps)
            if S in self.sampled_network_memory:
                index = self.sampled_network_memory.index(S)
                self.replay_memory.pop(index)
                self.sampled_network_memory.pop(index)
                is_training = False

            network_model = ControllerAgent(T=self.T, S=S, bs=self.batch_size).to(self.device)
            if network_model.network_built_success == False:
                failure_network_num += 1
                self.masterAgent.update_q_values(S=S, U=U, accuracy=0)
                rt_list.append(0)
                step += 1
                continue

            optimizer = optim.Adam(network_model.parameters(), lr=0.001)
            compute_agent = ComputeAgent(is_training=is_training, device=self.device, opt=optimizer,
                                         train_data=self.train_loader, test_data=(self.X_test, self.y_test),
                                         network_model=network_model, minibatch_size=self.batch_size, epoch=10,
                                         model_name="model_{0}.h5".format(S))
            accuracy = compute_agent.train_network(save_dir=self.name+'saved_models')

            self.replay_memory.append((S, U, accuracy))
            self.sampled_network_memory.append(S)

            # experience replay
            for memory in range(0, np.min((len(self.replay_memory),
                                          self.sampleBlock_num))):
                choiceIndex = np.random.choice(range(0, len(self.replay_memory)))
                s_sample, u_sample, accuracy_sample = self.replay_memory[choiceIndex]
                self.masterAgent.update_q_values(S=s_sample, U=u_sample, accuracy=accuracy_sample)
            rt_list.append(accuracy)
            step += 1

            if 0 <= accuracy < 0.5:
                dist1 = dist1 + 1
            if 0.5 <= accuracy < 0.55:
                dist2 = dist2 + 1
            if 0.55 <= accuracy < 0.6:
                dist3 = dist3 + 1
            if 0.6 <= accuracy < 0.65:
                dist4 = dist4 + 1
            if 0.65 <= accuracy < 0.7:
                dist5 = dist5 + 1
            if 0.7 <= accuracy < 0.75:
                dist6 = dist6 + 1
            if 0.75 <= accuracy < 0.8:
                dist7 = dist7 + 1
            if 0.8 <= accuracy < 0.85:
                dist8 = dist8 + 1
            if 0.85 <= accuracy < 0.9:
                dist9 = dist9 + 1
            if 0.9 <= accuracy < 1:
                dist10 = dist10 + 1
            if step == step2-201:
                dist1, dist2, dist3, dist4, dist5, dist6 = 0, 0, 0, 0, 0, 0
                dist7, dist8, dist9, dist10 = 0, 0, 0, 0
            if step == step2-1:
                print('last200: dist1:{0},dist2:{1},dist3:{2},dist4:{3},dist5:{4},dist6:{5},'
                      'dist7:{6},dist8:{7},dist9:{8},dist10:{9},'
                      .format(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10))

        print('bulited_network_num:{0}'.format(len(self.sampled_network_memory)))
        print('failure_network_num:{0}'.format(failure_network_num))
        np.save(self.name+'rt.npy', rt_list)
        rolling_reward(rt_list, self.name)
        # calculate the model accuracy distribution
        # 0-50, 50-60, 60-70, 70-80, 80-90, 90-100
        dist1, dist2, dist3, dist4, dist5, dist6 = 0, 0, 0, 0, 0, 0
        dist7, dist8, dist9, dist10 = 0, 0, 0, 0
        for eva_model_s in self.sampled_network_memory:
            network_model = ControllerAgent(T=self.T, S=eva_model_s, bs=self.batch_size).to(self.device)
            is_training = False
            optimizer = optim.Adam(network_model.parameters(), lr=0.001)
            compute_agent = ComputeAgent(is_training=is_training, device=self.device, opt=optimizer,
                                         train_data=self.train_loader, test_data=(self.X_test, self.y_test),
                                         network_model=network_model, minibatch_size=self.batch_size, epoch=8,
                                         model_name="model_{0}.h5".format(eva_model_s))
            accuracy = compute_agent.train_network(save_dir=self.name+'saved_models')
            if 0 <= accuracy < 0.5:
                dist1 = dist1 + 1
            if 0.5 <= accuracy < 0.55:
                dist2 = dist2 + 1
            if 0.55 <= accuracy < 0.6:
                dist3 = dist3 + 1
            if 0.6 <= accuracy < 0.65:
                dist4 = dist4 + 1
            if 0.65 <= accuracy < 0.7:
                dist5 = dist5 + 1
            if 0.7 <= accuracy < 0.75:
                dist6 = dist6 + 1
            if 0.75 <= accuracy < 0.8:
                dist7 = dist7 + 1
            if 0.8 <= accuracy < 0.85:
                dist8 = dist8 + 1
            if 0.85 <= accuracy < 0.9:
                dist9 = dist9 + 1
            if 0.9 <= accuracy < 1:
                dist10 = dist10 + 1
            acc_top_ten = check_top_ten(accuracy, network_model, eva_model_s, acc_top_ten)
        print('total: dist1:{0},dist2:{1},dist3:{2},dist4:{3},dist5:{4},dist6:{5},'
              'dist7:{6},dist8:{7},dist9:{8},dist10:{9},'
              .format(dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10))
        # ********************************
        # save the top ten model
        save_dir = os.path.join(os.getcwd(), self.name+'top_ten_models')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for i in range(10):
            optimizer = optim.Adam(acc_top_ten[i][2].parameters(), lr=0.001)

            compute_agent_best = ComputeAgent(is_training=True, device=self.device, opt=optimizer,
                                              train_data=self.train_loader, test_data=(self.X_test, self.y_test),
                                              network_model=acc_top_ten[i][2], minibatch_size=self.batch_size, epoch=30,
                                              model_name="model_{0}.pth".format(str(i)))

            if self.evaluate_best_model:
                best_accuracy = compute_agent_best.train_best_network(save_dir=save_dir)
                print('the final accuracy for model'+str(i)+'is %f' % best_accuracy)
                print('the description for the best model: states:{0}'.
                      format(acc_top_ten[i][0]))


if __name__ == '__main__':
    setup_seed(43)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    name = time.strftime("%Y%m%d-%H%M%S")
    new2Qnn = new2Qnn(device=device, name=name)
    new2Qnn.hnas()