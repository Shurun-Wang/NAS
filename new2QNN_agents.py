#!/usr/bin/python
# -*- coding: UTF-8 -*-
from new2QNN_tools import dict_argmax, random_u, par_argmax, adjust_learning_rate, get_parameter_number
import numpy as np
import copy
import torch
import collections
import math
import os
import random



class MasterAgent:
    """
    each operation is a dic item like:
    'Index_Type_Kernel size_Pred1_Pred2'
     Index_Type_ks_kn_s_h'
    """
    def __init__(self, T=10, q_lr=0.01, gamma=1.0):  # T is the total time steps
        self.T = T
        self.q_lr = q_lr  # the learning rate for Bellman equation
        self.Q_table = self.initiate_q_table()
        self.gamma = gamma

    def initiate_q_table(self):
        q_table = {}
        """
        use dict to store the Q_table
        Type == 1 : conv
        Type == 2 : Average_pooling
        Type == 3 : dense
        Type == 4 : Terminal
        """
        q_table['input_0'] = {}
        for index in range(1, self.T + 1):  # the right side is not included
            # initiate states
            for TYPE in range(1, 5):  # there are 4 kinds of Type
                if TYPE == 1:  # convolution
                    for k_s_b in (6, 51, 101):
                        for k_n in (6, 12, 18):
                            q_table["{0}_{1}_{2},{3}_{4}_{5},{6}_{7}_{8}".format(
                                index, TYPE, 1, k_s_b, k_n, 1, 1, 0, 0)] = {}
                            q_table["{0}_{1}_{2},{3}_{4}_{5},{6}_{7}_{8}".format(
                                index, TYPE, 1, k_s_b, k_n, 1, 1, 0, 1)] = {}
                elif TYPE == 2:  # Average_pooling
                    for k_s_b in (2, 4, 8, 12, 16, 20):
                        q_table["{0}_{1}_{2},{3}_{4}_{5},{6}_{7}_{8}".format(
                                index, TYPE, 1, k_s_b, 0, 1, k_s_b, 0, 0)] = {}
                        q_table["{0}_{1}_{2},{3}_{4}_{5},{6}_{7}_{8}".format(
                                index, TYPE, 1, k_s_b, 0, 1, k_s_b, 0, 1)] = {}
                elif TYPE == 3:  # Dense
                    for h in (10, 16, 32, 64, 128, 256):
                        q_table["{0}_{1}_{2},{3}_{4}_{5},{6}_{7}_{8}".format(
                                index, TYPE, 0, 0, 0, 0, 0, h, 0)] = {}
                elif TYPE == 4:  # Terminal
                    # if index == 1:  # output = input, meaningless
                    #     continue
                    q_table["{0}_{1}_{2},{3}_{4}_{5},{6}_{7}_{8}".format(
                        index, TYPE, 0, 0, 0, 0, 0, 2, 0)] = {}

        # initiate actions
        # original reward is 0.5 as random guessing accuracy
        for state in q_table.keys():
            for TYPE in range(1, 5):  # there are 4 kinds of Type
                if TYPE == 1:  # convolution
                    if int(state.split('_')[1]) in range(3, 5):  # only allow input, conv, pool
                        continue
                    for k_s_b in (6, 51, 101):
                        for k_n in (6, 12, 18):
                            q_table[state]["{0}_{1},{2}_{3}_{4},{5}_{6}_{7}".format(
                                TYPE, 1, k_s_b, k_n, 1, 1, 0, 0)] = 0.5
                            q_table[state]["{0}_{1},{2}_{3}_{4},{5}_{6}_{7}".format(
                                TYPE, 1, k_s_b, k_n, 1, 1, 0, 1)] = 0.5
                elif TYPE == 2:  # pool
                    if int(state.split('_')[1]) in range(2, 5):  # only allow input, conv
                        continue
                    for k_s_b in (2, 4, 8, 12, 16, 20):
                        q_table[state]["{0}_{1},{2}_{3}_{4},{5}_{6}_{7}".format(
                                TYPE, 1, k_s_b, 0, 1, k_s_b, 0, 0)] = 0.5
                        q_table[state]["{0}_{1},{2}_{3}_{4},{5}_{6}_{7}".format(
                                TYPE, 1, k_s_b, 0, 1, k_s_b, 0, 1)] = 0.5
                elif TYPE == 3:  # Dense
                    if int(state.split('_')[1]) == 4:
                        continue
                    for h in (10, 16, 32, 64, 128, 256):
                        q_table[state]["{0}_{1},{2}_{3}_{4},{5}_{6}_{7}".format(
                            TYPE, 0, 0, 0, 0, 0, h, 0)] = 0.5

                elif TYPE == 4:  # Terminal
                    if state == 'input_0':  # output = input, meaningless
                        q_table[state]["{0}_{1},{2}_{3}_{4},{5}_{6}_{7}".format(
                            TYPE, 0, 0, 0, 0, 0, 2, 0)] = -np.inf
                        continue
                    if int(state.split('_')[1]) == 4:
                        q_table[state]["{0}_{1},{2}_{3}_{4},{5}_{6}_{7}".format(
                            TYPE, 0, 0, 0, 0, 0, 2, 0)] = -np.inf
                        continue
                    q_table[state]["{0}_{1},{2}_{3}_{4},{5}_{6}_{7}".format(
                        TYPE, 0, 0, 0, 0, 0, 2, 0)] = 0.5
        return q_table

    def sample_new_network1(self, epsilon):
        """
        based on the algorithm posted in the metaQnn paper
        """
        # initialize S->state sequence;U->action sequence
        S = ['input_0']
        U = []
        index = 1
        # *****need to restrict conv layer number to 3, the same as avg_pooling and dense*****
        # not the terminate layer and not surpass the max index(can be infinite)
        conv_count = 0
        pool_count = 0
        dense_count = 0
        while index <= self.T:
            a = np.random.uniform(0, 1)
            if a > epsilon:
                Q = copy.deepcopy(self.Q_table[S[-1]])  # exploitation   # under a state, choose the action of max value
                dict = Q
                dict_tmp = copy.deepcopy(dict)
                if conv_count >= 3 and pool_count < 3 and dense_count == 0:
                    for i in range(len(dict_tmp)):
                        if index > 6:
                            break
                        if int(list(dict_tmp.keys())[i].split('_')[0]) == 1:
                            dict.pop(list(dict_tmp.keys())[i])
                    u = dict_argmax(dict)
                    u = random_u(u)
                elif conv_count < 3 and pool_count >= 3 and dense_count == 0:
                    for i in range(len(dict_tmp)):
                        if index > 6:
                            break
                        if int(list(dict_tmp.keys())[i].split('_')[0]) == 2:
                            dict.pop(list(dict_tmp.keys())[i])
                    u = dict_argmax(dict)
                    u = random_u(u)
                elif conv_count >= 3 and pool_count >= 3 and dense_count == 0:
                    for i in range(len(dict_tmp)):
                        if int(list(dict_tmp.keys())[i].split('_')[0]) == 1\
                                or int(list(dict_tmp.keys())[i].split('_')[0]) == 2:
                            dict.pop(list(dict_tmp.keys())[i])
                    u = dict_argmax(dict)
                    u = random_u(u)
                elif dense_count >= 1:
                    for i in range(len(dict_tmp)):
                        if int(list(dict_tmp.keys())[i].split('_')[0]) == 3:
                            dict.pop(list(dict_tmp.keys())[i])
                    u = dict_argmax(dict)
                    u = random_u(u)
                else:
                    u = dict_argmax(dict)
                    u = random_u(u)

                new_state = str(index) + '_' + u
                if int(u.split('_')[0]) == 1:
                    conv_count += 1
                if int(u.split('_')[0]) == 2:
                    pool_count += 1
                if int(u.split('_')[0]) == 3:
                    dense_count += 1
            else:  # exploration
                key_list = list(self.Q_table[S[-1]].keys())
                key_tmp = copy.deepcopy(key_list)
                if conv_count >= 3 and pool_count < 3 and dense_count == 0:
                    for i in range(len(key_tmp)):
                        if index > 6:
                            break
                        if int(key_tmp[i].split('_')[0]) == 1:
                            key_list.remove(key_tmp[i])
                    u = np.random.choice(key_list)

                elif conv_count < 3 and pool_count >= 3 and dense_count == 0:
                    for i in range(len(key_tmp)):
                        if index > 6:
                            break
                        if int(key_tmp[i].split('_')[0]) == 2:
                            key_list.remove(key_tmp[i])
                    u = np.random.choice(key_list)

                elif conv_count >= 3 and pool_count >= 3 and dense_count == 0:
                    for i in range(len(key_tmp)):
                        if int(key_tmp[i].split('_')[0]) == 1 or int(key_tmp[i].split('_')[0]) == 2:
                            key_list.remove(key_tmp[i])
                    u = np.random.choice(key_list)

                elif dense_count >= 1:
                    for i in range(len(key_tmp)):
                        if int(key_tmp[i].split('_')[0]) == 3:
                            key_list.remove(key_tmp[i])
                    u = np.random.choice(key_list)

                else:
                    u = np.random.choice(key_list)

                new_state = str(index) + '_' + u

                if int(u.split('_')[0]) == 1:
                    conv_count += 1
                if int(u.split('_')[0]) == 2:
                    pool_count += 1
                if int(u.split('_')[0]) == 3:
                    dense_count += 1

            U.append(u)
            if u != '4_0,0_0_0,0_2_0':  # u != terminate
                S.append(new_state)
            else:
                S.append('{0}_4_0,0_0_0,0_2_0'.format(index))
                return S, U
            index += 1
        U.append('4_0,0_0_0,0_2_0')
        return S, U

    def sample_new_network2(self, epsilon_par):
        """
        based on the algorithm posted in the metaQnn paper
        """
        # initialize S->state sequence;U->action sequence
        S = ['input_0']
        U = []
        index = 1
        # *****need to restrict conv layer number to 3, the same as avg_pooling and dense*****
        # not the terminate layer and not surpass the max index(can be infinite)
        conv_count = 0
        pool_count = 0
        dense_count = 0
        while index <= self.T:
            a = np.random.uniform(0, 1)
            b = np.random.uniform(0, 1)
            u = '_'
            new_state = '_'
            if b < 0.05:
                # layer explore, par explore
                if a < epsilon_par:
                    key_list = list(self.Q_table[S[-1]].keys())
                    key_tmp = copy.deepcopy(key_list)
                    if conv_count >= 3 and pool_count < 3 and dense_count == 0:
                        for i in range(len(key_tmp)):
                            if index > 6:
                                break
                            if int(key_tmp[i].split('_')[0]) == 1:
                                key_list.remove(key_tmp[i])
                        u = np.random.choice(key_list)

                    elif conv_count < 3 and pool_count >= 3 and dense_count == 0:
                        for i in range(len(key_tmp)):
                            if index > 6:
                                break
                            if int(key_tmp[i].split('_')[0]) == 2:
                                key_list.remove(key_tmp[i])
                        u = np.random.choice(key_list)

                    elif conv_count >= 3 and pool_count >= 3 and dense_count == 0:
                        for i in range(len(key_tmp)):
                            if int(key_tmp[i].split('_')[0]) == 1 or int(key_tmp[i].split('_')[0]) == 2:
                                key_list.remove(key_tmp[i])
                        u = np.random.choice(key_list)

                    elif dense_count >= 1:
                        for i in range(len(key_tmp)):
                            if int(key_tmp[i].split('_')[0]) == 3:
                                key_list.remove(key_tmp[i])
                        u = np.random.choice(key_list)

                    else:
                        u = np.random.choice(key_list)

                    new_state = str(index) + '_' + u

                    if int(u.split('_')[0]) == 1:
                        conv_count += 1
                    if int(u.split('_')[0]) == 2:
                        pool_count += 1
                    if int(u.split('_')[0]) == 3:
                        dense_count += 1
                # layer explore, par exploit
                elif a >= epsilon_par:
                    key_list = list(self.Q_table[S[-1]].keys())
                    key_tmp = copy.deepcopy(key_list)
                    if conv_count >= 3 and pool_count < 3 and dense_count == 0:
                        for i in range(len(key_tmp)):
                            if index > 6:
                                break
                            if int(key_tmp[i].split('_')[0]) == 1:
                                key_list.remove(key_tmp[i])
                        u = np.random.choice(key_list)
                        # in key_list, random find u
                        # u is the random layer, we need to split it
                        u = par_argmax(u, key_list)

                    elif conv_count < 3 and pool_count >= 3 and dense_count == 0:
                        for i in range(len(key_tmp)):
                            if index > 6:
                                break
                            if int(key_tmp[i].split('_')[0]) == 2:
                                key_list.remove(key_tmp[i])
                        u = np.random.choice(key_list)
                        u = par_argmax(u, key_list)

                    elif conv_count >= 3 and pool_count >= 3 and dense_count == 0:
                        for i in range(len(key_tmp)):
                            if int(key_tmp[i].split('_')[0]) == 1 or int(key_tmp[i].split('_')[0]) == 2:
                                key_list.remove(key_tmp[i])
                        u = np.random.choice(key_list)
                        u = par_argmax(u, key_list)

                    elif dense_count >= 1:
                        for i in range(len(key_tmp)):
                            if int(key_tmp[i].split('_')[0]) == 3:
                                key_list.remove(key_tmp[i])
                        u = np.random.choice(key_list)
                        u = par_argmax(u, key_list)

                    else:
                        u = np.random.choice(key_list)
                        u = par_argmax(u, key_list)

                    new_state = str(index) + '_' + u

                    if int(u.split('_')[0]) == 1:
                        conv_count += 1
                    if int(u.split('_')[0]) == 2:
                        pool_count += 1
                    if int(u.split('_')[0]) == 3:
                        dense_count += 1

            elif b >= 0.05:
                # layer exploit, par explore
                if a < epsilon_par:
                    Q = copy.deepcopy(
                        self.Q_table[S[-1]])  # exploitation   # under a state, choose the action of max value
                    dict = Q
                    dict_tmp = copy.deepcopy(dict)
                    if conv_count >= 3 and pool_count < 3 and dense_count == 0:
                        for i in range(len(dict_tmp)):
                            if index > 6:
                                break
                            if int(list(dict_tmp.keys())[i].split('_')[0]) == 1:
                                dict.pop(list(dict_tmp.keys())[i])
                        u = dict_argmax(dict)
                        u = random_u(u)
                    elif conv_count < 3 and pool_count >= 3 and dense_count == 0:
                        for i in range(len(dict_tmp)):
                            if index > 6:
                                break
                            if int(list(dict_tmp.keys())[i].split('_')[0]) == 2:
                                dict.pop(list(dict_tmp.keys())[i])
                        u = dict_argmax(dict)
                        u = random_u(u)
                    elif conv_count >= 3 and pool_count >= 3 and dense_count == 0:
                        for i in range(len(dict_tmp)):
                            if int(list(dict_tmp.keys())[i].split('_')[0]) == 1 \
                                    or int(list(dict_tmp.keys())[i].split('_')[0]) == 2:
                                dict.pop(list(dict_tmp.keys())[i])
                        u = dict_argmax(dict)
                        u = random_u(u)
                    elif dense_count >= 1:
                        for i in range(len(dict_tmp)):
                            if int(list(dict_tmp.keys())[i].split('_')[0]) == 3:
                                dict.pop(list(dict_tmp.keys())[i])
                        u = dict_argmax(dict)
                        u = random_u(u)
                    else:
                        u = dict_argmax(dict)
                        u = random_u(u)

                    new_state = str(index) + '_' + u
                    if int(u.split('_')[0]) == 1:
                        conv_count += 1
                    if int(u.split('_')[0]) == 2:
                        pool_count += 1
                    if int(u.split('_')[0]) == 3:
                        dense_count += 1
                # layer exploit, par exploit
                elif a >= epsilon_par:
                    Q = copy.deepcopy(self.Q_table[S[-1]])  # exploitation   # under a state, choose the action of max value
                    dict = Q
                    dict_tmp = copy.deepcopy(dict)
                    if conv_count >= 3 and pool_count < 3 and dense_count == 0:
                        for i in range(len(dict_tmp)):
                            if index > 6:
                                break
                            if int(list(dict_tmp.keys())[i].split('_')[0]) == 1:
                                dict.pop(list(dict_tmp.keys())[i])
                        u = dict_argmax(dict)
                    elif conv_count < 3 and pool_count >= 3 and dense_count == 0:
                        for i in range(len(dict_tmp)):
                            if index > 6:
                                break
                            if int(list(dict_tmp.keys())[i].split('_')[0]) == 2:
                                dict.pop(list(dict_tmp.keys())[i])
                        u = dict_argmax(dict)
                    elif conv_count >= 3 and pool_count >= 3 and dense_count == 0:
                        for i in range(len(dict_tmp)):
                            if int(list(dict_tmp.keys())[i].split('_')[0]) == 1\
                                    or int(list(dict_tmp.keys())[i].split('_')[0]) == 2:
                                dict.pop(list(dict_tmp.keys())[i])
                        u = dict_argmax(dict)
                    elif dense_count >= 1:
                        for i in range(len(dict_tmp)):
                            if int(list(dict_tmp.keys())[i].split('_')[0]) == 3:
                                dict.pop(list(dict_tmp.keys())[i])
                        u = dict_argmax(dict)
                    else:
                        u = dict_argmax(dict)

                    new_state = str(index) + '_' + u
                    if int(u.split('_')[0]) == 1:
                        conv_count += 1
                    if int(u.split('_')[0]) == 2:
                        pool_count += 1
                    if int(u.split('_')[0]) == 3:
                        dense_count += 1

            U.append(u)
            if u != '4_0,0_0_0,0_2_0':  # u != terminate
                S.append(new_state)
            else:
                S.append('{0}_4_0,0_0_0,0_2_0'.format(index))
                return S, U
            index += 1
        U.append('4_0,0_0_0,0_2_0')
        return S, U

    def update_q_values(self, S, U, accuracy, gamma=1.0):
        """
        based on the algorithm posted in the metaQnn paper
        :param gamma: the discount factor which measures the importance of future rewards
        :param S: state sequence
        :param U: action sequence
        :param accuracy: the model accuracy on the validation set
        :return: None
        """
        rt = accuracy
        # find the max action reward for the next step
        i = 0
        j = len(S) - 1
        U.append('4_0,0_0_0,0_2_0')
        while i < j:
            max_action_reward = 0
            for action in list(self.Q_table[S[i + 1]].keys()):
                if self.Q_table[S[i + 1]][action] > max_action_reward:
                    max_action_reward = self.Q_table[S[i + 1]][action]
            self.Q_table[S[i]][U[i]] = (1 - self.q_lr) * self.Q_table[S[i]][U[i]] + \
                                           self.q_lr * (rt + gamma * max_action_reward)
            i = i + 1

        return rt


class mypool(torch.nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(mypool, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(output_features, output_features), requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(output_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    def forward(self, input):
        input = input.permute([0,2,3,1])
        pred = torch.matmul(input, self.weight) + self.bias
        return pred.permute([0,3,1,2])


class ControllerAgent(torch.nn.Module):
    def __init__(self, T, S, bs):
        super(ControllerAgent, self).__init__()

        self.network_built_success = True
        self.T = T
        self.batch_size = bs
        self.channel = 4
        self.signal_width = 600  # shape (3186, 4, 1, 600)
        in_channel = self.channel
        Width_out = self.signal_width
        Height_out = 1
        feature_size = 4 * 600

        # load the state sequence and transfer into real network
        feature_extractor_list = []
        dense_extractor_list = []
        classifier_list = []
        # after each layer, we calculate the output channel and output size

        for s in S[1:]:
            Index, Type, KernelSize, KernelNumber, Stride, HiddenNumber, bool_tmp = s.split('_')
            ks_a, ks_b = KernelSize.split(',')
            st_a, st_b = Stride.split(',')
            ks_a, ks_b, st_a, st_b = int(ks_a), int(ks_b), int(st_a), int(st_b)
            KernelNumber, HiddenNumber = int(KernelNumber), int(HiddenNumber)
            bool_tmp = int(bool_tmp)
            if Type == "1":
                if Width_out < ks_b:
                    self.network_built_success = False
                    classifier_list.append(('{0}_terminal_2'.format(Index, HiddenNumber),
                                            torch.nn.Linear(in_features=feature_size,
                                                            out_features=2)))
                    # print('conv dim error')
                    break
                if bool_tmp == 0:
                    feature_extractor_list.append(('{0}_conv'.format(Index),
                                                   torch.nn.Conv2d(in_channels=in_channel, out_channels=KernelNumber,
                                                                   kernel_size=(ks_a, ks_b), stride=(st_a, st_b))))

                    feature_extractor_list.append(('{0}_relu'.format(Index), torch.nn.ReLU()))

                elif bool_tmp == 1:
                    feature_extractor_list.append(('{0}_conv'.format(Index),
                                                   torch.nn.Conv2d(in_channels=in_channel, out_channels=KernelNumber,
                                                                   kernel_size=(ks_a, ks_b), stride=(st_a, st_b))))
                    feature_extractor_list.append(('{0}_bn'.format(Index),
                                                   torch.nn.BatchNorm2d(num_features=KernelNumber)))

                    feature_extractor_list.append(('{0}_relu'.format(Index), torch.nn.ReLU()))

                in_channel = KernelNumber
                Height_out = Height_out
                Width_out = (Width_out-ks_b)//st_b + 1
                feature_size = in_channel * Height_out * Width_out

            elif Type == "2":
                if Width_out < ks_b:
                    self.network_built_success = False
                    classifier_list.append(('{0}_terminal_2'.format(Index, HiddenNumber),
                                            torch.nn.Linear(in_features=feature_size,
                                                            out_features=2)))
                    break
                if bool_tmp == 0:
                    feature_extractor_list.append(('{0}_avg_pool_{1}X{1}'.format(Index, ks_b),
                                                    torch.nn.AvgPool2d(kernel_size=(1, ks_b),
                                                                  stride=(1, st_b), ceil_mode=True)))
                    in_channel = in_channel
                    Height_out = Height_out
                    Width_out = int(math.ceil((Width_out - ks_b) / st_b) + 1)
                    feature_size = in_channel * Height_out * Width_out

                elif bool_tmp == 1:
                    feature_extractor_list.append(('{0}_avg_pool_{1}X{1}'.format(Index, ks_b),
                                                    torch.nn.AvgPool2d(kernel_size=(1, ks_b),
                                                                  stride=(1, st_b), ceil_mode=True)))
                    in_channel = in_channel
                    Height_out = Height_out
                    Width_out = int(math.ceil((Width_out - ks_b) / st_b) + 1)
                    feature_size = in_channel * Height_out * Width_out
                    feature_extractor_list.append(('{0}_par_{1}X{1}'.format(Index, ks_b),
                                                   mypool((1, Width_out), in_channel)))

            elif Type == "3":
                dense_extractor_list.append(('{0}_dense_{1}'.format(Index, HiddenNumber),
                                               torch.nn.Linear(in_features=feature_size,
                                                               out_features=HiddenNumber)))
                dense_extractor_list.append(('{0}_relu'.format(Index),
                                               torch.nn.ReLU()))
                dense_extractor_list.append(('{0}_dropout'.format(Index),
                                               torch.nn.Dropout(0.5)))
                feature_size = HiddenNumber

            elif Type == "4":
                classifier_list.append(('{0}_terminal_2'.format(Index, HiddenNumber),
                                               torch.nn.Linear(in_features=feature_size,
                                                               out_features=2)))
        self.feature_extractor_list = torch.nn.Sequential(collections.OrderedDict(feature_extractor_list))
        self.dense_extractor_list = torch.nn.Sequential(collections.OrderedDict(dense_extractor_list))
        self.classifier_list = torch.nn.Sequential(collections.OrderedDict(classifier_list))

    def forward(self, x):
        x = self.feature_extractor_list(x)
        x = x.reshape(x.size(0), -1)
        x = self.dense_extractor_list(x)
        x = self.classifier_list(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x


class ComputeAgent:
    def __init__(self, is_training, device, opt, train_data, test_data, network_model, model_name, epoch, minibatch_size=256):
        self.is_training = is_training
        self.device = device
        self.network_model = network_model
        self.epoch = epoch
        self.minibatch_size = minibatch_size
        self.train_load = train_data
        (self.X_test, self.y_test) = test_data
        self.model_name = model_name
        self.opt = opt

    def train_network(self, save_dir):
        if self.is_training == True:
            test_data, test_target = self.X_test, self.y_test
            test_data, test_target = test_data.to(self.device), test_target.to(self.device)
            for ep in range(1, self.epoch+1):
                self.network_model.train()
                for batch_idx, (data, target) in enumerate(self.train_load):
                    data, target = data.to(self.device), target.to(self.device)
                    self.opt.zero_grad()
                    output = self.network_model(data)
                    loss = torch.nn.functional.nll_loss(output, target.long())
                    loss.backward()
                    self.opt.step()

                self.network_model.eval()
                if ep in range(self.epoch-2, self.epoch+1):
                    # evaluate the trained model and get it's accuracy
                    with torch.no_grad():
                        output = self.network_model(test_data)
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct = pred.eq(test_target.view_as(pred)).sum().item()
                        correct_rate = correct / test_data.size(0)
            save_dir = os.path.join(os.getcwd(), save_dir)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, self.model_name)
            torch.save(self.network_model, model_path)

        else:
            save_dir = os.path.join(os.getcwd(), save_dir)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, self.model_name)
            self.network_model = torch.load(model_path)
            # evaluate the trained model and get it's accuracy
            self.network_model.eval()
            with torch.no_grad():
                test_data, test_target = self.X_test, self.y_test
                test_data, test_target = test_data.to(self.device), test_target.to(self.device)
                test_index = random.sample(range(0, test_data.size(0)), test_data.size(0) // 5)
                test_index = torch.tensor(test_index).cuda().to(device=self.device)
                test_data = torch.index_select(test_data, 0, index=test_index)
                test_target = torch.index_select(test_target, 0, index=test_index)
                output = self.network_model(test_data)
                test_loss = torch.nn.functional.nll_loss(output, test_target.long(),
                                                         reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(test_target.view_as(pred)).sum().item()
                correct_rate = correct / test_data.size(0)
        return correct_rate

    def train_best_network(self, save_dir):
        lr_init = self.opt.param_groups[0]['lr']
        # we need to change lr to train it.
        self.network_model.train()
        for epoch in range(1, self.epoch+1):
            adjust_learning_rate(self.opt, epoch, lr_init)
            for batch_idx, (data, target) in enumerate(self.train_load):
                data, target = data.to(self.device), target.to(self.device)
                self.opt.zero_grad()
                output = self.network_model(data)
                loss = torch.nn.functional.nll_loss(output, target.long())
                loss.backward()
                self.opt.step()
                # if batch_idx % 10 == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(self.train_load.dataset),
                #                100. * batch_idx / len(self.train_load), loss.item()))
        # save the top ten trained models
        save_dir = os.path.join(os.getcwd(), save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, self.model_name)
        torch.save(self.network_model.state_dict(), model_path, _use_new_zipfile_serialization=False)
        print('Saved best model at %s' % model_path)
        print('net parameter number:{0}'.format(get_parameter_number(self.network_model)))
        # evaluate the trained model and get it's accuracy
        self.network_model.eval()
        correct_rate = 0
        with torch.no_grad():
            test_data, test_target = self.X_test.to(self.device), self.y_test.to(self.device)
            output = self.network_model(test_data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(test_target.view_as(pred)).sum().item()
            correct_rate += correct / test_data.size(0)
        print('old: test correct:{}'.format(correct_rate))
        return correct_rate
