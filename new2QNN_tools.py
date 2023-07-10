#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random
import matplotlib

def draw_accuracy_each_iter(accuracyList, saveDir=None):
    """
    draw the accuracy of each iteration
    :param accuracyList:
    :param saveDir:
    :return:
    """

    assert isinstance(accuracyList, list)
    accuracyList.insert(0, 0.1)  # the accuracy in the first iter is random guess accuracy
    x = list(range(0, len(accuracyList)))
    plt.plot(x, accuracyList)
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.title("Q-learning process")
    if saveDir is not None:
        plt.savefig(os.path.join(saveDir, 'Q_learning_Iterations.png'))
    plt.show()


def dict_argmax(d):
    """
    find the max integer element's corresponding key value in the dict
    :param d: the dic object on which to perform argmax operation
    :return: the max integer element's corresponding key
    """
    assert isinstance(d, dict)
    max_value = 0
    max_key = list(d.keys())[0]
    for key in d.keys():
        if d[key] > max_value:
            max_value = d[key]
            max_key = key
    if max_value == 0:  # still 0, random chose
        max_key = np.random.choice(list(d.keys()))
    return max_key

def random_u(u):
    if int(u.split('_')[0]) == 1:
        bool_tmp = int(u.split('_')[-1])
        k_s_b = random.choice((6, 51, 101))
        k_n = random.choice((6, 12, 18))
        u = '1_' + '{0},{1}_{2}_{3},{4}_{5}_'.format(1, k_s_b, k_n, 1, 1, 0) + str(bool_tmp)
    elif int(u.split('_')[0]) == 2:
        bool_tmp = int(u.split('_')[-1])
        k_s_b = random.choice((2, 4, 8, 12, 16, 20))
        u = '2_' + '{0},{1}_{2}_{3},{4}_{5}_'.format(1, k_s_b, 0, 1, k_s_b, 0) + str(bool_tmp)
    elif int(u.split('_')[0]) == 3:
        h = random.choice((10, 16, 32, 64, 128, 256))
        u = '3_' + '{0},{1}_{2}_{3},{4}_{5}_'.format(0, 0, 0, 0, 0, h) + '0'
    return u


def par_argmax(u, key_list):
    same_list = []
    if int(u.split('_')[0]) == 1:
        if int(u.split('_')[-1]) == 0:
            for i in range(len(key_list)):
                if int(key_list[i].split('_')[0])==1 and int(key_list[i].split('_')[-1])==0:
                    same_list.append(key_list[i])
        if int(u.split('_')[-1]) == 1:
            for i in range(len(key_list)):
                if int(key_list[i].split('_')[0])==1 and int(key_list[i].split('_')[-1])==1:
                    same_list.append(key_list[i])
    if int(u.split('_')[0]) == 2:
        if int(u.split('_')[-1]) == 0:
            for i in range(len(key_list)):
                if int(key_list[i].split('_')[0])==2 and int(key_list[i].split('_')[-1])==0:
                    same_list.append(key_list[i])
        if int(u.split('_')[-1]) == 1:
            for i in range(len(key_list)):
                if int(key_list[i].split('_')[0])==2 and int(key_list[i].split('_')[-1])==1:
                    same_list.append(key_list[i])
    if int(u.split('_')[0]) == 3:
        for i in range(len(key_list)):
            if int(key_list[i].split('_')[0])==3:
                same_list.append(key_list[i])
    if int(u.split('_')[0]) == 4:
        for i in range(len(key_list)):
            if int(key_list[i].split('_')[0])==4:
                same_list.append(key_list[i])
    u = random.choice(same_list)
    return u

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr
    lr_new = 0
    if epoch in range(1, 6):
        lr_new = lr
    if epoch in range(6, 11):
        lr_new = lr * 0.8
    if epoch in range(11, 16):
        lr_new = lr * 0.8 * 0.8
    if epoch in range(16, 21):
        lr_new = lr * 0.8 * 0.8 * 0.8
    if epoch in range(21, 26):
        lr_new = lr * 0.8 * 0.8 * 0.8 * 0.8
    if epoch in range(26, 31):
        lr_new = lr * 0.8 * 0.8 * 0.8 * 0.8 * 0.8

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new


def rolling_reward(rt_list, name=None):
    matplotlib.use('Agg')
    cumsum_vec = np.nancumsum(np.insert(rt_list, 0, 0))
    ma_vec = (cumsum_vec[50:] - cumsum_vec[:-50]) / 50
    plt.plot(ma_vec)
    plt.savefig(name+'.png')


def check_top_ten(acc, network_model, S, acc_top_ten=None):
    def takeacc(elem):
        return elem[1]

    if acc_top_ten is None:
        acc_top_ten = []

    pack = (S, acc, network_model)
    if len(acc_top_ten) <= 10:
        acc_top_ten.append(pack)
    else:
        acc_top_ten.append(pack)
        acc_top_ten.sort(key=takeacc, reverse=True)
        acc_top_ten.pop()
    return acc_top_ten

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    return total_num


def epsilon_greedy(step, eps_start=0.95, eps_end=0.05, eps_decay=300):
    eps = eps_end + (eps_start - eps_end) * \
              math.exp(-1. * step / eps_decay)
    return eps