#!/usr/bin/python
# -*- coding: UTF-8 -*-


import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.io as scio


class data():
    def __init__(self):
        self.data_all = []
        for i in range(1,6):
            for j in [30,50,70]:
                signal = scio.loadmat('data_source/subject'+str(i)+'/'+str(j)+'%MVC.mat')
                self.data_all.append(np.array([signal['Data'][0][1], signal['Data'][0][2], signal['Data'][0][3], signal['Data'][0][4]]).squeeze(2))

    def signal_classify(self, muscle):
        no_fatigue = []
        fatigue = []
        for i in range(48):
            no_fatigue.append(muscle[:, 200*i:600+200*i])
        for i in range(48):
            fatigue.append(muscle[:, -200*i-601:-200*i-1])
        fatigue.reverse()
        return no_fatigue, fatigue

    def group_datasets(self):
        no_fatigue_list, fatigue_list = [], []
        for i in range(15):
            no_fatigue, fatigue = self.signal_classify(self.data_all[i])
            no_fatigue_list.extend(no_fatigue)
            fatigue_list.extend(fatigue)

        no_fatigue_label = np.array([0] * 720).reshape(1, 720)  # 48 * 15 = 720
        fatigue_label = np.array([1] * 720).reshape(1, 720)

        label_total = np.hstack((no_fatigue_label, fatigue_label)).ravel()
        signal_total = np.vstack((np.array(no_fatigue_list), np.array(fatigue_list)))
        signal_total = (signal_total / 1000).astype(np.float32)
        signal_total = signal_total.reshape(signal_total.shape[0], 4, 600, 1).astype('float32')
        X_train, X_test, y_train, y_test = train_test_split(signal_total, label_total, train_size=0.8, random_state=42)

        return torch.from_numpy(X_train).permute(0, 1, 3, 2).cuda(),\
               torch.from_numpy(X_test).permute(0, 1, 3, 2).cuda(),\
               torch.from_numpy(y_train).cuda(), torch.from_numpy(y_test).cuda(),

    def individual_datasets(self, data_num):
        no_fatigue_list, fatigue_list = [], []
        no_fatigue, fatigue = self.signal_classify(self.data_all[data_num])
        no_fatigue_list.extend(no_fatigue)
        fatigue_list.extend(fatigue)

        no_fatigue_label = np.array([0] * 48).reshape(1, 48)
        fatigue_label = np.array([1] * 48).reshape(1, 48)

        label_total = np.hstack((no_fatigue_label, fatigue_label)).ravel()
        signal_total = np.vstack((np.array(no_fatigue_list), np.array(fatigue_list)))

        signal_total = (signal_total / 1000).astype(np.float32)
        signal_total = signal_total.reshape(signal_total.shape[0], 4, 600, 1).astype('float32')

        X_train, X_test, y_train, y_test = train_test_split(signal_total, label_total, train_size=0.8, random_state=42)

        return torch.from_numpy(X_train).permute(0, 1, 3, 2).cuda(),\
               torch.from_numpy(X_test).permute(0, 1, 3, 2).cuda(),\
               torch.from_numpy(y_train).cuda(), torch.from_numpy(y_test).cuda(),

# if __name__ == '__main__':
    # datasets = data()
    # a,b,c,d = datasets.group_datasets()
    # pass


