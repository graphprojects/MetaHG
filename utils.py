# !/usr/bin/env python
# -*- coding: UTF-8 -*-


import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
import pandas as pd

def F1(output, labels):
    output = output.argmax(1)
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, output,average='macro')
    return micro

def accuracy(output, labels):
    output = output.argmax(1)
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = accuracy_score(labels, output)
    return micro

def recall(output, labels):
    output = output.argmax(1)
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = recall_score(labels, output)
    return micro

def prec(output, labels):
    output = output.argmax(1)
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    micro = precision_score(labels, output, average='macro')
    return micro

def get_performance(logits_q, y_qry):
    return F1(logits_q, y_qry), accuracy(logits_q, y_qry), recall(logits_q, y_qry), prec(logits_q, y_qry)

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def get_metalearning_data(data_dir, train_list, test_list, negative_list):

    f = open(data_dir, 'r', encoding='utf8', errors='ignore')
    train_label_list, test_label_list = [], []
    feature_list, test_feature_list = [], []
    count_0_test, count_0_train, count_test, count_train = 0, 0, 0, 0

    label_count_dict = dict()
    for line in f.readlines():
        parts = line.replace('\n', '').replace("", "").split(' ')
        label = int(float(parts[0]))
        if label not in label_count_dict.keys():
            label_count_dict[label] = 0
        else:
            label_count_dict[label] += 1

    test_count = 0
    for item in test_list:
        test_count += label_count_dict[item]

    f = open(data_dir, 'r', encoding='utf8', errors='ignore')
    for line in f.readlines():
        parts = line.replace('\n', '').replace("", "").split(' ')
        label = int(float(parts[0]))
        feature = parts[1:]
        if label == negative_list[0]:
            if count_0_test < test_count:
                count_0_test += 1
                test_label_list.append(label)
                test_feature_list.append(feature)
            else:
                count_0_train += 1
                train_label_list.append(label)
                feature_list.append(feature)

        else:

            if label in test_list:
                count_test += 1
                test_label_list.append(label)
                test_feature_list.append(feature)

            elif label in train_list:

                count_train += 1
                train_label_list.append(label)
                feature_list.append(feature)

    df_feature = np.array(pd.DataFrame(feature_list).astype('float32'))
    df_label = np.array(pd.DataFrame(train_label_list).astype('int64'))

    test_df_feature = np.array(pd.DataFrame(test_feature_list).astype('float32'))
    test_df_label = np.array(pd.DataFrame(test_label_list).astype('int64'))

    feature_tensor = torch.from_numpy(0.1*df_feature)
    label_tensor = torch.from_numpy(df_label)

    test_feature_tensor = torch.from_numpy(0.1*test_df_feature)
    test_label_tensor = torch.from_numpy(test_df_label)


    return feature_tensor, label_tensor, test_feature_tensor, test_label_tensor, len(train_label_list), len(test_label_list)



def get_metatrain_data(features, labels, select_label, k_spt, k_qry, batch_num):
    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []
    class1_idx = []
    class2_idx = []

    select_class = [0,select_label]

    labels_local = labels.clone().detach()

    for j in range(labels_local.size()[0]):
        if (labels_local[j] == select_class[0]):

            class1_idx.append(j)
            labels_local[j] = 0

        elif (labels_local[j] == select_class[1]):
            class2_idx.append(j)
            labels_local[j] = 1

    for t in range(batch_num):
        class1_train = random.sample(class1_idx, k_spt)
        class2_train = random.sample(class2_idx, k_spt)

        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]

        class1_test = random.sample(class1_test, k_qry)
        class2_test = random.sample(class2_test, k_qry)
        num = min(len(class1_test), len(class2_test))

        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        class1_test_num = random.sample(class1_test, num)
        class2_test_num = random.sample(class2_test, num)
        test_idx = class1_test_num + class2_test_num

        random.shuffle(test_idx)

        x_spt.append(features[train_idx])
        y_spt.append(labels_local[train_idx])
        x_qry.append(features[test_idx])
        y_qry.append(labels_local[test_idx])

    return x_spt, y_spt, x_qry, y_qry

def get_metatest_data(features, labels, select_label, k_spt, k_qry, batch_num):
    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []
    class1_idx = []
    class2_idx = []

    select_class = [0,select_label]

    labels_local = labels.clone().detach()

    for j in range(labels_local.size()[0]):
        if (labels_local[j] == select_class[0]):
            class1_idx.append(j)
            labels_local[j] = 0
        elif (labels_local[j] == select_class[1]):
            class2_idx.append(j)
            labels_local[j] = 1

    for t in range(batch_num):
        class1_train = random.sample(class1_idx, k_spt)
        class2_train = random.sample(class2_idx, k_spt)

        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]

        class1_test = random.sample(class1_test, k_qry)
        class2_test = random.sample(class2_test, k_qry)
        num = min(len(class1_test), len(class2_test))

        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        class1_test_num = random.sample(class1_test, num)
        class2_test_num = random.sample(class2_test, num)
        test_idx = class1_test_num + class2_test_num

        random.shuffle(test_idx)
        x_spt.append(features[train_idx])
        y_spt.append(labels_local[train_idx])
        x_qry.append(features[test_idx])
        y_qry.append(labels_local[test_idx])

    return x_spt, y_spt, x_qry, y_qry



def get_metalearndata(logits, labels, fs_label, neg_label):
    embed = torch.cat([labels.unsqueeze(dim=1), logits], dim=1)

    meta_testdata = embed[torch.where((embed[:, 0] == fs_label))]
    meta_traindata = embed[torch.where((embed[:, 0] != fs_label) & (embed[:, 0] != neg_label))]
    meta_negative = embed[torch.where((embed[:, 0] == neg_label))]

    meta_trainfeat = torch.cat([meta_traindata[:, 1:], meta_negative[:meta_traindata.shape[0], 1:]], dim=0)
    meta_trainlabel = torch.cat([meta_traindata[:, 0], meta_negative[:meta_traindata.shape[0], 0]], dim=0).long()

    meta_testfeat = torch.cat([meta_testdata[:, 1:], meta_negative[meta_traindata.shape[0]:, 1:]], dim=0)
    meta_testlabel = torch.cat([meta_testdata[:, 0], meta_negative[meta_traindata.shape[0]:, 0]], dim=0).long()

    return meta_trainfeat, meta_trainlabel, meta_testfeat, meta_testlabel

