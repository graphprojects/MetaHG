#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity


class SSL(nn.Module):
    def __init__(self,input_size, hidden_size1, out_size,ssl_label_dir,device):
        super(SSL, self).__init__()
        ssl_index_label = np.loadtxt(ssl_label_dir)

        self.index_1 = ssl_index_label[:, 0]
        self.index_2 = ssl_index_label[:, 1]
        self.index_label = torch.tensor(ssl_index_label[:, -1]).to(device)


        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, out_size)


    def forward(self,embed):

        embed_index_1 = embed[self.index_1,:]
        embed_index_2 = embed[self.index_2, :]

        embed_cat = torch.cat((embed_index_1,embed_index_2),dim=1)

        h1 = F.relu(self.layer1(embed_cat))
        h = self.layer2(h1)

        output = F.log_softmax(h, dim=1)
        loss = F.nll_loss(output, self.index_label.squeeze().long())

        return loss


def get_label(feature):

    print("calculating the similairty")
    sims = cosine_similarity(feature)
    print("finishing the similairty calculation")

    k = 3

    fo = open('../data/ssl_label.txt','w',encoding='utf-8')

    sort_index = np.argsort(sims, axis=1)
    for line in range(sims.shape[0]):
        for col in range(2 * k):
            if col < k:
                fo.write("\t".join([str(line),str(int(sort_index[line, col])),str(0)])+'\n')
            else:
                fo.write("\t".join([str(line), str(int(sort_index[line, -col + 1])), str(1)]) + '\n')

    print("the end of get_label")



