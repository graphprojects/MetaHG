#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import dgl
import torch
import torch.nn as nn



def build_relation(postid_dir,userid_dir,keywordid_dir,usertype_dir,relation_dir):

    fp = open(postid_dir,'r')
    post_id_dict = json.load(fp)

    fu = open(userid_dir, 'r')
    name_id_dict = json.load(fu)


    fk = open(keywordid_dir, 'r')
    keyword_id_dict = json.load(fk)


    fc = open(usertype_dir, 'r')
    userid_cate_dict = json.load(fc)


    similar_relation,follow_relation,tagger_relation,have_relation,mention_relation,reply_relation,include_relation,tag_relation,profile_relation = [],[],[],[],[],[],[],[],[]

    fr = open(relation_dir, 'r', encoding='utf8')
    for line in fr.readlines():
        parts = line.replace('\n','').split('\t')
        relation_type = parts[2]

        try:
            if relation_type == 'follow':
                follow_relation.append((name_id_dict[parts[0]],name_id_dict[parts[1]]))
            elif relation_type == 'similar':
                similar_relation.append((name_id_dict[parts[0]],post_id_dict[parts[1]]))
            elif relation_type == 'tagger':
                tagger_relation.append((name_id_dict[parts[0]],name_id_dict[parts[1]]))
            elif relation_type == 'have':
                have_relation.append((name_id_dict[parts[0]],post_id_dict[parts[1]]))
            elif relation_type == 'mention':
                mention_relation.append((name_id_dict[parts[0]], post_id_dict[parts[1]]))
            elif relation_type == 'reply':
                reply_relation.append((name_id_dict[parts[0]], post_id_dict[parts[1]]))
            elif relation_type == 'include':
                include_relation.append((post_id_dict[parts[0]], keyword_id_dict[parts[1]]))
            elif relation_type == 'tag':
                tag_relation.append((post_id_dict[parts[0]], keyword_id_dict[parts[1]]))
            elif relation_type == 'profile':
                profile_relation.append((name_id_dict[parts[0]], keyword_id_dict[parts[1]]))
        except:
            continue
    relation_list = [follow_relation, similar_relation, tagger_relation, have_relation, mention_relation, reply_relation, include_relation, tag_relation, profile_relation]
    user_label = list(userid_cate_dict.values())

    return relation_list,user_label


def build_graph(relation_list,user_feature,post_feature,keyword_feature):
    g = dgl.heterograph({
        ('user', 'follow', 'user'): relation_list[0],
        ('user', 'similar', 'post'): relation_list[1],
        ('user', 'tagger', 'user'):relation_list[2],
    ('user', 'have', 'post'):relation_list[3],
    ('user', 'mention', 'post'):relation_list[4],
    ('user', 'reply', 'post'):relation_list[5],
    ('post', 'include', 'keyword'):relation_list[6],
    ('post', 'tag', 'keyword'):relation_list[7],
    ('user', 'profile', 'keyword'):relation_list[8],
    })


    g.nodes['user'].data['h'] = user_feature
    g.nodes['post'].data['h'] = post_feature
    g.nodes['keyword'].data['h'] = keyword_feature

    print("node statistics:")
    print("# of posts:")
    print(g.number_of_nodes('post'))
    print("# of user:")
    print(g.number_of_nodes('user'))
    print("# of keyword:")
    print(g.number_of_nodes('keyword'))
    print("edge statistics:")
    print("# of follow relation:")
    print(g.number_of_edges(('user', 'follow', 'user')))
    print("# of tagger relation:")
    print(g.number_of_edges(('user', 'tagger', 'user')))
    print("# of have relation:")
    print(g.number_of_edges(('user', 'have', 'post')))
    print("# of simiarity relation:")
    try:
        print(g.number_of_edges(('user', 'similar', 'post')))
    except:
        print(0)
    print("# of mention relation:")
    print(g.number_of_edges(('user', 'mention', 'post')))
    print("# of reply relation:")
    print(g.number_of_edges(('user', 'reply', 'post')))
    print("# of include relation:")
    print(g.number_of_edges(('post', 'include', 'keyword')))
    print("# of tag relation:")
    print(g.number_of_edges(('post', 'tag', 'keyword')))
    print("# of profile relation:")
    print(g.number_of_edges(('user', 'profile', 'keyword')))

    print(g.successors(588,etype='include'))
    print(g.canonical_etypes)
    print(g.etypes)

    edge_count = 0
    for item in g.canonical_etypes:
        edge_count += g.number_of_edges(item)

    return g,edge_count


def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class StructureLearning(nn.Module):
    def __init__(self,threshold,fea_ft,w_ft):
        super(StructureLearning,self).__init__()

        self.w1 = nn.Parameter(torch.empty(fea_ft,w_ft))
        nn.init.xavier_uniform_(self.w1, gain=nn.init.calculate_gain('relu'))

        self.w2 = nn.Parameter(torch.empty(fea_ft,w_ft))
        nn.init.xavier_uniform_(self.w2, gain=nn.init.calculate_gain('relu'))

        self.threshold = threshold

        self.add_relation_up,self.add_relation_uu = [],[]

    def forward(self,fea1,fea2):

        embed1 = torch.mm(fea1,self.w1)
        embed2 = torch.mm(fea2,self.w2)

        sim_up = cos_sim(embed1, embed2).float()
        sims_up = torch.where(sim_up < self.threshold, torch.zeros_like(sim_up), sim_up)
        edge_index = torch.nonzero(sims_up,as_tuple=False)

        for irow in range(edge_index.shape[0]):
            self.add_relation_up.append([edge_index[irow,0],edge_index[irow,1]])

        sim_uu = cos_sim(embed1,embed1)
        sims_uu = torch.where(sim_uu < self.threshold, torch.zeros_like(sim_uu), sim_uu)
        edge_index_uu = torch.nonzero(sims_uu, as_tuple=False)

        for irow in range(edge_index_uu.shape[0]):
            self.add_relation_uu.append([edge_index_uu[irow, 0], edge_index_uu[irow, 1]])


        return self.add_relation_up, self.add_relation_uu





