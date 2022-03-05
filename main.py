#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
from representationlearning.rgcn_model import RGCN
from representationlearning.ssl import SSL
from representationlearning.bg_gsr import StructureLearning, build_relation, build_graph
from metalearning.meta import *
from utils import *


def main(args):

    device = torch.device("cuda:0" if args.gpu >= 0 else "cpu")

    post_feature = torch.from_numpy(np.delete(np.loadtxt(args.postvectorpath), 0, 1)).float()
    user_feature = torch.from_numpy(np.delete(np.loadtxt(args.uservectorpath), 0, 1)).float()
    keyword_feature = torch.from_numpy(np.delete(np.loadtxt(args.keywordvectorpath), 0, 1)).float()


    gsr = StructureLearning(threshold=args.simthreshold, fea_ft=args.fea_dim, w_ft=args.w_dim).to(device)
    ssl = SSL(input_size=2 * args.out_dim, hidden_size1=args.ssl_hidden_dim, out_size=args.ssl_out_dim, ssl_label_dir=args.ssllabelpath,device=device).to(device)
    maml = Meta(args).to(device)

    print("start training...")

    gsr.train()
    ssl.train()
    maml.train()

    relation_list, user_label = build_relation(postid_dir=args.postidpath, userid_dir=args.useridpath,
                                               keywordid_dir=args.keywordidpath, usertype_dir=args.usertypepath,
                                               relation_dir=args.relatiopath)
    if args.gpu >=0:

        post_feature = post_feature.to(device)
        user_feature = user_feature.to(device)
        keyword_feature = keyword_feature.to(device)

    labels = torch.tensor(user_label).to(device)

    for epoch in range(1, args.n_epochs):

        relation_list_copy = relation_list.copy()

        edge_add_up,edge_add_uu = gsr(user_feature, post_feature) # graph structure refinement
        relation_list_copy[1] += edge_add_up
        relation_list_copy[1] += edge_add_uu

        g, edge_count = build_graph(relation_list_copy, user_feature.cpu(), post_feature.cpu(), keyword_feature.cpu())  # build construction

        g = g.to(device)
        edge_count = torch.tensor(edge_count).to(device)

        rgcn = RGCN(g, h_dim=args.hidden_dim, out_dim=args.out_dim, num_bases=args.n_bases,
                    num_hidden_layers=args.n_layers, dropout=args.dropout, use_self_loop=args.use_self_loop).to(device)   # rgcn representation learning

        optimizer = torch.optim.Adam(
            list(rgcn.parameters()) + list(gsr.parameters()) + list(ssl.parameters()) + list(maml.parameters()), lr=args.lr, weight_decay=5e-4)

        print("start training...")

        print("Epoch : {}".format(epoch))

        for g_epoch in range(0, args.gcn_epoch):

            logits = rgcn()[args.target_ent]

            meta_trainfeat, meta_trainlabel, meta_testfeat, meta_testlabel = get_metalearndata(logits, labels,  args.fs_label, args.neg_label)  # get data for meta-learning
            meta_train_acc = []
            meta_train_f1 = []

            for train_task in args.metatrainlabel:
 
                   x_spt_train, y_spt_train, x_qry_train, y_qry_train = get_metatrain_data(meta_trainfeat, meta_trainlabel, train_task, args.k_spt,
                                                                args.k_qry, args.batch_num)     # get mete-training  data
                   loss_meta, accs, f1 = maml.forward(x_spt_train, y_spt_train, x_qry_train, y_qry_train)   # mete-training

                meta_train_acc.append(accs[-1])
                meta_train_f1.append(f1[-1])

            print("{} epoch: |train_F1: {:.4f} |train_accuracy: {:.4f}".format(g_epoch, sum(meta_train_f1)/len(meta_train_f1), sum(meta_train_acc)/len(meta_train_acc)))

            if args.ssl:

                loss_ssl = ssl(logits)     # self-supervised learning augmentation
                loss_total = (args.ldargcn * loss_meta + args.ldassl * loss_ssl + args.ldagsr * edge_count).float()
            else:
                loss_total = (args.ldargcn *loss_meta + args.ldagsr * edge_count).float()

            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer.step()

            torch.save(maml.state_dict(), 'metalearning/maml.params')    # save the parameters of pre-trained model
            maml_copy = copy.deepcopy(maml)   # pre-trained model (teacher model)

            model_meta_trained = Meta(args).to(device)
            model_meta_trained.load_state_dict(torch.load('maml.params'))  # pre-trained parameters
            model_meta_trained.eval()

            for k in range(args.metateststep):
                x_spt_test, y_spt_test, x_qry_test, y_qry_test = get_metatest_data(meta_testfeat, meta_testlabel, args.fs_label, args.k_spt,
                                                           args.k_qry,args.batch_num)    # get data for meta-testing
                with torch.no_grad():
                    teacher_score = [maml_copy.predict(item) for item in x_qry_test]   # teacher logit scores

                test_f1, test_acc = model_meta_trained.forward_kd(
                    x_spt_test, y_spt_test, x_qry_test, y_qry_test, teacher_score, kd=args.kd,temp=args.temp,alpha=args.ldakd)   # meta-testing

            print("{} epoch: |test_F1: {:.4f} |test_accuracy: {:.4f}".format(g_epoch,test_f1[-1] ,test_acc[-1]))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta-HG')


    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="number of training epochs")

    ################ self-supervised learning#####################
    parser.add_argument('--ssl', default=True, help='self-supervised learning')
    parser.add_argument('--ldassl', type=float, default=5, help='hyparameters of ssl loss')
    parser.add_argument('--ssllabelpath', type=str, default='./data/ssl_label.txt', help='label of ssl task')
    parser.add_argument('--ssl_hidden_dim', type=int, default=200, help='hyparameters of ssl loss')
    parser.add_argument('--ssl_out_dim', type=int, default=2, help='hyparameters of ssl loss')

    ################ graph structure learning#####################
    parser.add_argument('--ldagsr', type=int, default=0.0001, help='hyparameters of gsr regularizer')
    parser.add_argument('--simthreshold', type=float, default=0.95, help='similarity threshold ')
    parser.add_argument('--fea_dim', type=int, default=400, help='deminsion of attributed feature')
    parser.add_argument('--w_dim', type=int, default=400, help='deminsion of weight W')
    parser.add_argument('--useridpath', type=str, default='./data/user_id_add.json', help='user-id match json')
    parser.add_argument('--postidpath', type=str, default='./data/post_id_add.json', help='post-id match json')
    parser.add_argument('--keywordidpath', type=str, default='./data/keyword_id_add_content.json',
                        help='keyword-id match json')
    parser.add_argument('--uservectorpath', type=str, default='./data/userid_merged_vector_add.txt',
                        help='user feature vector')
    parser.add_argument('--postvectorpath', type=str, default='./data/postid_merged_vector_add.txt',
                        help='post feature vector')
    parser.add_argument('--keywordvectorpath', type=str, default='./data/keywordid_merged_vector_add.txt',
                        help='keyword feature vector')
    parser.add_argument('--relatiopath', type=str, default='./data/relation.txt',
                        help='relation file')
    parser.add_argument('--usertypepath', type=str, default='./data/userid_categ_add.json',
                        help='user type file')

    ################ rgcn#####################
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument('--target_ent', type=str, default='user', help='user entity to train gcn')
    parser.add_argument('--gcn_epoch', type=int, default=100, help='gcn learning epochs')
    parser.add_argument("--model_path", type=str, default=None, help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")
    parser.add_argument("--use_self_loop", default=True, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument('--ldargcn', type=int, default=1, help='hyparameters of rgcn loss')
    parser.add_argument('--hidden_dim', type=int, default=200, help='dimension of hidden layer')
    parser.add_argument('--out_dim', type=int, default=200, help='dimension of out layer')
    parser.add_argument("--n_layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("--n_bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")

    ################ meta-learning#####################
    parser.add_argument('--n_way', type=int, help='number of classification', default=2)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.05)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.08)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    parser.add_argument('--batch_num', type=int, help='meta batch size', default=10)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=20)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=250)
    parser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)
    parser.add_argument('--kd', type=int, default=1, help='Use knowledge distillation')
    parser.add_argument('--temp', type=float, default=10.0, help='temperature index in knowledge distillation')
    parser.add_argument('--ldakd', type=float, default=0.01, help='trade-off value for knowledge distillation')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                        help='Normalization method for the adjacency matrix.')
    parser.add_argument('--metaseed', type=int, default=3, help='Random seed.')
    parser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    parser.add_argument('--metateststep', type=int, default=50, help='How many times to random select node to test')
    parser.add_argument('--fs_label', type=int, default=1, help='the label of few shot')
    parser.add_argument('--neg_label', type=int, default=0, help='label of negative data')
    parser.add_argument('--metatrainlabel', type=list, default=[2, 3, 4, 5], help='meta train task labels')
    parser.add_argument('--embed_dim', type=int, default=200, help='node embedding dimension')

    args = parser.parse_args()
    print(args)
    main(args)
