import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from metalearning.learner import Learner
from utils import get_performance



class Meta(nn.Module):
    def __init__(self, args):
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.batch_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.embed_dim = args.out_dim
        self.config = [
            ('linear', [args.hidden, self.embed_dim]),
            ('relu', [args.hidden, self.embed_dim]),
            ('linear', [args.n_way, args.hidden])
        ]
        self.net = Learner(self.config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)



    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num = self.task_num
        querysz = y_qry[0].shape[0]

        losses_q = [0 for _ in range(self.update_step + 1)]
        f1s = [0 for _ in range(self.update_step + 1)]
        accs = [0 for _ in range(self.update_step + 1)]
        recalls = [0 for _ in range(self.update_step + 1)]
        precs = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            x_spt[i] = x_spt[i]
            y_spt[i] = y_spt[i]
            x_qry[i] = x_qry[i]
            y_qry[i] = y_qry[i]

            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i].squeeze())
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].squeeze())
                losses_q[0] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i].squeeze()).sum().item()
                corrects[0] = corrects[0] + correct

                f1_sub, acc_sub, recall_sub, prec_sub = get_performance(logits_q, y_qry[i])

                f1s[0] = f1s[0] + f1_sub
                accs[0] = accs[0] + acc_sub
                recalls[0] = recalls[0] + recall_sub
                precs[0] = precs[0] + prec_sub

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].squeeze())
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i].squeeze()).sum().item()
                corrects[1] = corrects[1] + correct

                f1_sub,acc_sub,recall_sub,prec_sub = get_performance(logits_q, y_qry[i])

                f1s[1] = f1s[1] + f1_sub
                accs[1] = accs[1] + acc_sub
                recalls[1] = recalls[1] + recall_sub
                precs[1] = precs[1] + prec_sub

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].squeeze())
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].squeeze())
                losses_q[k + 1] += loss_q   #

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i].squeeze()).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

                    f1_sub, acc_sub, recall_sub, prec_sub = get_performance(logits_q, y_qry[i])

                    f1s[k+1] = f1s[k+1] + f1_sub
                    accs[k+1] = accs[k+1] + acc_sub
                    recalls[k+1] = recalls[k+1] + recall_sub
                    precs[k+1] = precs[k+1] + prec_sub


        loss_q = losses_q[-1] / task_num

        acc = np.array(corrects) / (querysz * task_num)

        f1 = np.array(f1s) / (task_num)

        return loss_q, acc, f1


    def forward_kd(self, x_spt, y_spt, x_qry, y_qry,teacher_score,kd,temp,alpha):
        task_num = self.task_num


        losses_q = [0 for _ in range(self.update_step + 1)]
        f1s = [0 for _ in range(self.update_step + 1)]
        accs = [0 for _ in range(self.update_step + 1)]
        recalls = [0 for _ in range(self.update_step + 1)]
        precs = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            x_spt[i] = x_spt[i]
            y_spt[i] = y_spt[i]
            x_qry[i] = x_qry[i]
            y_qry[i] = y_qry[i]
            logits_meta_train = self.net(x_spt[i], vars=None, bn_training=True)

            with torch.no_grad():
                logits_meta_val = self.net(x_qry[i], vars=None, bn_training=True)
            if kd ==1:

                distill_loss = self.net.distill(logits_meta_train,y_spt[i].squeeze(),teacher_score[i],logits_meta_val,temp=temp,alpha=alpha)  # distillation loss
                grad = torch.autograd.grad(distill_loss, self.net.parameters())

            elif kd == 0:
                loss = F.cross_entropy(logits_meta_train, y_spt[i].squeeze())
                grad = torch.autograd.grad(loss, self.net.parameters())

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].squeeze())
                losses_q[0] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i].squeeze()).sum().item()
                corrects[0] = corrects[0] + correct

                f1_sub, acc_sub, recall_sub, prec_sub = get_performance(logits_q, y_qry[i])

                f1s[0] = f1s[0] + f1_sub
                accs[0] = accs[0] + acc_sub
                recalls[0] = recalls[0] + recall_sub
                precs[0] = precs[0] + prec_sub

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].squeeze())
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i].squeeze()).sum().item()
                corrects[1] = corrects[1] + correct

                f1_sub, acc_sub, recall_sub, prec_sub = get_performance(logits_q, y_qry[i])

                f1s[1] = f1s[1] + f1_sub
                accs[1] = accs[1] + acc_sub
                recalls[1] = recalls[1] + recall_sub
                precs[1] = precs[1] + prec_sub

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)

                with torch.no_grad():
                    logits_meta_val = self.net(x_qry[i], fast_weights, bn_training=True)
                if kd == 1:
                    distill_loss = self.net.distill(logits, y_spt[i].squeeze(), teacher_score[i],logits_meta_val,
                                                              temp=temp, alpha=alpha)  # distillation loss
                    grad = torch.autograd.grad(distill_loss, fast_weights)
                elif kd == 0:

                    loss = F.cross_entropy(logits, y_spt[i].squeeze())
                    grad = torch.autograd.grad(loss, fast_weights)

                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)

                loss_q = F.cross_entropy(logits_q, y_qry[i].squeeze())

                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i].squeeze()).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

                    f1_sub, acc_sub, recall_sub, prec_sub = get_performance(logits_q, y_qry[i])

                    f1s[k + 1] = f1s[k + 1] + f1_sub
                    accs[k + 1] = accs[k + 1] + acc_sub
                    recalls[k + 1] = recalls[k + 1] + recall_sub
                    precs[k + 1] = precs[k + 1] + prec_sub

        f1 = np.array(f1s) / (task_num)
        acc = np.array(accs) / (task_num)

        return f1, acc


    def predict(self, x_qry):

        task_num = self.task_num

        with torch.no_grad():
            for i in range(task_num):
                logits = self.net(x_qry, vars=self.net.parameters(), bn_training=True)
                teacher_score = F.softmax(logits, dim=-1)

        return teacher_score