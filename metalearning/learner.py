import torch
from torch import nn
from torch.nn import functional as F


class Learner(nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config
        self.vars = nn.ParameterList()
        # self.vars.cuda()
        self.vars
        self.vars_bn = nn.ParameterList()
        # self.vars_bn.cuda()


        for i, (name, param) in enumerate(self.config):
            if name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.xavier_uniform_(w)
                # torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d', 'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError
        return info

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            # vars = self.vars.cuda()
            vars = self.vars
        idx = 0
        bn_idx = 0
        # x = x.cuda()
        x = x
        for name, param in self.config:
            if name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            # elif name is 'linear':
            #     w, b = vars[idx], vars[idx + 1]
            #     x = F.linear(x, w, b)
            #     idx += 2
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars

    def distill(self,y,label,teacher_score,student_score,temp,alpha):
        # y = y.cuda()
        # label = label.cuda()
        # y_fake = teacher_score.F.softmax(logits_q, dim=1).argmax(dim=1)
        return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_score / temp,dim=1),F.softmax(teacher_score/ temp ,dim=1))*(temp*temp*2.*alpha)+F.cross_entropy(y,label)*(1.0-alpha)