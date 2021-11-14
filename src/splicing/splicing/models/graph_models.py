import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

"""
Full Chromosome  Models

ChromeGCN

Input: DNA window features across an entire chromosome
Output: Epigenomic state prediction for all windows

"""


class SpliceGraph(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, gate, layers):
        super(SpliceGraph, self).__init__()
        self.GC1 = GraphConvolution(
            input_size, hidden_size, bias=True, init='xavier')
        self.W1 = nn.Linear(input_size, 1)
        if layers == 2:
            self.GC2 = GraphConvolution(
                hidden_size, input_size, bias=True, init='xavier')
            self.W2 = nn.Linear(input_size, 1)
        self.dropout = dropout
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.out = nn.Linear(input_size, 3)

    def forward(self, x_in, adj, deg, src_dict=None, return_gate=False):
        g2 = None
        x = x_in
        z = self.GC1(x, adj, deg)
        z = F.tanh(z)
        g = F.sigmoid(self.W1(z))
        x = (1 - g) * x + g * z
        if hasattr(self, 'GC2'):
            x = F.dropout(x, self.dropout, training=self.training)
            z2 = self.GC2(x, adj, deg)
            z2 = F.tanh(z2)
            g2 = F.sigmoid(self.W2(z2))
            x = (1 - g2) * x + (g2) * z2

        x = F.relu(x)
        x = self.batch_norm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.out(x)
        return x_in, out, (g, g2), None


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            self.reset_parameters_uniform()
        elif init == 'xavier':
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data,
                               gain=0.02)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj, deg):
        support = torch.mm(input, self.weight)
        if adj is not None:
            output = torch.spmm(adj, support)
        else:
            output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
