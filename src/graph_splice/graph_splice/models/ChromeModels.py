import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
import math
from torch.nn.parameter import Parameter
from models.SubLayers import GraphConvolution

"""
Full Chromosome  Models

ChromeGCN

Input: DNA window features across an entire chromosome
Output: Epigenomic state prediction for all windows

"""


class ChromeGCN(nn.Module):
    def __init__(self, input_size, hidden_size, nclass, dropout, gate, layers):
        super(ChromeGCN, self).__init__()
        self.GC1 = GraphConvolution(
            input_size, hidden_size, bias=True, init='xavier')
        self.W1 = nn.Linear(input_size, 1)
        if layers == 2:
            self.GC2 = GraphConvolution(
                hidden_size, input_size, bias=True, init='xavier')
            self.W2 = nn.Linear(input_size, 1)
        self.dropout = dropout
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.out = nn.Linear(input_size, nclass)

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