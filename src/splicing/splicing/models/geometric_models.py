import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class SpliceGraph(torch.nn.Module):
    def __init__(self, in_channels, n_channels, dropout_ratio):
        super().__init__()
        self.conv1 = GCNConv(in_channels, n_channels)
        self.bn1 = BatchNorm(n_channels)
        self.conv2 = GCNConv(n_channels, 3)
        self.dropout_ratio = dropout_ratio

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
