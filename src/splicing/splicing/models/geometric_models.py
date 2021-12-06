import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, Linear


class FullModel(nn.Module):
    def __init__(self, opt, device='cuda'):
        super(FullModel, self).__init__()
        # TODO: if concat ...
        self.out = nn.Conv1d(
            in_channels=opt.n_channels + opt.n_hidden,
            out_channels=3,
            kernel_size=1
        ).to(device)
        # self.linear = nn.Linear(2 * n_channels, 3).to(device)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x, node_rep):

        bs, _, sl = x.shape
        _, n_h = node_rep.shape
        x = torch.cat(
            (x, torch.ones(size=(bs, n_h, sl)).cuda() * node_rep.unsqueeze(2)),
            axis=1)

        x = self.out(x)

        x = self.out_act(x)

        return x


# Look at 'graph_models.py' for ChromeGCN reference
class SpliceGraph(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        if opt.node_representation == 'min-max':
            n_channels = opt.n_channels * 2
        else:
            n_channels = opt.n_channels    

        self.lin = Linear(n_channels, opt.hidden_size)
        self.conv1 = GCNConv(n_channels, opt.hidden_size)
        self.gate1 = Linear(opt.hidden_size, opt.hidden_size)
        self.bn1 = BatchNorm(opt.hidden_size)
        self.conv2 = GCNConv(opt.hidden_size, opt.hidden_size)
        self.gate2 = Linear(opt.hidden_size, opt.hidden_size)
        self.bn2 = BatchNorm(opt.hidden_size)
        self.dropout = nn.Dropout(opt.gcn_dropout)
        self.post_g1_relu_bn = opt.g1_relu_bn

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        z = self.conv1(x, edge_index)
        z = F.tanh(z)
        g = F.sigmoid(self.gate1(z))
        x = self.lin(x)  # change dimension
        x = (1 - g) * x + g * z
        
        if self.post_g1_relu_bn:
            x = F.relu(x)  # todo: ?
            x = self.bn1(x)  # todo: ?

        x = self.dropout(x)
        z = self.conv2(x, edge_index)
        z = F.tanh(z)
        g = F.sigmoid(self.gate2(z))
        x = (1 - g) * x + g * z
        x = F.relu(x)
        x = self.bn2(x)
        
        x = self.dropout(x)
        return x
