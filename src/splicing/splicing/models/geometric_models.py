import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, Linear


class FullModel(nn.Module):
    def __init__(self, opt, device='cuda'):
        super(FullModel, self).__init__()

        # self.out_residual_conv = nn.Conv1d(
        #     in_channels=opt.n_channels,
        #     out_channels=3,
        #     kernel_size=1)

        # self.residual_conv1 = nn.Conv1d(
        #     in_channels=opt.n_channels,
        #     out_channels=opt.hidden_size_full,
        #     kernel_size=1)

        # self.residual_conv2 = nn.Conv1d(
        #     in_channels=opt.n_channels,
        #     out_channels=opt.hidden_size_full,
        #     kernel_size=1)

        self.device = device

        self.batch_norm0 = nn.BatchNorm1d(opt.n_channels + opt.hidden_size)

        self.conv1 = nn.Conv1d(
            in_channels=opt.n_channels + opt.hidden_size,
            out_channels=opt.hidden_size_full,
            kernel_size=1).to(self.device)
        self.batch_norm1 = nn.BatchNorm1d(opt.hidden_size_full)

        self.conv2 = nn.Conv1d(
            in_channels=opt.hidden_size_full,
            out_channels=opt.hidden_size_full,
            kernel_size=1).to(self.device)
        self.batch_norm2 = nn.BatchNorm1d(opt.hidden_size_full)

        self.conv3 = nn.Conv1d(
            in_channels=opt.hidden_size_full,
            out_channels=opt.hidden_size_full,
            kernel_size=1).to(self.device)
        self.batch_norm3 = nn.BatchNorm1d(opt.hidden_size_full)

        self.out = nn.Conv1d(
            in_channels=opt.hidden_size_full,
            out_channels=3,
            kernel_size=1
        ).to(self.device)

        self.dropout = nn.Dropout(opt.gcn_dropout)

        self.out_act = nn.Softmax(dim=1)

        self.nucleotide_conv_1 = nn.Conv1d(
            in_channels=opt.n_channels,
            out_channels=opt.n_channels,
            kernel_size=1)
        self.nucleotide_conv_2 = nn.Conv1d(
            in_channels=opt.n_channels,
            out_channels=opt.n_channels,
            kernel_size=1)
        self.batch_norm_n_1 = nn.BatchNorm1d(opt.n_channels)
        self.batch_norm_n_2 = nn.BatchNorm1d(opt.n_channels)

    def forward(self, x, node_rep):

        # individual_out = self.out_residual_conv(x)
        # residual_x_1 = self.residual_conv1(x)
        # residual_x_2 = self.residual_conv2(x)

        x = self.nucleotide_conv_1(x)
        x = F.relu(x)
        x = self.batch_norm_n_1(x)

        x = self.nucleotide_conv_2(x)
        x = F.relu(x)
        x = self.batch_norm_n_2(x)

        bs, _, sl = x.shape
        _, n_h = node_rep.shape
        x = torch.cat(
            (x, torch.ones(size=(bs, n_h, sl)).to(
                self.device) * node_rep.unsqueeze(2)),
            dim=1)

        x = self.batch_norm0(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)

        x = self.dropout(x)

        t = x

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)

        x = torch.add(x, t)

        x = self.dropout(x)

        t = x

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch_norm3(x)

        x = torch.add(x, t)

        out = self.out(x)

        out = self.out_act(out)

        return out


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
        # self.bn1 = BatchNorm(opt.hidden_size)
        # self.conv2 = GCNConv(opt.hidden_size, opt.hidden_size)
        # self.gate2 = Linear(opt.hidden_size, opt.hidden_size)
        self.bn2 = BatchNorm(opt.hidden_size)
        self.dropout = nn.Dropout(opt.gcn_dropout)

        self.lin2 = Linear(opt.hidden_size, opt.hidden_size)
        self.lin3 = Linear(opt.hidden_size, opt.hidden_size)
        self.lin4 = Linear(opt.hidden_size, opt.hidden_size)
        # self.lin5 = Linear(opt.hidden_size, opt.hidden_size)
        self.bn3 = BatchNorm(opt.hidden_size)
        self.bn4 = BatchNorm(opt.hidden_size)
        # self.bn5 = BatchNorm(opt.hidden_size)

        self.g1_relu_bn = opt.g1_relu_bn

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        z = self.conv1(x, edge_index)
        z = F.tanh(z)
        g = F.sigmoid(self.gate1(z))
        x = self.lin(x)  # change dimension
        x = (1 - g) * x + g * z
        
        # if self.g1_relu_bn:
        #     x = F.relu(x)  # todo: ?
        #     x = self.bn1(x)  # todo: ?

        # x = self.dropout(x)
        # z = self.conv2(x, edge_index)
        # z = F.tanh(z)
        # g = F.sigmoid(self.gate2(z))
        # x = (1 - g) * x + g * z

        x = self.lin2(x)
        x = F.relu(x)
        x = self.bn2(x)

        t = x

        x = self.lin3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = torch.add(x, t)

        x = self.dropout(x)

        t = x

        x = self.lin4(x)
        x = F.relu(x)
        x = self.bn4(x)

        x = torch.add(x, t)

        x = self.dropout(x)

        return x
