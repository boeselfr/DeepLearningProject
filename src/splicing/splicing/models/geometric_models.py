from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv, BatchNorm, Linear
from splicing.utils.graph_utils import compute_conv1d_lout

class FullModel(nn.Module):
    def __init__(self, opt, device='cuda'):
        super(FullModel, self).__init__()

        self.device = device

        self.batch_norm0 = nn.BatchNorm1d(opt.n_channels + opt.hidden_size)

        self.conv1 = nn.Conv1d(
            in_channels=opt.n_channels + opt.hidden_size,
            out_channels=opt.hidden_size_full,
            kernel_size=1).to(self.device)
        self.batch_norm1 = nn.BatchNorm1d(opt.hidden_size_full)

        # self.conv2 = nn.Conv1d(
        #     in_channels=opt.hidden_size_full,
        #     out_channels=opt.hidden_size_full,
        #     kernel_size=1).to(self.device)
        # self.batch_norm2 = nn.BatchNorm1d(opt.hidden_size_full)
        #
        # self.conv3 = nn.Conv1d(
        #     in_channels=opt.hidden_size_full,
        #     out_channels=opt.hidden_size_full,
        #     kernel_size=1).to(self.device)
        # self.batch_norm3 = nn.BatchNorm1d(opt.hidden_size_full)

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
        # self.nucleotide_conv_2 = nn.Conv1d(
        #     in_channels=opt.n_channels,
        #     out_channels=opt.n_channels,
        #     kernel_size=1)
        self.batch_norm_n_1 = nn.BatchNorm1d(opt.n_channels)
        # self.batch_norm_n_2 = nn.BatchNorm1d(opt.n_channels)

    def forward(self, x, node_rep):

        # individual_out = self.out_residual_conv(x)
        # residual_x_1 = self.residual_conv1(x)
        # residual_x_2 = self.residual_conv2(x)

        x = self.nucleotide_conv_1(x)
        x = F.relu(x)
        x = self.batch_norm_n_1(x)

        # x = self.nucleotide_conv_2(x)
        # x = F.relu(x)
        # x = self.batch_norm_n_2(x)

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

        # x = self.dropout(x)
        #
        # t = x
        #
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.batch_norm2(x)
        #
        # x = torch.add(x, t)
        #
        # x = self.dropout(x)
        #
        # t = x
        #
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.batch_norm3(x)
        #
        # x = torch.add(x, t)

        out = self.out(x)

        out = self.out_act(out)

        return out


# Look at 'graph_models.py' for ChromeGCN reference
class SpliceGraph(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        if opt.node_representation == 'min-max':
            n_channels = opt.n_channels * 2
        elif opt.node_representation == 'pca':
            n_channels = opt.n_channels * opt.pca_dims # might be changed
        elif opt.node_representation == 'summary':
            n_channels = opt.n_channels * 5
        else:
            n_channels = opt.n_channels    

        # learned node matrix
        if opt.node_representation == 'conv1d':
            
            self.conv1d_1 = nn.Conv1d(
                in_channels=n_channels,
                out_channels=16,
                kernel_size=1
            )
            self.relu1 = nn.ReLU()
            self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
            #self.batch1 = BatchNorm1d(4)
            self.dropout1 = nn.Dropout(p = 0.2)
            self.conv1d_2 = nn.Conv1d(
                4, 
                4,
                kernel_size=11,
                dilation=1,
                stride=4
            )
            # compute resulting dim
            l_out_1 = compute_conv1d_lout(5000, 1, 11, 4) 
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p = 0.2)
            #self.batch2 = nn.BatchNorm1d(4)
            self.conv1d_3 = nn.Conv1d(
                4, 
                1, 
                kernel_size=11,
                dilation=1,
                stride=6
            )
            l_out_2 = compute_conv1d_lout(l_out_1, 1, 11, 6)
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.dropout(p = 0.3)
            #self.batch3 = nn.BatchNorm1d(1)

            opt.rep_size = l_out_2

        # single graph conv
        self.conv1 = GCNConv(opt.rep_size, opt.hidden_size)
        self.lin1 = Linear(opt.rep_size, opt.hidden_size)
        self.gate1 = Linear(opt.hidden_size, opt.hidden_size)
        self.bn1 = BatchNorm(opt.hidden_size)
        self.dropout = nn.Dropout(opt.gcn_dropout)

        #nn.BatchNorm(opt.hidden_size)

    def forward(self, x, edge_index):
        
        # node rep convolution
        x = self.conv1d_1(x)
        x = self.relu1(x)
        x = self.maxpool(x.permute(0,2,1)).permute(0,2,1)
        x = self.dropout1(x)
        x = self.conv1d_2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv1d_3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = torch.squeeze(x)

        z = self.conv1(x, edge_index)
        z = F.tanh(z)
        g = F.sigmoid(self.gate1(z))
        x = self.lin1(x)  # change dimension
        x = (1 - g) * x + g * z
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)

        return x
