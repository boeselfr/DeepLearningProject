import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, \
    Linear, Sequential

from splicing.utils.general_utils import compute_conv1d_lout
from splicing.utils.graph_utils import build_node_representations

class FullModel(nn.Module):
    def __init__(self, opt, device='cuda'):
        super(FullModel, self).__init__()

        self.device = device

        self.zeronuc = opt.zeronuc
        
        self.nt_conv = False
        if opt.nt_conv:
            self.nt_conv = True
            self.batch_norm_n_1 = nn.BatchNorm1d(opt.n_channels)
            self.nucleotide_conv_1 = nn.Conv1d(
                in_channels=opt.n_channels,
                out_channels=opt.n_channels,
                kernel_size=1)

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
        )

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

        if self.zeronuc:
            x = torch.zeros(x.shape).to('cuda')
        
        if self.nt_conv:
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
        
        self.node_representation = opt.node_representation
        if opt.node_representation == 'min-max':
            n_channels = opt.n_channels * 2
        elif opt.node_representation == 'pca':
            n_channels = opt.n_channels * opt.pca_dims # might be changed
        elif opt.node_representation == 'summary':
            n_channels = opt.n_channels * 5
        elif opt.node_representation == "conv1d":
            n_channels = opt.hidden_size
        else:
            n_channels = opt.n_channels    

        # learned node matrix
        self.nr_conv1d = False
        if opt.node_representation == 'conv1d':

            self.nr_conv1d = True
            self.nr_model = opt.nr_model
            
            if opt.nr_model == 'fredclem':
                self.nr_conv1d_1 = nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=16,
                    kernel_size=1
                )
                self.nr_relu1 = nn.ReLU()
                self.nr_maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
                self.nr_bn_1 = nn.BatchNorm1d(4)
                self.nr_conv1d_2 = nn.Conv1d(
                    4, 
                    4,
                    kernel_size=11,
                    dilation=1,
                    stride=4
                )
                # compute resulting dim
                l_out_1 = compute_conv1d_lout(5000, 1, 11, 4) 
                self.nr_relu2 = nn.ReLU()
                self.nr_bn_2 = nn.BatchNorm1d(4)
                self.nr_conv1d_3 = nn.Conv1d(
                    4, 
                    1, 
                    kernel_size=11,
                    dilation=1,
                    stride=6
                )
                l_out_2 = compute_conv1d_lout(l_out_1, 1, 11, 6)
                self.nr_relu3 = nn.ReLU()
                self.nr_bn_3 = nn.BatchNorm1d(1)

                self.nr_linear = nn.Linear(l_out_2, n_channels)
                

            elif opt.nr_model == "clem_bn":
                
                self.nr_bn_start = nn.BatchNorm1d(32)
                self.nr_conv1d_1 = nn.Conv1d(32, 16, kernel_size=1)
                self.nr_maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4)
                self.nr_bn_1 = nn.BatchNorm1d(4)
                
                self.nr_conv1d_2 = nn.Conv1d(4,12, kernel_size=22, stride=10, padding=6)
                self.nr_maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=3)
                self.nr_bn_2 = nn.BatchNorm1d(4)
                
                self.nr_conv1d_3 = nn.Conv1d(4,8, kernel_size=22, stride=10, padding=6)
                self.nr_maxpool_3 = nn.MaxPool1d(kernel_size=2, stride=2)
                self.nr_bn_3 = nn.BatchNorm1d(4)
                self.nr_linear = nn.Linear(50*4, n_channels)
            
            elif opt.nr_model == "linbig":
                self.nr_bn_start = nn.BatchNorm1d(32)
                self.nr_conv1d_1 = nn.Conv1d(32, 16, kernel_size=1)
                self.nr_maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4)
                self.nr_bn_1 = nn.BatchNorm1d(4)
                self.nr_conv1d_2 = nn.Conv1d(4,1,kernel_size=1)
                self.nr_bn_2 = nn.BatchNorm1d(1)
                self.nr_linear = nn.Linear(5000, n_channels)

            elif opt.nr_model == "linmed": 
                self.nr_bn_start = nn.BatchNorm1d(32)
                self.nr_conv1d_1 = nn.Conv1d(32, 4, kernel_size=1)
                self.nr_maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4)
                self.nr_bn_1 = nn.BatchNorm1d(1)
                self.nr_linear = nn.Linear(5000, n_channels)

            elif opt.nr_model == "linsmall":
                self.nr_bn_start = nn.BatchNorm1d(32)
                self.nr_conv1d_1 = nn.Conv1d(32, 1, kernel_size=1)
                self.nr_bn_1 = nn.BatchNorm1d(1)
                self.nr_linear = nn.Linear(5000, n_channels)
                

        # single graph conv
        self.gcn_conv = GCNConv(n_channels, opt.hidden_size)
        self.gcn_lin = Linear(n_channels, opt.hidden_size)
        self.gcn_gate = Linear(opt.hidden_size, opt.hidden_size)
        self.gcn_bn = BatchNorm(opt.hidden_size)
        self.gcn_dropout = nn.Dropout(opt.gcn_dropout)

        #nn.BatchNorm(opt.hidden_size)

    def forward(self, xs, edge_index, opt):
        
        # build node representations:
        x = build_node_representations(
            xs, opt.node_representation, opt
        )

        # node rep convolution
        if self.nr_conv1d:

            if self.nr_model == "clem_bn": 
                x = self.nr_bn_start(x)
                x = self.nr_conv1d_1(x)
                x = nn.ReLU()(x)
                x = self.nr_maxpool_1(x.permute(0,2,1)).permute(0,2,1)
                x = self.nr_bn_1(x)
                x = self.nr_conv1d_2(x)
                x = nn.ReLU()(x)
                x = self.nr_maxpool_2(x.permute(0,2,1)).permute(0,2,1)
                x = self.nr_bn_2(x)
                x = self.nr_conv1d_3(x)
                x = nn.ReLU()(x)
                x = self.nr_maxpool_3(x.permute(0,2,1)).permute(0,2,1)
                x = self.nr_bn_3(x)

                x = torch.reshape(x, (x.shape[0],x.shape[1]*x.shape[2]))

                x = self.nr_linear(x)
            
            elif opt.nr_model == "linbig":
                x = self.nr_bn_start(x)
                x = self.nr_conv1d_1(x)
                x = nn.ReLU()(x)
                x = self.nr_maxpool_1(x.permute(0,2,1)).permute(0,2,1)
                x = self.nr_bn_1(x)
                x = self.nr_conv1d_2(x)
                x = nn.ReLU()(x)
                x = self.nr_bn_2(x)
                x = torch.reshape(x, (x.shape[0],x.shape[1]*x.shape[2]))
                x = self.nr_linear(x)

            elif opt.nr_model == "linmed": 
                x = self.nr_bn_start(x)
                x = self.nr_conv1d_1(x)
                x = nn.ReLU()(x)
                x = self.nr_maxpool_1(x.permute(0,2,1)).permute(0,2,1)
                x = self.nr_bn_1(x)
                x = torch.reshape(x, (x.shape[0],x.shape[1]*x.shape[2]))
                x = self.nr_linear(x)

            elif opt.nr_model == "linsmall":
                x = self.nr_bn_start(x)
                x = self.nr_conv1d_1(x)
                x = nn.ReLU()(x)
                x = self.nr_bn_1(x)
                x = torch.reshape(x, (x.shape[0],x.shape[1]*x.shape[2]))
                x = self.nr_linear(x)

        z = self.gcn_conv(x, edge_index)
        z = F.tanh(z)
        g = F.sigmoid(self.gcn_gate(z))
        x = self.gcn_lin(x)  # change dimension
        x = (1 - g) * x + g * z
        x = F.relu(x)
        x = self.gcn_bn(x)
        x = self.gcn_dropout(x)

        return x
