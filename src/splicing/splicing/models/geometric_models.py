import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, Linear, GATv2Conv

from splicing.utils.general_utils import compute_conv1d_lout
from splicing.utils.graph_utils import build_node_representations


class FullModel(nn.Module):
    """ the model that predicts the splicing based on combined window and
        base pair information """
    def __init__(self, opt, device='cuda'):
        super(FullModel, self).__init__()

        self.device = device        
        
        self.nt_conv = opt.nt_conv
        self.zeronuc = opt.zeronuc
        self.zeronodes = opt.zeronodes

        if self.nt_conv:
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

        self.out = nn.Conv1d(
            in_channels=opt.hidden_size_full,
            out_channels=3,
            kernel_size=1
        )

        self.dropout = nn.Dropout(opt.gcn_dropout)

        self.out_act = nn.Softmax(dim=1)

    def forward(self, x, node_rep):

        # ignore certain features
        if self.zeronuc:
            x = torch.zeros(x.shape).to('cuda')
        if self.zeronodes:
            node_rep = torch.zeros(node_rep.shape).to('cuda')

        # transform the base pair representations before concatenation
        if self.nt_conv:
            x = self.nucleotide_conv_1(x)
            x = F.relu(x)
            x = self.batch_norm_n_1(x)

        # concatenate the base pair and replicated node representations
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

        # prediction based on the combined features
        out = self.out(x)
        out = self.out_act(out)

        return out


class SpliceGraph(torch.nn.Module):
    """ the graph convolutional model """
    def __init__(self, opt):
        super().__init__()
        
        self.node_representation = opt.node_representation
        
        # rep_size: size of the node representation
        self.rep_size = opt.n_channels
        if opt.node_representation == 'min-max':
            self.rep_size = opt.n_channels * 2
        elif opt.node_representation == 'pca':
            self.rep_size = opt.n_channels * opt.pca_dims # might be changed
        elif opt.node_representation == 'summary':
            self.rep_size = opt.n_channels * 5
        elif opt.node_representation == "conv1d":
            self.rep_size = opt.hidden_size   

        # different possible node representations
        # learned node matrix
        self.nr_conv1d = False
        if opt.node_representation == 'conv1d':

            self.nr_conv1d = True
            self.nr_model = opt.nr_model
            
            if opt.nr_model == 'fredclem':
                self.nr_conv1d_1 = nn.Conv1d(
                    in_channels=self.rep_size,
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

                self.nr_linear = nn.Linear(l_out_2, self.rep_size)
                

            elif opt.nr_model == "clem_bn":
                
                self.nr_bn_start = nn.BatchNorm1d(32)
                self.nr_conv1d_1 = nn.Conv1d(32, 16, kernel_size=1)
                self.nr_maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4)
                self.nr_bn_1 = nn.BatchNorm1d(4)
                
                self.nr_conv1d_2 = nn.Conv1d(4,12, kernel_size=22, stride=10, padding=6)
                s1 = compute_conv1d_lout(opt.window_size, padding=6, kernel_size=22, stride=10)
                self.nr_maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=3)
                self.nr_bn_2 = nn.BatchNorm1d(4)
                
                self.nr_conv1d_3 = nn.Conv1d(4,8, kernel_size=22, stride=10, padding=6)
                s2 = compute_conv1d_lout(s1, padding=6, kernel_size=22, stride=10)
                self.nr_maxpool_3 = nn.MaxPool1d(kernel_size=2, stride=2)
                self.nr_bn_3 = nn.BatchNorm1d(4)

                self.nr_linear = nn.Linear(s2*4, self.rep_size)
            
            elif opt.nr_model == "linbig":
                self.nr_bn_start = nn.BatchNorm1d(32)
                self.nr_conv1d_1 = nn.Conv1d(32, 16, kernel_size=1)
                self.nr_maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4)
                self.nr_bn_1 = nn.BatchNorm1d(4)
                self.nr_conv1d_2 = nn.Conv1d(4,1,kernel_size=1)
                self.nr_bn_2 = nn.BatchNorm1d(1)
                self.nr_linear = nn.Linear(opt.window_size, self.rep_size)

            elif opt.nr_model == "linmed": 
                self.nr_bn_start = nn.BatchNorm1d(32)
                self.nr_conv1d_1 = nn.Conv1d(32, 4, kernel_size=1)
                self.nr_maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4)
                self.nr_bn_1 = nn.BatchNorm1d(1)
                self.nr_linear = nn.Linear(opt.window_size, self.rep_size)

            elif opt.nr_model == "linsmall":
                self.nr_bn_start = nn.BatchNorm1d(32)
                self.nr_conv1d_1 = nn.Conv1d(32, 1, kernel_size=1)
                self.nr_bn_1 = nn.BatchNorm1d(1)
                self.nr_linear = nn.Linear(opt.window_size, self.rep_size)
                
        # single-layer graph conv
        if opt.gat_conv:
            self.gcn_conv = GATv2Conv(
                self.rep_size, opt.hidden_size, heads=opt.n_heads)
        else:
            self.gcn_conv = GCNConv(self.rep_size, opt.hidden_size)
        self.gcn_lin = Linear(self.rep_size, opt.hidden_size)
        self.gcn_gate = Linear(opt.hidden_size, opt.hidden_size)
        self.gcn_bn = BatchNorm(opt.hidden_size)
        self.gcn_dropout = nn.Dropout(opt.gcn_dropout)

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

        # gated graph convolution
        z = self.gcn_conv(x, edge_index)
        z = F.tanh(z)
        g = F.sigmoid(self.gcn_gate(z))
        x = self.gcn_lin(x)  # change dimension
        x = (1 - g) * x + g * z
        x = F.relu(x)
        x = self.gcn_bn(x)
        x = self.gcn_dropout(x)

        return x


# ensemble models to test trained models
class SpliceGraphEnsemble(nn.Module):
    def __init__(self, graph_models):
        super(SpliceGraphEnsemble, self).__init__()
        self.graph_models = graph_models

    def forward(self, xs, edge_index, opt):

        node_reps = [model.cuda()(
            xs, edge_index, opt).cpu() for model in self.graph_models]

        return node_reps


class FullModelEnsemble(nn.Module):
    def __init__(self, full_models, window_size):
        super(FullModelEnsemble, self).__init__()
        self.full_models = full_models
        self.window_size = window_size

    def forward(self, input, node_reps):

        predictions = torch.zeros(
            size=(len(self.full_models), input.shape[0], 3, self.window_size)
        )

        for ii, (model, node_rep) in enumerate(
                zip(self.full_models, node_reps)):
            predictions[ii, :, :, :] = model.cuda()(
                input.cuda(), node_rep.cuda()).cpu()

        combined_predictions = torch.mean(predictions, dim=0)

        return combined_predictions.cpu()