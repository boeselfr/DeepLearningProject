import os
from os import path
import logging

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score
import wandb
import math

from splicing.models.losses import CategoricalCrossEntropy2d
from splicing.utils.utils import IX2CHR
from sklearn.decomposition import PCA


split2desc = {
    'train': 'Train',
    'valid': 'Validation',
    'test': 'Test',
}


def directory_setup(dir_name: str) -> None:
    """
    Small helper function to handle directory management
    :param dir_name: path to the directory
    """
    if not os.path.exists(dir_name):
        # print(f'Creating directory {dir_name}')
        os.makedirs(dir_name)
    else:
        # print(f'Directory {dir_name} already exists.')
        pass


def get_wandb_config(opt):
    config = wandb.config

    config.n_channels = opt.n_channels
    config.hidden_size = opt.hidden_size
    config.hidden_size_full = opt.hidden_size_full
    config.gcn_dropout = opt.gcn_dropout
    config.context_length = opt.context_length
    config.kernel_size = opt.kernel_size
    config.dilation_rate = opt.dilation_rate
    # config.epochs = opt.epochs

    if opt.pretrain:
        config.cnn_lr = opt.cnn_lr
    elif opt.finetune:
        config.lr = opt.gcn_lr
        config.gcn_lr = opt.gcn_lr
        config.cnn_lr = opt.cnn_lr
        config.full_lr = opt.full_lr

    config.class_weights = opt.class_weights
    config.kernel_size = opt.kernel_size
    config.dilation_rate = opt.dilation_rate
    config.batch_size = opt.batch_size

    config.node_representation = opt.node_representation
    config.adj_type = opt.adj_type

    return config


def report_wandb(predictions, targets, loss, opt, split, step):

    # sums_true = y.sum(axis=(0, 2))
    # sums_pred = predictions.sum(axis=(0, 2))
    #
    # total = sums_true.sum()

    is_expr = targets.sum(axis=(1, 2)) >= 1
    auprcs = {}
    for ix, prediction_type in enumerate(['Acceptor', 'Donor']):
        targets_ix = targets[
                     is_expr, ix + 1, :].flatten().detach().cpu().numpy()
        predictions_ix = predictions[
                         is_expr, ix + 1, :].flatten().detach().cpu().numpy()
        auprcs[prediction_type] = average_precision_score(
            targets_ix, predictions_ix)

    wandb.log({
        f'{split}/loss': loss.item() / opt.batch_size,
        f'{split}/Acceptor AUPRC': auprcs['Acceptor'],
        f'{split}/Donor AUPRC': auprcs['Donor'],
        # f'{split}/true inactive': sums_true[0] / total,
        # f'{split}/true acceptors': sums_true[1] / total,
        # f'{split}/true donors': sums_true[2] / total,
        # f'{split}/predicted inactive': sums_pred[0] / sums_true[0],
        # f'{split}/predicted acceptors': sums_pred[1] / sums_true[1],
        # f'{split}/predicted donors': sums_pred[2] / sums_true[2],
        # f'{split}/proportion of epoch done': batch / (size // batch_size),
    })


def get_criterion(opt):
    return CategoricalCrossEntropy2d(opt.class_weights)


def shuffle_chromosomes(datasets):
    for key in ['train', 'test', 'valid']:
        datasets[key] = datasets[key][
            np.random.permutation(len(datasets[key]))]

    return datasets


def get_optimizer(model, opt):
    if opt.gcn_optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.98), lr=opt.gcn_lr)
    elif opt.gcn_optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.gcn_lr, weight_decay=1e-6, momentum=0.9)
    return optimizer


def get_combined_optimizer(graph_model, full_model, opt):
    if opt.gcn_optim == 'adam':
        optimizer = torch.optim.Adam(
            [
                {'params': list(graph_model.parameters()), 'lr': opt.gcn_lr},
                {'params': list(full_model.parameters()), 'lr': opt.full_lr}
            ], betas=(0.9, 0.98), lr=opt.gcn_lr)
    elif opt.gcn_optim == 'sgd':
        optimizer = torch.optim.SGD(
            [
                {'params': list(graph_model.parameters()), 'lr': opt.gcn_lr},
                {'params': list(full_model.parameters()), 'lr': opt.full_lr}
            ], lr=opt.gcn_lr, weight_decay=1e-6, momentum=0.9)
    return optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_batch(batch_size, src, tgt):
    # Need to pad for dataparallel so all minibatches same size
    diff = batch_size - src[0].size(0)
    src = [
        torch.cat(
            (
                src[0],
                torch.zeros(diff, src[0].size(1)).type(src[0].type()).cuda()),
            0
        ),
        torch.cat(
            (
                src[1],
                torch.zeros(diff, src[1].size(1)).type(src[1].type()).cuda()
            ),
            0
        )
    ]
    tgt = torch.cat(
        (
            tgt,
            torch.zeros(diff, tgt.size(1)).type(tgt.type()).cuda()
        ),
        0
    )
    return src, tgt


def unpad_batch(size, x_out_f, x_out_r, pred, tgt, attn_f, attn_r):
    x_out_f = x_out_f[0:size]
    x_out_r = x_out_r[0:size]
    pred = pred[0:size]
    tgt = tgt[0:size]
    if attn_f is not None:
        attn_f = attn_f[0:size]
        attn_r = attn_r[0:size]
    return x_out_f, x_out_r, pred, tgt, attn_f, attn_r


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # values = train_adj.data.astype(np.float32)
    # indices = np.vstack((train_adj.row, train_adj.col))
    # i = torch.LongTensor(indices)
    # v = torch.FloatTensor(values)
    # shape = train_adj.shape
    # train_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    # train_adj_d = train_adj.to_dense()
    # train_deg = train_adj_d.sum(0)
    # train_deg[train_deg == 0] = 1
    return torch.sparse.FloatTensor(indices, values, shape)


def create_constant_graph(constant_range, x_size):
    diagonals, indices = [], []
    for i in range(-constant_range, constant_range + 1):
        if i != 0:
            diagonals.append(np.ones(x_size - abs(i)))
            indices.append(i)
    split_adj = sparse.diags(diagonals, indices).tocoo()
    return split_adj


def create_constant_graph_intragenetic(constant_range, x_size, chrom, bin_dict, window_size):
    diagonals, indices = [], []
    for i in range(-constant_range, constant_range + 1):
        if i != 0:
            diagonals.append(np.ones(x_size - abs(i)))
            indices.append(i)
    split_adj = sparse.diags(diagonals, indices).tocoo()
    mat_format = split_adj.toarray()
    location_list = list(bin_dict[chrom].keys())
    for row_ind in range(0, x_size):
        # bin_dict[chrom] is an ordered dict, thats why we can just access by index
        total_pos_row = location_list[row_ind]
        for col_ind in range(row_ind - constant_range, row_ind + constant_range + 1):
            # need to check for out of index due to the range going out of bounds at the edges:
            if 0 <= col_ind < x_size:
                total_pos_col = location_list[col_ind]
                # check if difference is more than the expected because of jumps and then set to zero
                if np.abs(total_pos_col-total_pos_row) > np.abs(window_size * (col_ind - row_ind)):
                    mat_format[row_ind, col_ind] = 0
    split_adj = sparse.coo_matrix(mat_format)
    return split_adj


def process_graph(adj_type, split_adj_dict_chrom, x_size,
                  chrom, bin_dict, window_size):
    constant_range = 7
    if adj_type == 'constant':
        split_adj = create_constant_graph_intragenetic(
            constant_range, x_size, chrom, bin_dict, window_size)
        split_adj = split_adj + sparse.eye(split_adj.shape[0])

    elif adj_type in ['hic']:
        split_adj = split_adj_dict_chrom[chrom].tocoo()
        # Set [i,i] = 1 for any row i with no positives
        # diag = split_adj.sum(0)
        # diag = np.array(diag).squeeze()
        # diag[diag>0]=-1
        # diag[diag==0]=1
        # diag[diag==-1]=0
        # split_adj = split_adj.tocsr()
        # split_adj.setdiag(diag)
        split_adj = split_adj + sparse.eye(split_adj.shape[0])

        split_adj[split_adj > 0] = 1
        split_adj[split_adj < 0] = 0

        # split_adj = split_adj.tocoo()
    elif adj_type == 'both':
        const_adj = create_constant_graph_intragenetic(constant_range, x_size, chrom, bin_dict, window_size)
        split_adj = split_adj_dict_chrom[chrom].tocoo() + const_adj
        split_adj = split_adj + sparse.eye(split_adj.shape[0])

    elif adj_type == 'none':
        split_adj = sparse.eye(x_size).tocoo()

    split_adj = normalize(split_adj)
    # not sure we need this:
    split_adj = sparse_mx_to_torch_sparse_tensor(split_adj)
    # the above does exactly what we need already: ...
    # output from sparse_mx_to_torch...:
    """tensor(indices=tensor([[    0,     1,     2,  ..., 19636, 19637, 19638],
                       [    0,     0,     0,  ..., 19638, 19638, 19638]]),
       values=tensor([0.3333, 0.3333, 0.3333,  ..., 0.0312, 0.0263, 0.0233]),
       size=(19639, 19639), nnz=519639, layout=torch.sparse_coo)"""
    return split_adj


def build_node_representations(xs, mode, opt, chromosome=''):

    node_features_fname = path.join(
        opt.model_name,
        f'node_features_{chromosome}_{opt.node_representation}.pt')
    if path.exists(node_features_fname):
        return torch.load(node_features_fname)

    # xs[loc] is of shape : [1, 32, 5000]
    if mode == 'average':
        x = torch.stack([xs[loc][0].mean(axis=1) for loc in xs])
    elif mode == 'max':
        x = torch.stack([xs[loc][0].max(axis=1).values for loc in xs])
    elif mode == 'min':
        x = torch.stack([xs[loc][0].min(axis=1).values for loc in xs])
    elif mode == 'min-max':
        x_min = torch.stack([xs[loc][0].min(axis=1).values for loc in xs])
        # print(x_min.shape)
        x_max = torch.stack([xs[loc][0].max(axis=1).values for loc in xs])
        # print(x_max.shape)
        x = torch.cat((x_min, x_max), 1)
        # print(x.shape)
    elif mode == 'conv1d':
        #device = 'cuda'
        #n_elements = list(xs.values())[0].numel() / opt.n_channels
        #conv1 = torch.nn.Conv1d(
        #    in_channels=n_elements,
        #    out_channels=1,
        #    kernel_size=1).to(device)
        #x_all = torch.stack([(xs[loc][0]) for loc in xs])
        #x_all = torch.moveaxis(x_all,1,2)
        #x = conv1(x_all)
        # of shape:  n_windows x 1 x 32
        #x = torch.squeeze(x)
        x = torch.stack([(xs[loc][0]) for loc in xs])
    elif mode == 'pca':
        pca = PCA(n_components=opt.pca_dims)
        x = torch.stack([torch.flatten(torch.tensor(pca.fit_transform(xs[loc][0]),
                                                    dtype=torch.float).squeeze()) for loc in xs])
    elif mode == 'summary':
        x_min = torch.stack([xs[loc][0].min(axis=1).values for loc in xs])
        # print(x_min.shape)
        x_max = torch.stack([xs[loc][0].max(axis=1).values for loc in xs])
        x_avg = torch.stack([xs[loc][0].mean(axis=1) for loc in xs])
        x_median = torch.stack([xs[loc][0].median(axis=1).values for loc in xs])
        x_std = torch.stack([xs[loc][0].std(axis=1) for loc in xs])
        x = torch.cat((x_min, x_max, x_avg, x_median, x_std), 1)
    elif mode == 'zeros':
        x = torch.stack([torch.zeros(32) for loc in xs])

    return x


def save_node_representations(node_representation, chromosome, opt):
    node_features_fname = path.join(
        opt.results_dir,
        f'node_features_{chromosome}_{opt.node_representation}.pt')
    torch.save(node_representation, node_features_fname)


def save_feats(model_name, split, Y, locations, X, chromosome, epoch):
    # logging.info(f'Saving features for model {model_name}.')

    features_dir = model_name.split('/finetune')[0]
    directory_setup(features_dir)
    data_fname = path.join(
        features_dir, f'chrom_feature_dict_{split}_chr{chromosome}.pt')
    location_feature_dict = {}
    location_index_dict = {}
    if path.exists(data_fname):
        location_feature_dict = torch.load(data_fname)
    else:
        location_feature_dict['x'] = {}
        location_feature_dict['y'] = {}

    for idx, location in enumerate(locations):
        if location not in location_index_dict:
            location_index_dict[location] = []
        location_index_dict[location].append(idx)

    for location in location_index_dict:
        chrom_indices = torch.Tensor(location_index_dict[location]).long()
        x = torch.index_select(X, 0, chrom_indices)
        y = torch.index_select(Y, 0, chrom_indices)
        location_feature_dict['x'][location] = x
        location_feature_dict['y'][location] = y
    torch.save(location_feature_dict, data_fname)


# Deprecated
def save_feats_per_chromosome(
        model_name, split, Y, locations, X, chromosomes, epoch):
    # logging.info(f'Saving features for model {model_name}.')

    features_dir = model_name.split('/finetune')[0]
    directory_setup(features_dir)
    all_chromosomes = set(chromosomes)
    data_fnames = {
        chromosome: path.join(
            features_dir, f'chrom_feature_dict_{split}_chr{chromosome}.pt')
        for chromosome in all_chromosomes
    }

    location_feature_dict = {}
    location_index_dict = {}

    for chromosome in all_chromosomes:
        if path.exists(data_fnames[chromosome]):
            location_feature_dict[chromosome] = torch.load(
                data_fnames[chromosome])
        else:
            location_feature_dict[chromosome] = {
                'x': {}, 'y': {}
            }

    for chromosome in all_chromosomes:
        location_index_dict[chromosome] = {}

    for idx, (chromosome, location) in enumerate(zip(chromosomes, locations)):
        if location not in location_index_dict[chromosome]:
            location_index_dict[chromosome][location] = []
        location_index_dict[chromosome][location].append(idx)

    for chromosome in all_chromosomes:
        for location in location_index_dict[chromosome].keys():
            chrom_indices = torch.Tensor(
                location_index_dict[chromosome][location]).long()
            x = torch.index_select(X, 0, chrom_indices)
            y = torch.index_select(Y, 0, chrom_indices)
            location_feature_dict[chromosome]['x'][location] = x
            location_feature_dict[chromosome]['y'][location] = y

    for chromosome in all_chromosomes:
        torch.save(location_feature_dict[chromosome], data_fnames[chromosome])


def analyze_gradients(graph_model, full_model, _x, nodes, opt):

    nucleotide_grad = list(
        full_model.conv1.parameters())[0].grad[:opt.n_channels, ...].data
    node_grad = list(
        full_model.conv1.parameters())[0].grad[opt.n_channels:, ...].data

    nucleotide_weight = list(
        full_model.conv1.parameters())[0][:opt.n_channels, ...].data
    node_weight = list(
        full_model.conv1.parameters())[0][opt.n_channels:, ...].data

    log_message = {
        'full_grad/full_nucleotide': np.linalg.norm(
            nucleotide_grad.detach().cpu().numpy()) / opt.n_channels,
        'full_grad/full_node': np.linalg.norm(
            node_grad.detach().cpu().numpy()) / opt.hidden_size,
        'full_weight/full_nucleotide': np.linalg.norm(
            nucleotide_weight.detach().cpu().numpy()) / opt.n_channels,
        'full_weight/full_node': np.linalg.norm(
            node_weight.detach().cpu().numpy()) / opt.hidden_size,
        #'node_gradients/node_grad': np.linalg.norm(
        #    nodes.grad.detach().cpu().numpy()) / len(nodes),
        #'node_representations/node_weights': np.linalg.norm(
        #    nodes.detach().cpu().numpy()) / len(nodes),
        #'nucleotide_gradients/nucleotide_grad': np.linalg.norm(
        #    _x.grad.detach().cpu().numpy()) / len(_x),
    }

    for jj, parameter in enumerate(full_model.parameters()):
        if jj == 2:
            continue
        m = parameter.data.shape[1] if len(parameter.data.shape) > 1 else 1

        log_message[f'full_grad/full{str(jj)}'] = np.linalg.norm(
            parameter.grad.data.detach().cpu().numpy()) / m
        log_message[f'full_weight/full{str(jj)}'] = np.linalg.norm(
            parameter.data.detach().cpu().numpy()) / m

    for jj, parameter in enumerate(graph_model.parameters()):
        m = parameter.data.shape[1] if len(parameter.data.shape) > 1 else 1

        log_message[f'graph_grad/graph{str(jj)}'] = np.linalg.norm(
            parameter.grad.data.detach().cpu().numpy()) / m
        log_message[f'graph_weight/graph{str(jj)}'] = np.linalg.norm(
            parameter.data.detach().cpu().numpy()) / m

    wandb.log(log_message)


def compute_conv1d_lout(l_in, dilation=1, kernel_size=1, stride=1):
    return math.floor((l_in - dilation * (kernel_size - 1) - 1) / stride + 1)
