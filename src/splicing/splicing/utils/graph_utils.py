import os
from os import path
import logging

import numpy as np
import torch
from scipy import sparse
import wandb
import pickle
from splicing.models.losses import CategoricalCrossEntropy2d


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
    config.context_length = opt.context_length
    config.kernel_size = opt.kernel_size
    config.dilation_rate = opt.dilation_rate
    config.batch_size = opt.batch_size
    config.epochs = opt.epochs
    config.context_length = opt.context_length
    config.lr = opt.lr
    config.train_ratio = opt.train_ratio
    config.class_weights = opt.class_weights
    config.kernel_size = opt.kernel_size
    config.dilation_rate = opt.dilation_rate
    config.batch_size = opt.batch_size

    return config


def get_criterion(opt):
    return CategoricalCrossEntropy2d(opt.class_weights)


def get_optimizer(model, opt):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.98), lr=opt.lr)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr, weight_decay=1e-6, momentum=0.9)
    return optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize_data(data):
    num_train = len(data['train']['tgt'])
    num_valid = len(data['valid']['tgt'])
    num_test = len(data['test']['tgt'])

    print('Train set size: ' + str(num_train))
    print('Validation set size: ' + str(num_valid))
    print('Test set size: ' + str(num_test))

    # unconditional_probs = torch.zeros(
    #   len(data['dict']['tgt']),len(data['dict']['tgt']))
    train_label_vals = torch.zeros(
        len(data['train']['tgt']), len(data['dict']['tgt']))
    for i in range(len(data['train']['tgt'])):
        indices = torch.from_numpy(np.array(data['train']['tgt'][i]))
        x = torch.zeros(len(data['dict']['tgt']))
        x.index_fill_(0, indices, 1)
        train_label_vals[i] = x

        # for idx1 in indices:
        #     for idx2 in indices:
        #         unconditional_probs[idx1,idx2] += 1

    # unconditional_probs = unconditional_probs[4:,4:]
    train_label_vals = train_label_vals[:, 4:]

    pearson_matrix = np.corrcoef(
        train_label_vals.transpose(0, 1).cpu().numpy())

    valid_label_vals = torch.zeros(len(data['valid']['tgt']),
                                   len(data['dict']['tgt']))
    for i in range(len(data['valid']['tgt'])):
        indices = torch.from_numpy(np.array(data['valid']['tgt'][i]))
        x = torch.zeros(len(data['dict']['tgt']))
        x.index_fill_(0, indices, 1)
        valid_label_vals[i] = x
    valid_label_vals = valid_label_vals[:, 4:]

    train_valid_labels = torch.cat((train_label_vals, valid_label_vals), 0)

    mean_pos_labels = torch.mean(train_valid_labels.sum(1))
    median_pos_labels = torch.median(train_valid_labels.sum(1))
    max_pos_labels = torch.max(train_valid_labels.sum(1))

    print('Mean Labels Per Sample: ' + str(mean_pos_labels))
    print('Median Labels Per Sample: ' + str(median_pos_labels))
    print('Max Labels Per Sample: ' + str(max_pos_labels))

    mean_samples_per_label = torch.mean(train_valid_labels.sum(0))
    median_samples_per_label = torch.median(train_valid_labels.sum(0))
    max_samples_per_label = torch.max(train_valid_labels.sum(0))

    print('Mean Samples Per Label: ' + str(mean_samples_per_label))
    print('Median Samples Per Label: ' + str(median_samples_per_label))
    print('Max Samples Per Label: ' + str(max_samples_per_label))


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


def process_graph(adj_type, split_adj_dict_chrom, x_size, chrom, bin_dict, window_size):
    constant_range = 7
    if adj_type == 'constant':
        split_adj = create_constant_graph_intragenetic(constant_range, x_size, chrom, bin_dict, window_size)
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


def save_feats(model_name, split, Y, locations, X):
    logging.info(f'Saving features for model {model_name}.')
    location_feature_dict = {}
    location_index_dict = {}
    for idx, location in enumerate(locations):
        if location not in location_index_dict:
            location_index_dict[location] = []
        location_index_dict[location].append(idx)

    for location in location_index_dict:
        chrom_indices = torch.Tensor(location_index_dict[location]).long()
        x = torch.index_select(X, 0, chrom_indices)
        y = torch.index_select(Y, 0, chrom_indices)
        location_feature_dict[location] = {'x': x, 'y': y}

    features_dir = model_name.split('.finetune')[0]
    directory_setup(features_dir)
    torch.save(location_feature_dict, path.join(
        features_dir, f'chrom_feature_dict_{split}.pt'))