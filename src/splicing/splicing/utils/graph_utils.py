import os
from os import path
import logging

import numpy as np
import torch
from scipy import sparse

from sklearn.decomposition import PCA


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

        #split_adj[split_adj > 0] = 1
        #split_adj[split_adj < 0] = 0

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
    print(split_adj)
    return split_adj


def build_node_representations(xs, mode, opt, chromosome=''):

    node_features_fname = path.join(
        opt.model_name,
        f'node_features_{chromosome}_{opt.node_representation}.pt')
    if path.exists(node_features_fname):
        return torch.load(node_features_fname)

    # xs[loc] is of shape : [1, 32, 5000]
    if mode == 'average':
        x = xs.mean(axis=2)
    elif mode == 'max':
        x = xs.max(axis=2).values
    elif mode == 'min':
        x = xs.min(axis=2).values
    elif mode == 'min-max':
        x_min = xs.min(axis=2).values
        x_max = xs.max(axis=2).values
        x = torch.cat((x_min, x_max), 1)
    elif mode == 'conv1d':
        x = xs
    elif mode == 'pca':
        #TODO: fix this to handle current xs format
        pca = PCA(n_components=opt.pca_dims)
        x = torch.stack([torch.flatten(torch.tensor(pca.fit_transform(xs[loc][0]),
                                                    dtype=torch.float).squeeze()) for loc in xs])
    elif mode == 'summary':
        x_min = xs.min(axis=2).values
        x_max = xs.max(axis=2).values
        x_avg = xs.mean(axis=2)
        x_median = xs.median(axis=2).values
        x_std = xs.std(axis=2)
        x = torch.cat((x_min, x_max, x_avg, x_median, x_std), 1)
    elif mode == 'zeros':
        x = torch.zeros(xs.shape)
    return x


def save_node_representations(node_representation, chromosome, opt):
    node_features_fname = path.join(
        opt.results_dir,
        f'node_features_{chromosome}_{opt.node_representation}.pt')
    torch.save(node_representation, node_features_fname)


