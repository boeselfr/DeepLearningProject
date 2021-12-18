import os.path as path
import pickle
import logging

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from splicing.utils.graph_utils import process_graph, split2desc, \
    build_node_representations, save_node_representations, report_wandb, \
    analyze_gradients
from splicing.data_models.splice_dataset import ChromosomeDataset
from splicing.utils.utils import IX2CHR


def finetune(graph_model, full_model, chromosomes, criterion, optimizer,
             epoch, opt, split):
    if split == 'train':
        graph_model.train()
    else:
        graph_model.eval()

    all_preds = torch.Tensor().cpu()
    all_targets = torch.Tensor().cpu()

    total_loss = 0

    # bin dict has information for all chromosomes
    # that's why we can load it here before:
    # we need it also for the constant case, thus always has to be loaded
    bin_dict_file = path.join(
        opt.graph_data_root,
        f'test_val_train_bin_dict_{opt.hicsize}_{opt.hicnorm}norm.pkl'
    )
    bin_dict = pickle.load(open(bin_dict_file, "rb"))

    if opt.adj_type in ['hic', 'both']:
        graph_file = path.join(
            opt.graph_data_root,
            split + '_graphs_' + opt.hicsize + '_' + opt.hicnorm + 'norm.pkl')
        split_adj_dict = pickle.load(open(graph_file, "rb"))
    else:
        split_adj_dict = None

    chromosome = chromosomes[epoch % len(chromosomes)]

    # tqdm.write(f'Reading in chromosome {chromosome} data.')
    chromosome_data = torch.load(
        path.join(
            opt.model_name.split('/finetune')[0],
            f'chrom_feature_dict_{split}_chr{chromosome}.pt'
        )
    )

    xs = chromosome_data['x']
    ys = chromosome_data['y']

    chromosome_dataset = ChromosomeDataset(xs, ys)
    dataloader = DataLoader(
        chromosome_dataset, batch_size=opt.graph_batch_size, shuffle=False)

    nodes = build_node_representations(
        xs, opt.node_representation, opt
    )
    # nodes.requires_grad = True

    graph = process_graph(
        opt.adj_type, split_adj_dict, len(nodes),
        IX2CHR(chromosome), bin_dict, opt.window_size
    ).cuda() #TODO: is this .cuda() doing anything?
    graph_data = Data(
        x=nodes,
        edge_index=graph.coalesce().indices()
    ).cuda()
    #if split == 'train':
        #graph_data.x.requires_grad = True

    desc_i = f'({split2desc[split]} on chromosome {chromosome})'
    logging.info(f'Number of batches of size {opt.graph_batch_size}:'
                 f' {len(dataloader)}')

    # rep_optimizer = torch.optim.Adam(
    #     [graph_data.x], betas=(0.9, 0.98), lr=opt.gcn_lr)

    for batch, (_x, _y) in tqdm(enumerate(dataloader), leave=False,
                                total=len(dataloader), desc=desc_i):

        #_x.requires_grad = True
        node_representation = graph_model(graph_data)

        l_ix = opt.graph_batch_size * batch
        u_ix = opt.graph_batch_size * (batch + 1)

        batch_node_representation = node_representation[l_ix: u_ix, :]
        _y_hat = full_model(_x, batch_node_representation)

        loss = criterion(_y_hat, _y)

        if split == 'train':
            loss.backward()
            optimizer.step()
            # rep_optimizer.step()

            if batch % 8 == 0:
                analyze_gradients(
                    graph_model, full_model, _x, graph_data.x, opt)

        total_loss += loss.sum().item()
        all_preds = torch.cat((all_preds, _y_hat.cpu().data), 0)
        all_targets = torch.cat((all_targets, _y.cpu().data), 0)

        report_wandb(_y_hat, _y, loss, opt, split, step=batch)

        if split == 'train':

            optimizer.zero_grad()
            # rep_optimizer.zero_grad()

    if epoch == opt.finetune_epochs:
        save_node_representations(graph_data.x, chromosome, opt)

    return all_preds, all_targets, total_loss
