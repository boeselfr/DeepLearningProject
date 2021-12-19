import os.path as path
import pickle
import logging

import numpy as np
import math
from tqdm import tqdm
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from splicing.utils.general_utils import SPLIT2DESC, IX2CHR
from splicing.utils.graph_utils import process_graph, \
    build_node_representations, save_node_representations
from splicing.utils.wandb_utils import report_wandb, analyze_gradients

from splicing.data_models.splice_dataset import ChromosomeDataset


def get_gpu_stats():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return a, f


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

    chromosome_data = torch.load(
        path.join(
            opt.model_name.split('/finetune')[0],
            f'chrom_feature_dict_{split}_chr{chromosome}.pt'
        )
    )

    xs = chromosome_data['x']
    ys = chromosome_data['y']

    #chromosome_dataset = ChromosomeDataset(xs, ys)
    #dataloader = DataLoader(
    #    chromosome_dataset, batch_size=opt.graph_batch_size, shuffle=False)

    nodes = build_node_representations(
        xs, opt.node_representation, opt
    )
    # nodes.requires_grad = True
    ys = torch.stack([(ys[loc][0]) for loc in ys])

    graph = process_graph(
        opt.adj_type, split_adj_dict, len(nodes),
        IX2CHR(chromosome), bin_dict, opt.window_size
    )
    graph_data = Data(
        x=nodes,
        y = ys,
        edge_index=graph.coalesce().indices()
    )

    graph_loader = NeighborLoader(
        graph_data, 
        num_neighbors=[-1],
        batch_size = opt.graph_batch_size
    )

    desc_i = f'({str.upper(SPLIT2DESC[split])} on chromosome {chromosome})'
    logging.info(f'{desc_i} - batch size {opt.graph_batch_size}, nbatches '
                 f' {len(graph_loader)}')

    #a, f = get_gpu_stats()

    analyze_grad_interval = math.ceil(len(graph_loader) / 100)

    for batch, graph_batch in tqdm(enumerate(graph_loader), leave=False,
                                total=len(graph_loader), desc=desc_i):

        _x = graph_batch['x'].to('cuda')
        _y = graph_batch['y'].to('cuda')
        _edge_index = graph_batch['edge_index'].to('cuda')

        #_x.requires_grad = True
        #a, f = get_gpu_stats()
        node_representation = graph_model(_x, _edge_index)

        #l_ix = opt.graph_batch_size * batch
        #u_ix = opt.graph_batch_size * (batch + 1)

        #batch_node_representation = node_representation[l_ix: u_ix, :]
        #_y_hat = full_model(_x, batch_node_representation)
        _y_hat = full_model(_x, node_representation)

        loss = criterion(_y_hat, _y)

        #a, f = get_gpu_stats()

        if split == 'train':
            loss.backward()
            optimizer.step()

            if batch % 100 == 0 and opt.wandb:
                #start = time.time()
                analyze_gradients(
                    graph_model, full_model, _x, node_representation, opt
                )
                #end = time.time()
                #logging.info(f"gradient_reporting: {end-start}")
        

        total_loss += loss.sum().item()
        all_preds = torch.cat((all_preds, _y_hat.cpu().data), 0)
        all_targets = torch.cat((all_targets, _y.cpu().data), 0)

        # wandb reporting
        if split == 'train':
            optimizer.zero_grad()
            if batch % 25 == 0 and opt.wandb:
                report_wandb(_y_hat, _y, loss, opt, split, step=batch)
        else:
            report_wandb(_y_hat, _y, loss, opt, split, step=batch)

    if epoch == opt.finetune_epochs:
        save_node_representations(graph_data.x, chromosome, opt)

    return all_preds, all_targets, total_loss
