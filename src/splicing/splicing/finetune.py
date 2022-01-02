import os.path as path
import pickle
import logging

import math
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from splicing.utils.general_utils import SPLIT2DESC, IX2CHR, \
    compute_scores, compute_average_scores
from splicing.utils.graph_utils import process_graph
from splicing.utils.wandb_utils import report_wandb, analyze_gradients


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

    total_loss = 0
    batch_count = 0

    scores = {}

    # bin dict has information for all chromosomes
    # that's why we can load it here before:
    # we need it also for the constant case, thus always has to be loaded
    bin_dict_file = path.join(
        opt.graph_data_root,
        f'{opt.window_size}_test_val_train_bin_dict_{opt.hicsize}_{opt.hicnorm}norm.pkl'
    )
    bin_dict = pickle.load(open(bin_dict_file, "rb"))

    if opt.adj_type in ['hic', 'both']:
        graph_file = path.join(
            opt.graph_data_root,
            f"{opt.window_size}_{split}_graphs_{opt.hicsize}_{opt.hicnorm}norm.pkl")
        split_adj_dict = pickle.load(open(graph_file, "rb"))
    else:
        split_adj_dict = None

    for chromosome in chromosomes:

        chromosome_data = torch.load(
            path.join(
                opt.model_name.split('/finetune')[0],
                f'chrom_feature_dict_{split}_chr{chromosome}.pt'
            ), map_location=torch.device('cpu')
        )

        xs = chromosome_data['x']
        ys = chromosome_data['y']

        xs = torch.stack([(xs[loc][0]) for loc in xs])
        ys = torch.stack([(ys[loc][0]) for loc in ys])

        d1, d2, d3 = ys.shape
        all_preds = torch.zeros((d1 - d1 % opt.graph_batch_size, d2, d3)).cpu()
        all_targets = torch.zeros((d1 - d1 % opt.graph_batch_size, d2, d3)).cpu()
        count_ys = 0

        graph = process_graph(
            opt.adj_type, split_adj_dict, len(xs),
            IX2CHR(chromosome), bin_dict, opt.window_size
        )
        graph_data = Data(
            x=xs,
            y=ys,
            edge_index=graph.coalesce().indices()
        )

        graph_loader = NeighborLoader(
            graph_data, 
            num_neighbors=[-1],
            batch_size=opt.graph_batch_size,
            drop_last=True
        )

        desc_i = f'({str.upper(SPLIT2DESC[split])} on chromosome {chromosome})'
        logging.info(f'{desc_i} - batch size {opt.graph_batch_size}, nbatches '
                    f' {len(graph_loader)}')

        # a, f = get_gpu_stats()

        for batch, graph_batch in tqdm(enumerate(graph_loader), leave=False,
                                    total=len(graph_loader), desc=desc_i):

            _x = graph_batch['x'].to('cuda')
            _y = graph_batch['y'].to('cuda')
            #_y = graph_batch['y'].to('cuda' if split != 'test' else 'cpu')
            _edge_index = graph_batch['edge_index'].to('cuda')

            # a, f = get_gpu_stats()

            if opt.boost_period > 1:
                if batch_count % opt.boost_period != 0:
                    # only update the full model every boost_period epochs
                    for param in full_model.parameters():
                        param.requires_grad = False
                else:
                    for param in full_model.parameters():
                        param.requires_grad = True

            node_representation = graph_model(_x, _edge_index, opt)
            
            if epoch <= opt.node_headstart * len(chromosomes):
                _x = torch.zeros(_x.shape).to('cuda')

            _x.requires_grad = opt.ingrad
            # node_representation.requires_grad = opt.ingrad

            _y_hat = full_model(_x, node_representation)

            loss = criterion(_y_hat, _y)

            #a, f = get_gpu_stats()

            if split == 'train':
                if opt.ingrad:
                    node_representation.retain_grad()
                loss.backward()
                optimizer.step()

                if batch_count % opt.log_interval == 0 and opt.wandb:
                    analyze_gradients(
                        graph_model, full_model, _x, node_representation, opt
                    )

                optimizer.zero_grad()

            if split != 'train':
                total_loss += loss.sum().item()
                #all_preds[batch * opt.graph_batch_size:(batch+1) * opt.graph_batch_size] = torch.cat((all_preds, _y_hat.cpu().data), 0)
                all_preds[count_ys:count_ys + opt.graph_batch_size] = _y_hat[:opt.graph_batch_size].cpu().data
                #all_targets[] = torch.cat((all_targets, _y.cpu().data), 0)
                all_targets[count_ys:count_ys + opt.graph_batch_size] = _y[:opt.graph_batch_size].cpu().data
                # count_ys += _y_hat.shape[0]
                count_ys += opt.graph_batch_size

            # wandb reporting
            if split == 'train' and batch_count % opt.log_interval == 0 and opt.wandb:
                report_wandb(_y_hat, _y, loss, opt, split, step=batch_count)
            elif split == 'valid' and batch_count % opt.validation_interval == 0 and opt.wandb:
                report_wandb(_y_hat, _y, loss, opt, split, step=batch_count)

            batch_count+=1

        if split in ["valid", "test"]:
            scores[chromosome] = compute_scores(
                all_preds.numpy(), 
                all_targets.numpy(),
                total_loss,
                opt.wandb,
                epoch,
                split,
                chromosome
            )

            all_preds = torch.Tensor().cpu()
            all_targets = torch.Tensor().cpu()
            total_loss = 0

    if split in ["valid", "test"]:
        combined_scores = compute_average_scores(
            scores, opt.wandb, split
        )
    else:
        combined_scores = {}

        # TODO: check node reps before and after
        # if epoch == opt.finetune_epochs:
        #     save_node_representations(graph_data.x, chromosome, opt)

    return combined_scores
