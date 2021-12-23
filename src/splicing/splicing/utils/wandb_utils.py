import os
from os import path
import logging

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score
import wandb
import math



def get_wandb_config(opt):
    config = wandb.config

    if opt.wandb_name:
        config.name = opt.wandb_name
    else:
        config.name = opt.model_id
    
    # pretrain args
    config.context_length = opt.context_length
    config.kernel_size = opt.kernel_size
    config.dilation_rate = opt.dilation_rate
    config.batch_size = opt.batch_size
    config.class_weights = opt.class_weights
    config.cnn_lr = opt.cnn_lr

    #finetune args
    config.graph_batch_size = opt.graph_batch_size
    config.finetune_epochs = opt.finetune_epochs
    config.ft_optim = opt.ft_optim
    config.nr_lr = opt.nr_lr
    config.gcn_lr = opt.gcn_lr
    config.full_lr = opt.full_lr
    config.ft_sched = opt.ft_sched
    config.rlr_factor = opt.rlr_factor

    config.n_channels = opt.n_channels
    config.hidden_size = opt.hidden_size
    config.hidden_size_full = opt.hidden_size_full
    config.node_representation = opt.node_representation
    config.adj_type = opt.adj_type

    config.gcn_dropout = opt.gcn_dropout
    config.nr_model = opt.nr_model
    config.zero_nuc = opt.zeronuc
    config.boost_graph = opt.boost_graph
    config.boost_period = opt.boost_period

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
    

def analyze_gradients(graph_model, full_model, _x, nodes, opt):
    log_message = {}
    try:
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
    except Exception as e:
        # needed for boost_graph option
        pass

    try:
        for jj, param in enumerate(full_model.named_parameters()):
            #if jj == 2:
            #    continue
            param_name = param[0]
            param_data = param[1].data
            param_grad = param[1].grad.data

            m = param_data.shape[1] if len(param_data.shape) > 1 else 1

            log_message[f'full_grad/{param_name}'] = np.linalg.norm(
                param_grad.detach().cpu().numpy()) / m
            log_message[f'full_weight/{param_name}'] = np.linalg.norm(
                param_data.detach().cpu().numpy()) / m
    except Exception as e:
        # sry for the code
        pass

    for jj, param in enumerate(graph_model.named_parameters()):
        param_name = param[0]
        try:
            param_data = param[1].data
            param_grad = param[1].grad.data
        except AttributeError:
            continue
        else:
            m = param_data.shape[1] if len(param_data.shape) > 1 else 1

            log_message[f'graph_grad/{param_name}'] = np.linalg.norm(
                param_grad.detach().cpu().numpy()) / m
            log_message[f'graph_weight/{param_name}'] = np.linalg.norm(
                param_data.detach().cpu().numpy()) / m

    wandb.log(log_message)

