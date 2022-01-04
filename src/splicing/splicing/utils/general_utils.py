import os
from os import path
import logging

from math import floor
import numpy as np
import wandb
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from collections import OrderedDict

from splicing.models.losses import CategoricalCrossEntropy2d
from splicing.models.splice_ai import SpliceAI, SpliceAIEnsemble

# Maps etc.
SPLIT2DESC = {
    'train': 'Train',
    'valid': 'Validation',
    'test': 'Test',
}

CHR2IX = lambda chr: 23 if chr == 'X' else 24 if chr == 'Y' else int(chr)
IX2CHR = lambda ix: 'chrX' if ix == 23 else 'chrY' \
    if ix == 24 else f'chr{str(ix)}'

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


# Losses, Metrics, etc.
def get_criterion(opt):
    return CategoricalCrossEntropy2d(opt.class_weights)


def compute_scores(predictions, targets, loss, log_wandb, step, split, chromosome):
    """ compute the evaluation scores for a single chromosome """
    is_expr = targets.sum(axis=(1, 2)) >= 1

    chromosome = IX2CHR(chromosome)

    scores = {
        'loss': loss,
        'n_obs': len(targets)
    }
    for ix, prediction_type in enumerate(['Acceptor', 'Donor']):
        y_true = targets[is_expr, ix + 1, :].flatten()
        y_pred = predictions[is_expr, ix + 1, :].flatten()

        idx_true = np.nonzero(y_true == 1)[0]
        argsorted_y_pred = np.argsort(y_pred)
        # sorted_y_pred = np.sort(y_pred)

        topkl_accuracy = []
        # threshold = []

        for top_length in [0.5, 1, 2, 4]:
            idx_pred = argsorted_y_pred[-int(top_length * len(idx_true)):]

            topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred))
                               / (float(min(len(idx_pred), len(idx_true))) + 1e-6)]
            # threshold += [sorted_y_pred[-int(top_length * len(idx_true))]]

        auprc = average_precision_score(y_true, y_pred)

        scores[f"{prediction_type}_auprc"] = auprc
        scores[f"{prediction_type}_topk_0.5"] = topkl_accuracy[0]
        scores[f"{prediction_type}_topk_1"] = topkl_accuracy[1]
        scores[f"{prediction_type}_topk_2"] = topkl_accuracy[2]
        scores[f"{prediction_type}_topk_4"] = topkl_accuracy[3]

        if log_wandb:
            wandb.log({
                f'{split}/{chromosome} Test Loss - {prediction_type}': loss,
                f'{split}/{chromosome} AUPRC - {prediction_type}': auprc,
                f'{split}/{chromosome} Top-K Accuracy: {prediction_type}': topkl_accuracy[1],
                # f'{split}/Thresholds for K: {prediction_type}': threshold[1],
                # f'{split}/Proportion of True Splice Sites Predicted'
                # f': {prediction_type}': no_positive_predictions / len(idx_true),
            })

    return scores


def compute_average_scores(chrom_scores, log_wandb, split):
    """ average the scores across chromosomes """
    all_scores = {}
    for chrom, scores in chrom_scores.items():
        for score, value in scores.items():
            score_list = all_scores.get(score, [])
            score_list.append(value)
            all_scores[score] = score_list
    combined_scores = {}
    for score, values in all_scores.items():
        if score in ["loss", "n_obs"]:
            combined_scores[score] = np.sum(values)
        else:
            combined_scores[score] = np.mean(values)
    combined_scores['avg_loss'] = combined_scores['loss'] / combined_scores['n_obs']
    combined_scores['avg_auprc'] = np.mean([combined_scores['Acceptor_auprc'], combined_scores['Donor_auprc']])
    combined_scores['avg_topk_1'] = np.mean([combined_scores['Acceptor_topk_1'], combined_scores['Donor_topk_1']])
    if log_wandb:
        wandb.log({
                f'{split}/aggregated/Total Test Loss': combined_scores['loss'],
                f'{split}/aggregated/Average Test Loss': combined_scores['avg_loss'],
                f'{split}/aggregated/AUPRC - Acceptor': combined_scores['Acceptor_auprc'],
                f'{split}/aggregated/AUPRC - Donor': combined_scores['Donor_auprc'],
                f'{split}/aggregated/AUPRC - Average': combined_scores['avg_auprc'],
                f'{split}/aggregated/Top-K Accuracy - Acceptor:': combined_scores['Acceptor_topk_1'],
                f'{split}/aggregated/Top-K Accuracy - Donor:': combined_scores['Donor_topk_1'],
                f'{split}/aggregated/Top-K Accuracy - Average:': combined_scores['avg_topk_1']
            })

    logging.info(f'------------------------>>>> {split} <<<<------------------------')
    logging.info(f'Total Loss: {combined_scores["loss"]}')
    logging.info(f'Top-K Accuracy: {combined_scores["avg_topk_1"]}')
    logging.info(f'AUPRC: {combined_scores["avg_auprc"]:.6f}')

    return combined_scores


# Training functions
def shuffle_chromosomes(datasets):
    for key in ['train', 'test', 'valid']:
        datasets[key] = datasets[key][
            np.random.permutation(len(datasets[key]))]

    return datasets


def get_optimizer(graph_model, full_model, opt):

    nr_params = []
    gcn_params = []
    other_graph_params = []
    for name, param in graph_model.named_parameters():
        if name.startswith("nr_"):
            nr_params.append(param)
        elif name.startswith("gcn_"):
            gcn_params.append(param)
        else:
            other_graph_params.append(param)

    if opt.ft_optim == 'adam':
        optimizer = torch.optim.Adam(
            [
                {'params': nr_params, 'lr': opt.nr_lr},
                {'params': gcn_params, 'lr': opt.gcn_lr},
                {'params': other_graph_params, 'lr': opt.gcn_lr},
                {'params': list(full_model.parameters()), 'lr': opt.full_lr}
            ], betas=(0.9, 0.98), lr=opt.gcn_lr
        )
    elif opt.ft_optim == 'sgd':
        optimizer = torch.optim.SGD(
            [
                {'params': nr_params, 'lr': opt.nr_lr},
                {'params': gcn_params, 'lr': opt.gcn_lr},
                {'params': other_graph_params, 'lr': opt.gcn_lr},
                {'params': list(full_model.parameters()), 'lr': opt.full_lr}
            ], lr=opt.gcn_lr, weight_decay=opt.weight_decay, momentum=0.9)
    return optimizer


def get_scheduler(opt, optimizer, len_datasets):
    if opt.ft_sched == "multisteplr":
        step_size_milestones = [list(range(4, opt.epochs + 1))]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_size_milestones,
            gamma=0.5, verbose=False
        )
    elif opt.ft_sched == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_step_size,
            gamma=opt.lr_decay, verbose=False
        )
    elif opt.ft_sched == "reducelr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=opt.rlr_factor,
            patience=opt.rlr_patience,
            threshold=opt.rlr_threshold,
            verbose=True
        )
    else:
        scheduler = None
    return scheduler


# Model loading, model utils
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_conv1d_lout(l_in, padding=0, dilation=1, kernel_size=1, stride=1):
    return floor((l_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def load_base_checkpoint(base_model, checkpoint_path):
    """ load the saved checkpoint of a base CNN model """
    logging.info(f'==> Loading saved base_model {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path)

    base_model = nn.DataParallel(base_model)

    try:
        base_model.load_state_dict(checkpoint['model'])
    except RuntimeError:
        logging.info(f'==> Applying DataParrallel Fix')

        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        base_model.load_state_dict(new_state_dict)


def load_pretrained_base_model(opt, config):
    """ get a pre-trainted base CNN model """
    if opt.test_baseline:

        base_models = []

        model_fnames = [
            fname for fname in os.listdir(opt.test_baseline_models_dir)
            if fname[-3:] == '.h5']

        for model_fname in model_fnames:

            base_model = SpliceAI(
                config.n_channels,
                config.kernel_size,
                config.dilation_rate
            ).to('cuda')

            checkpoint_path = path.join(
                opt.test_baseline_models_dir, model_fname)

            load_base_checkpoint(base_model, checkpoint_path)

            base_model.eval()

            base_models.append(base_model)

        base_model = SpliceAIEnsemble(base_models, opt.window_size)

    else:

        base_model = SpliceAI(
            config.n_channels,
            config.kernel_size,
            config.dilation_rate
        )

        model_fname = f'base' \
            f'_e{opt.model_iteration}' \
            f'_cl{opt.context_length}' \
            f'_g{opt.model_index}.h5'

        # in case you're loading a pretrained model in a finetune
        # workflow, the main folder name is obtained by
        # cutting before finetune bit.
        checkpoint_path = path.join(
            opt.model_name.split('/finetune')[0],
            model_fname
        )
        load_base_checkpoint(base_model, checkpoint_path)

    return base_model


def load_finetuned_checkpoint(graph_model, checkpoint_path, splice_ai_device):
    """ load the saved checkpoint of a graph model """

    logging.info(f'==> Loading saved graph_model {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path,
                            map_location=torch.device(splice_ai_device))

    graph_model.load_state_dict(checkpoint['model'])


def load_pretrained_graph_model(opt, config):
    """ get a pre-trainted graph full model """
    from splicing.models.geometric_models import SpliceGraph, FullModel
    from splicing.models.geometric_models import SpliceGraphEnsemble, \
        FullModelEnsemble

    graph_models, full_models = [], []

    graph_model_fnames = [
        fname for fname in sorted(os.listdir(
            path.join(opt.test_graph_models_dir, 'graph_models')))
        if fname[-3:] == '.h5']
    full_model_fnames = [
        fname for fname in sorted(os.listdir(
            path.join(opt.test_graph_models_dir, 'full_models')))
        if fname[-3:] == '.h5']

    for graph_model_fname, full_model_fname in zip(
            graph_model_fnames, full_model_fnames):

        graph_model = SpliceGraph(opt).to('cpu')
        full_model = FullModel(opt, device='cuda').to('cpu')

        graph_checkpoint_path = path.join(
            opt.test_graph_models_dir, 'graph_models', graph_model_fname)

        full_checkpoint_path = path.join(
            opt.test_graph_models_dir, 'full_models', full_model_fname)

        load_finetuned_checkpoint(
            graph_model, graph_checkpoint_path, 'cpu')
        load_finetuned_checkpoint(
            full_model, full_checkpoint_path, 'cpu')

        graph_model.eval()
        full_model.eval()

        graph_models.append(graph_model)
        full_models.append(full_model)

    graph_model = SpliceGraphEnsemble(graph_models)
    full_model = FullModelEnsemble(full_models, opt.window_size)

    return graph_model.cuda(), full_model.cuda()


def save_model(opt, epoch, model, model_type='base'):
    """ save a model checkpoint """
    model_suffix = f'{model_type}' \
                   f'_e{epoch}' \
                   f'_cl{opt.context_length}' \
                   f'_g{opt.model_index}.h5'

    checkpoint = {'model': model.state_dict(), 'settings': opt, 'epoch': epoch}

    directory_setup(opt.model_name)
    model_fname = path.join(opt.model_name, model_suffix)
    logging.info(f'Saving model {model_fname}')
    torch.save(checkpoint, model_fname)


def save_feats(model_name, split, Y, locations, X, chromosome):
    """ Saves the nucleotide representations and targets so that these can
        later be combined into window representations.
        Each chromosome gets its own file. """
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

    # save per location
    for location in location_index_dict:
        chrom_indices = torch.Tensor(location_index_dict[location]).long()
        x = torch.index_select(X, 0, chrom_indices)
        y = torch.index_select(Y, 0, chrom_indices)
        location_feature_dict['x'][location] = x
        location_feature_dict['y'][location] = y
    torch.save(location_feature_dict, data_fname)
    logging.info(f"Exporting {data_fname}")
