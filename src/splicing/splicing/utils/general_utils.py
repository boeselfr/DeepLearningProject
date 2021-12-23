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

def print_topl_statistics(
        y_true, y_pred, loss, prediction_type, log_wandb, step, split):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

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

    no_positive_predictions = len(np.nonzero(y_pred > 0.5)[0])
    logging.info('Top-K Accuracy')
    logging.info('|0.5\t|1\t|2\t|4\t|')
    logging.info('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|'.format(
        topkl_accuracy[0], topkl_accuracy[1],
        topkl_accuracy[2], topkl_accuracy[3]))
    # logging.info('Thresholds for K')
    # logging.info('|0.5\t|1\t|2\t|4\t|')
    # logging.info('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|'.format(
    #     threshold[0], threshold[1], threshold[2], threshold[3]))
    logging.info(f'AUPRC: {auprc:.6f}')
    logging.info(f'# True Splice Sites: {len(idx_true)} / {len(y_true)}')
    logging.info('# Predicted Splice Sites: '
                 f'{no_positive_predictions} / {len(y_pred)}')
    if log_wandb:
        wandb.log({
            f'{split}/Test Loss: {prediction_type}': loss,
            f'{split}/AUPRC: {prediction_type}': auprc,
            f'{split}/Top-K Accuracy: {prediction_type}': topkl_accuracy[1],
            # f'{split}/Thresholds for K: {prediction_type}': threshold[1],
            # f'{split}/Proportion of True Splice Sites Predicted'
            # f': {prediction_type}': no_positive_predictions / len(idx_true),
        })

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
        step_size_milestones = [(len_datasets * x) + 1
                                for x in list(range(8, opt.epochs))]
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
            mode='max',
            factor=opt.rlr_factor,
            patience=opt.rlr_patience,
            threshold=opt.rlr_threshold
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

    if opt.test_baseline:

        base_models = []

        model_fnames = [
            fname for fname in os.listdir(opt.test_models_dir)
            if fname[-3:] == '.h5']

        for model_fname in model_fnames:

            base_model = SpliceAI(
                config.n_channels,
                config.kernel_size,
                config.dilation_rate
            )

            checkpoint_path = path.join(opt.test_models_dir, model_fname)

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

        model_fname = f'SpliceAI_base' \
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



