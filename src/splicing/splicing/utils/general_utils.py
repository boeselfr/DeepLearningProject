import os
from os import path
import logging

SPLIT2DESC = {
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


def compute_conv1d_lout(l_in, dilation=1, kernel_size=1, stride=1):
    return math.floor((l_in - dilation * (kernel_size - 1) - 1) / stride + 1)


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



