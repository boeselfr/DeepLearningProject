import os
from os import path
import argparse
import warnings
import logging
import yaml

import numpy as np
import h5py
import torch
import torch.nn as nn
from torchsummary import summary
import wandb
import coloredlogs

from splicing.utils.config_args import config_args, get_args
# from splicing.models.graph_models import SpliceGraph
from splicing.models.geometric_models import SpliceGraph, FullModel
from runner import run_model
from splicing.utils import graph_utils
from splicing.utils.utils import CHR2IX

from splicing.models.splice_ai import SpliceAI

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------
# Loading Config
# ----------------------------------------------------------------
with open("config.yaml", "r") as stream:
    try:
        project_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args, project_config)

# TODO
train_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                     project_config['DATA_PIPELINE']['train_chroms']][:1]
test_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                    project_config['DATA_PIPELINE']['test_chroms']][:1]
valid_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                     project_config['DATA_PIPELINE']['test_chroms']][:1]


def main(opt):

    # Loading Dataset
    logging.info(opt.model_name)

    logging.info('==> Loading Data')
    if not opt.pretrain and not opt.save_feats:
        features_dir = opt.model_name.split('.finetune')[0]

        datasets = {
            'train': {
                chrom: torch.load(
                    path.join(features_dir,
                              f'chrom_feature_dict_train_chr{chrom}.pt'))
                for chrom in train_chromosomes},
            'valid': {
                chrom: torch.load(
                    path.join(features_dir,
                              f'chrom_feature_dict_test_chr{chrom}.pt'))
                for chrom in valid_chromosomes},
            'test': {
                chrom: torch.load(
                    path.join(features_dir,
                              f'chrom_feature_dict_test_chr{chrom}.pt'))
                for chrom in test_chromosomes},
        }

        opt.epochs = opt.finetune_epochs
    else:

        train_data_file = h5py.File(path.join(
            opt.splice_data_root,
            f'dataset_train_all_{opt.window_size}.h5'), 'r')

        test_data_file = h5py.File(path.join(
            opt.splice_data_root, f'dataset_test_0_{opt.window_size}.h5'), 'r')

        num_idx = train_data_file.attrs['n_datasets']
        idx_all = np.random.permutation(num_idx)

        opt.idxs = {
            'train': idx_all[:int(opt.train_ratio * num_idx)],
            'valid': idx_all[int(opt.train_ratio * num_idx):],
            'test': list(range(test_data_file.attrs['n_datasets']))
        }

        if opt.save_feats:
            opt.epochs = len(opt.idxs['train'])
        else:
            opt.epochs = len(opt.idxs['train']) * opt.passes

        datasets = {
            'train': train_data_file,
            'valid': train_data_file,
            'test': test_data_file
        }

    logging.info('==> Creating window_model')

    config = graph_utils.get_wandb_config(opt)

    base_model = SpliceAI(
        config.n_channels, config.kernel_size, config.dilation_rate)

    if opt.wandb:
        wandb.init(project='dl-project', config=config)

    # TODO: I think using non-strand specific is not necessary
    opt.total_num_parameters = int(graph_utils.count_parameters(base_model))
    logging.info('>>>>>>>>>> BASE MODEL <<<<<<<<<<<')
    logging.info('Total number of parameters in the base model: '
                 f'{opt.total_num_parameters}.')
    # summary(base_model,
    #         input_size=(4, opt.context_length + opt.window_size),
    #         device='cuda')

    optimizer = graph_utils.get_optimizer(base_model, opt)

    graph_model, full_model = None, None
    if not opt.pretrain:
        # Creating GNNModel

        if not opt.save_feats:
            # graph_model = SpliceGraph(
            #     32, opt.hidden_size, opt.gcn_dropout, opt.gate, opt.gcn_layers)
            graph_model = SpliceGraph(
                config.n_channels, opt.hidden_size, opt.gcn_dropout)
            full_model = FullModel(opt.n_channels)

            logging.info(graph_model)
            logging.info(full_model)
            # summary(full_model,
            #         input_size=(2 * opt.n_channels,
            #                     opt.context_length + opt.window_size))

        if opt.load_gcn:
            logging.info('Loading Saved GCN')
            checkpoint = torch.load(
                opt.model_name.replace('.load_gcn', '') + '/model.chkpt')
            graph_model.load_state_dict(checkpoint['model'])
        else:
            # Initialize GCN output layer with window_model output layer
            logging.info('Loading Saved base_model')

            model_fname = f'SpliceAI' \
                          f'_e1190' \
                          f'_cl{opt.context_length}' \
                          f'_g{opt.model_index}.h5'

            checkpoint = torch.load(path.join(
                opt.model_name.split('.finetune')[0], model_fname))

            for param in base_model.parameters():
                param.requires_grad = False

            base_model = nn.DataParallel(base_model)
            base_model.load_state_dict(checkpoint['model'])

            if opt.cuda:
                base_model = base_model.cuda()

        optimizer = graph_utils.get_combined_optimizer(
            graph_model, full_model, opt)

            # graph_model.out.weight.data = \
            #     base_model.module.model.classifier.weight.data
            # graph_model.out.bias.data = \
            #     base_model.module.model.classifier.bias.data
            # graph_model.batch_norm.weight.data = \
            #     base_model.module.model.batch_norm.weight.data
            # graph_model.batch_norm.bias.data = \
            #     base_model.module.model.batch_norm.bias.data

        # optimizer = graph_utils.get_optimizer(graph_model, opt)

    scheduler = torch.torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.lr_step_size, gamma=0.5)
    logging.info(optimizer)

    criterion = graph_utils.get_criterion(opt)

    if torch.cuda.device_count() > 0 and opt.pretrain:
        logging.info(f'Using {torch.cuda.device_count()} GPUs!')
        base_model = nn.DataParallel(base_model)

    if torch.cuda.is_available() and opt.cuda:
        criterion = criterion.cuda()
        if opt.pretrain:
            base_model = base_model.cuda()
        if graph_model is not None:
            graph_model = graph_model.cuda()
        if opt.gpu_id != -1:
            torch.cuda.set_device(opt.gpu_id)

    logging.info(f'Model name: {opt.model_name}')
    # logger = Logger(opt)

    try:
        run_model(base_model, graph_model, full_model, datasets, criterion,
                  optimizer, scheduler, opt, logger=None)
    except KeyboardInterrupt:
        logging.info('-' * 89 + '\nManual Exit')
        exit()


if __name__ == '__main__':
    main(opt)
