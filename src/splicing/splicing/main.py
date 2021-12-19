import os
from os import path
import argparse
import warnings
import logging

import numpy as np
import yaml

import h5py
import torch
import torch.nn as nn
from torchsummary import summary
import wandb
import coloredlogs
from collections import OrderedDict

from splicing.utils.config_args import config_args, get_args
from splicing.models.geometric_models import SpliceGraph, FullModel
from splicing.runner import run_model
from splicing.utils import graph_utils
from splicing.utils import wandb_utils
from splicing.utils.utils import CHR2IX

from splicing.models.splice_ai import SpliceAI, SpliceAIEnsemble

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

train_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                     project_config['DATA_PIPELINE']['train_chroms']]
valid_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                     project_config['DATA_PIPELINE']['valid_chroms']]
test_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                    project_config['DATA_PIPELINE']['test_chroms']]


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

        model_fname = f'SpliceAI' \
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


def main(opt):

    # Workflow
    logging.info(f"==> Workflow: {opt.workflow}, Model name: {opt.model_name}")

    # Loading Dataset
    logging.info('==> Loading Data')
    if opt.finetune:
        # features_dir = opt.model_name.split('.finetune')[0]

        datasets = {
            'train': np.asarray(train_chromosomes, dtype=int),
            'valid': np.asarray(valid_chromosomes, dtype=int),
            'test': np.asarray(test_chromosomes, dtype=int),
        }

        opt.full_validation_interval = len(datasets['train'])

        opt.epochs = opt.finetune_epochs * len(datasets['train'])

    else:
        train_data_path = path.join(
            opt.splice_data_root,
            f'dataset_train_all_{opt.window_size}_{opt.context_length}.h5'
        )
        logging.info(f"Importing train data file: {train_data_path}")
        train_data_file = h5py.File(train_data_path, 'r')

        valid_data_file = h5py.File(path.join(
            opt.splice_data_root,
            f'dataset_valid_all_{opt.window_size}_{opt.context_length}.h5'),
            'r')

        test_data_file = h5py.File(path.join(
            opt.splice_data_root, 
            f'dataset_test_0_{opt.window_size}_{opt.context_length}.h5'), 'r')

        opt.full_validation_interval = train_data_file.attrs['n_datasets']

        if opt.save_feats:
            opt.epochs = len(train_chromosomes)
        else:
            opt.epochs = train_data_file.attrs['n_datasets'] * opt.passes

        datasets = {
            'train': train_data_file,
            'valid': valid_data_file,
            'test': test_data_file
        }

    opt.chromosomes = {
        'train': np.asarray(train_chromosomes, dtype=int),
        'valid': np.asarray(valid_chromosomes, dtype=int),
        'test': np.asarray(test_chromosomes, dtype=int),
    }

    # Initialize wandb
    config = wandb_utils.get_wandb_config(opt)

    if opt.wandb:
        if opt.pretrain:
            wandb_project_name = 'dl-project-pretrain'
        elif opt.finetune:
            wandb_project_name = 'dl-project-finetune'
        wandb.init(
            project=wandb_project_name, config=config, name=config.name
        )

    # Initialize Models
    if opt.load_pretrained:
        base_model = load_pretrained_base_model(opt, config)
    else:
        base_model = SpliceAI(
            opt.n_channels,
            opt.kernel_size,
            opt.dilation_rate
        )

    opt.total_num_parameters = int(graph_utils.count_parameters(base_model))

    graph_model, full_model = None, None

    # if fine_tuning, create the SpliceGraph and Full Model
    if opt.finetune:
        graph_model = SpliceGraph(opt)
        full_model = FullModel(opt)

        logging.info(graph_model)
        logging.info(full_model)

    # if finetune and load_gcn
    if opt.finetune and opt.load_gcn:
        gcn_model_path = opt.model_name.replace('.load_gcn', '') + '/model.chkpt'
        logging.info(f'==> Loading Saved GCN {gcn_model_path}')

        checkpoint = torch.load(gcn_model_path)
        graph_model.load_state_dict(checkpoint['model'])

    # if saving feats or finetuning, need to load a base model
    if opt.save_feats or opt.finetune:
        
        if opt.save_feats:
            assert opt.load_pretrained == True, "Have to load pretrained model"

        logging.info(f"==> Turning off base model params for {opt.workflow}")

        for param in base_model.parameters():
            param.requires_grad = False

    # Optimizer and Scheduler
    optimizer, scheduler = None, None

    if opt.pretrain or opt.save_feats:
        # default schedule: 6 epochs at 0.001, then halve lr every epoch
        optimizer = torch.optim.Adam(
            base_model.parameters(), lr=opt.cnn_lr
        )
        step_size_milestones = [(train_data_file.attrs['n_datasets'] * x) + 1 \
            for x in [6, 7, 8, 9, 10]]
        scheduler = torch.torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_size_milestones, 
            gamma=0.5, verbose=False
        )
    elif opt.finetune:
        optimizer = graph_utils.get_combined_optimizer(
            graph_model, full_model, opt
        )
        step_size_milestones = [(len(datasets['train']) * x) + 1
                                for x in list(range(8, opt.epochs))]
        scheduler = torch.torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_size_milestones,
            gamma=0.5, verbose=False
        )
    elif not (opt.test_baseline or opt.test_graph):
        optimizer = graph_utils.get_optimizer(base_model, opt)
        scheduler = torch.torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_step_size, 
            gamma=0.5, verbose=False
        )
    
    logging.info(f"==> Optimizer: {optimizer}")

    # optimizer = graph_utils.get_optimizer(graph_model, opt)

    criterion = graph_utils.get_criterion(opt)

    if torch.cuda.device_count() > 0 and opt.pretrain:
        logging.info(f'Using {torch.cuda.device_count()} GPUs!')
        base_model = nn.DataParallel(base_model)

    if torch.cuda.is_available() and opt.cuda:
        criterion = criterion.cuda()
        base_model = base_model.cuda()
        if graph_model is not None:
            graph_model = graph_model.cuda()
        if full_model is not None:
            full_model = full_model.cuda()

    logging.info(f'Model name: {opt.model_name}')

    try:
        run_model(base_model, graph_model, full_model, datasets, criterion,
                  optimizer, scheduler, opt, logger=None)
    except KeyboardInterrupt:
        logging.info('-' * 89 + '\nManual Exit')
        exit()


if __name__ == '__main__':
    main(opt)
