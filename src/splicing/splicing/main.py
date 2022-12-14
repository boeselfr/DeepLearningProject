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
import wandb
import coloredlogs


from splicing.utils.config_args import config_args, get_args

from splicing.utils.wandb_utils import get_wandb_config
from splicing.utils.general_utils import CHR2IX, count_parameters, \
    get_optimizer, get_scheduler, get_criterion, \
    load_pretrained_base_model, load_pretrained_graph_model

from splicing.models.splice_ai import SpliceAI
from splicing.models.geometric_models import SpliceGraph, FullModel
from splicing.runner import run_model

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

# ----------------------------------------------------------------
# Loading Chromosome Split
# ----------------------------------------------------------------
TRAIN_CHROMOSOMES = [CHR2IX(chrom[3:]) for chrom in
                     project_config['DATA_PIPELINE']['train_chroms']]
VALID_CHROMOSOMES = [CHR2IX(chrom[3:]) for chrom in
                     project_config['DATA_PIPELINE']['valid_chroms']]
TEST_CHROMOSOMES = [CHR2IX(chrom[3:]) for chrom in
                    project_config['DATA_PIPELINE']['test_chroms']]


def main(opt):

    # Workflow
    logging.info(f"==> Workflow: {opt.workflow}, Model name: {opt.model_name}")

    # Loading Dataset
    logging.info('==> Loading Data')
    if opt.finetune or opt.test_graph:
        # Load just the chromosome numbers if training graph.
        # Window features will be read in later.
        datasets = {
            'train': np.asarray(TRAIN_CHROMOSOMES, dtype=int),
            'valid': np.asarray(VALID_CHROMOSOMES, dtype=int),
            'test': np.asarray(TEST_CHROMOSOMES, dtype=int),
        }

        # passed over the genome
        opt.epochs = opt.finetune_epochs

    else:

        # Load the actual created dataset of sequences
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
            f'dataset_test_all_{opt.window_size}_{opt.context_length}.h5'),
            'r')

        if opt.save_feats:
            # only loop through the genome once
            opt.epochs = 1
        else:
            opt.epochs = opt.passes

        datasets = {
            'train': train_data_file,
            'valid': valid_data_file,
            'test': test_data_file
        }

    opt.chromosomes = {
        'train': np.asarray(TRAIN_CHROMOSOMES, dtype=int),
        'valid': np.asarray(VALID_CHROMOSOMES, dtype=int),
        'test': np.asarray(TEST_CHROMOSOMES, dtype=int),
    }

    # Initialize wandb
    config = get_wandb_config(opt)

    if opt.wandb:
        if opt.pretrain:
            wandb_project_name = 'pretrain'
        elif opt.finetune:
            wandb_project_name = 'finetune'
        elif opt.test_baseline:
            wandb_project_name = 'test_baseline'
        elif opt.test_graph:
            wandb_project_name = 'test_graph'
        wandb.init(
            entity='splicegraph', project=wandb_project_name, config=config, name=config.name
        )

    base_model = None
    graph_model, full_model = None, None
    # Initialize Models
    if opt.load_pretrained:
        if opt.test_graph:
            # currently, the only reason to load pretrained
            # graph model is to test it
            graph_model, full_model = load_pretrained_graph_model(opt, config)
        else:
            base_model = load_pretrained_base_model(opt, config)
    else:
        base_model = SpliceAI(
            opt.n_channels,
            opt.kernel_size,
            opt.dilation_rate
        )

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

    # if saving feats or finetuning, need to load a trained base model
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
        if opt.cnn_sched == "multisteplr":
            step_size_milestones = [6, 7, 8, 9, 10]
            scheduler = torch.torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_size_milestones, 
                gamma=0.5, verbose=False
            )
        elif opt.cnn_sched == "reducelr":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=opt.rlr_factor,
                patience=opt.rlr_patience,
                threshold=opt.rlr_threshold,
                verbose=True
            )
    elif opt.finetune:
        optimizer = get_optimizer(
            graph_model, full_model, opt
        )
        scheduler = get_scheduler(
            opt, optimizer, len(datasets['train'])
        )
        
    elif not (opt.test_baseline or opt.test_graph):
        optimizer = get_optimizer(base_model, opt)
    
    logging.info(f"==> Optimizer: {optimizer}")

    criterion = get_criterion(opt)

    # Take care of the GPU stuff
    if torch.cuda.device_count() > 0 and opt.pretrain:
        logging.info(f'Using {torch.cuda.device_count()} GPUs!')
        base_model = nn.DataParallel(base_model)

    if torch.cuda.is_available() and opt.cuda:
        criterion = criterion.cuda()
        if base_model is not None:
            base_model = base_model.cuda()
        if graph_model is not None:
            graph_model = graph_model.cuda()
        if full_model is not None:
            full_model = full_model.cuda()

    logging.info(f'Model name: {opt.model_name}')

    try:
        # Run the model
        run_model(base_model, graph_model, full_model, datasets, criterion,
                  optimizer, scheduler, opt)
    except KeyboardInterrupt:
        logging.info('-' * 89 + '\nManual Exit')
        exit()


if __name__ == '__main__':
    main(opt)
