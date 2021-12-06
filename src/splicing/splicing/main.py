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
from collections import OrderedDict

from splicing.utils.config_args import config_args, get_args
# from splicing.models.graph_models import SpliceGraph
from splicing.models.geometric_models import SpliceGraph, FullModel
from splicing.runner import run_model
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

train_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                     project_config['DATA_PIPELINE']['train_chroms']]
valid_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                     project_config['DATA_PIPELINE']['valid_chroms']]
test_chromosomes = [CHR2IX(chrom[3:]) for chrom in
                    project_config['DATA_PIPELINE']['test_chroms']]


def main(opt):

    # Workflow
    logging.info(f"==> Workflow: {opt.workflow}, Model name: {opt.model_name}")

    # Loading Dataset
    logging.info('==> Loading Data')
    if opt.finetune:
        # features_dir = opt.model_name.split('.finetune')[0]

        datasets = {
            'train': train_chromosomes,
            'valid': valid_chromosomes,
            'test': test_chromosomes,
        }

        opt.full_validation_interval = len(datasets['train'])

        opt.epochs = opt.finetune_epochs * len(datasets['train'])

    else:

        train_data_file = h5py.File(path.join(
            opt.splice_data_root,
            f'dataset_train_all_{opt.window_size}_{opt.context_length}.h5'), 'r')

        valid_data_file = h5py.File(path.join(
            opt.splice_data_root,
            f'dataset_valid_all_{opt.window_size}_{opt.context_length}.h5'), 'r')

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
        'train': train_chromosomes,
        'valid': valid_chromosomes,
        'test': test_chromosomes,
    }

    config = graph_utils.get_wandb_config(opt)

    if opt.wandb:
        wandb.init(project='dl-project', config=config)

    base_model = SpliceAI(
        config.n_channels, 
        config.kernel_size, 
        config.dilation_rate
    )

    if opt.load_pretrained: 

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

    # if fine_tuning, create the SpliceGraph and Full Model
    if opt.finetune:
        # Creating GNNModel
        # graph_model = SpliceGraph(
        #     32, opt.hidden_size, opt.gcn_dropout, opt.gate, opt.gcn_layers)
        # need to change n channels here:
        if opt.node_representation == 'min-max':
            graph_model = SpliceGraph(
                config.n_channels*2, config.hidden_size, config.gcn_dropout)
        else:
            graph_model = SpliceGraph(
                config.n_channels, config.hidden_size, config.gcn_dropout)
        full_model = FullModel(config.n_channels, config.hidden_size)

        if opt.cuda:
            graph_model = graph_model.cuda()
            full_model = full_model.cuda()

        logging.info(graph_model)
        logging.info(full_model)
        # summary(full_model,
        #         input_size=(2 * opt.n_channels,
        #                     opt.context_length + opt.window_size))

    # if finetune and load_gcn
    if opt.finetune and opt.load_gcn:
        gcn_model_path = opt.model_name.replace('.load_gcn', '') + '/model.chkpt'
        logging.info(f'==> Loading Saved GCN {gcn_model_path}')

        checkpoint = torch.load(gcn_model_path)
        graph_model.load_state_dict(checkpoint['model'])

    # if saving feats or finetuning, need to load a base model
    if opt.save_feats or opt.finetune:
        
        if opt.save_feats:
            assert opt.load_pretrained == True, " pretrained model"

        logging.info(f"==> Turning off base model params for {opt.workflow}")

        for param in base_model.parameters():
            param.requires_grad = False

        # Initialize GCN output layer with window_model output layer
        if opt.finetune and opt.load_pretrained:
            combined_params = list(zip(
                list(base_model.parameters())[-2:],
                list(full_model.parameters())[-2:]
            ))
            combined_params[0][1].data[:, :32, :] = \
                combined_params[0][0].data[:, :, :]
            combined_params[1][1].data[:] = \
                combined_params[1][0].data[:]

    if opt.finetune:
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
    if opt.pretrain:
        spliceai_step_size = (
            train_data_file.attrs['n_datasets'] * opt.lr_step_size
        )
        logging.info("==> Pretraining step size: every "
                     f"{spliceai_step_size} epochs out of "
                     f"{opt.epochs}")
        scheduler = torch.torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=spliceai_step_size, gamma=0.5
        )
    else: 
        scheduler = torch.torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_step_size, gamma=0.5)
    logging.info(optimizer)

    criterion = graph_utils.get_criterion(opt)

    if torch.cuda.device_count() > 0 and opt.pretrain:
        logging.info(f'Using {torch.cuda.device_count()} GPUs!')
        base_model = nn.DataParallel(base_model)

    if torch.cuda.is_available() and opt.cuda:
        criterion = criterion.cuda()
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
