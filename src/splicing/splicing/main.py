from os import path
import argparse
import warnings
import logging

import numpy as np
import h5py
import torch
import torch.nn as nn
from torchsummary import summary
import wandb
import coloredlogs

from splicing.utils.config_args import config_args, get_args
from splicing.models.graph_models import ChromeGCN
from runner import run_model
from splicing.utils import graph_utils

from splicing.models.splice_ai import SpliceAI
from splicing.utils.constants import SL

coloredlogs.install(level=logging.INFO)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)


def main(opt):

    # Loading Dataset
    logging.info(opt.model_name)

    logging.info('==> Loading Data')
    logging.info(opt.data)

    logging.info('==> Processing Data')
    if not opt.pretrain and not opt.save_feats:
        train_data = torch.load(
            opt.model_name.split('.finetune')[0]
            + '/chrom_feature_dict_train.pt')
        valid_data = torch.load(
            opt.model_name.split('.finetune')[0]
            + '/chrom_feature_dict_valid.pt')
        test_data = torch.load(
            opt.model_name.split('.finetune')[0]
            + '/chrom_feature_dict_test.pt')

        data_file = None
    else:

        data_file = h5py.File(path.join(
            opt.splice_data_root, 'dataset_train_all.h5'), 'r')

        num_idx = data_file.attrs['n_datasets']
        idx_all = np.random.permutation(num_idx)

        opt.idx_train = idx_all[:int(opt.train_ratio * num_idx)]
        opt.idx_valid = idx_all[int(opt.train_ratio * num_idx):]

        if not opt.save_feats:
            opt.epochs = len(opt.idx_train) * opt.passes

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
    summary(base_model,
            input_size=(4, opt.context_length + SL),
            device='cuda')

    optimizer = graph_utils.get_optimizer(base_model, opt)

    graph_model = None
    if not opt.pretrain:
        # Creating GNNModel

        if not opt.save_feats:
            graph_model = ChromeGCN(
                32, opt.hidden_size, opt.gcn_dropout, opt.gate, opt.gcn_layers)

            logging.info(graph_model)

        if opt.load_gcn:
            logging.info('Loading Saved GCN')
            checkpoint = torch.load(
                opt.model_name.replace('.load_gcn', '') + '/model.chkpt')
            graph_model.load_state_dict(checkpoint['model'])
        else:
            # Initialize GCN output layer with window_model output layer
            logging.info('Loading Saved base_model')

            model_fname = f'SpliceAI{opt.context_length}_g{opt.model_index}.h5'

            checkpoint = torch.load(path.join(
                opt.model_name.split('.finetune')[0], model_fname))
            base_model = nn.DataParallel(base_model)
            base_model = base_model.cuda()
            base_model.load_state_dict(checkpoint['model'])

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
        optimizer, step_size=config.epochs, gamma=0.5)
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
        run_model(base_model, graph_model, data_file, criterion,
                  optimizer, scheduler, opt, logger=None)
    except KeyboardInterrupt:
        logging.info('-' * 89 + '\nManual Exit')
        exit()


if __name__ == '__main__':
    main(opt)
