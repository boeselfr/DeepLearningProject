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

from config_args import config_args, get_args
from models.ChromeModels import ChromeGCN
from runner import run_model
from utils import util_methods
from utils.evals import Logger

from splicing.models.splice_ai import SpliceAI
from splicing.utils.utils import get_architecture
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

        opt.epochs = len(opt.idx_train) * opt.passes

    logging.info('==> Creating window_model')
    kernel_size, dilation_rate, batch_size = get_architecture(
        opt.context_length)

    opt.kernel_size = kernel_size
    opt.dilation_rate = dilation_rate
    opt.batch_size = batch_size

    config = util_methods.get_wandb_config(opt)

    base_model = SpliceAI(
        config.n_channels, config.kernel_size, config.dilation_rate)

    wandb.init(project='dl-project', config=config)

    # TODO: I think using non-strand specific is not necessary
    opt.total_num_parameters = int(util_methods.count_parameters(base_model))
    logging.info('>>>>>>>>>> BASE MODEL <<<<<<<<<<<')
    logging.info('Total number of parameters in the base model: '
                 f'{opt.total_num_parameters}.')
    summary(base_model,
            input_size=(4, opt.context_length + SL),
            device='cuda')

    optimizer = util_methods.get_optimizer(base_model, opt)

    chrome_model = None
    if not opt.pretrain:
        # Creating GNNModel
        chrome_model = ChromeGCN(
            128, 128, opt.tgt_vocab_size, opt.gcn_dropout,
            opt.gate, opt.gcn_layers)

        logging.info(chrome_model)

        if opt.load_gcn:
            logging.info('Loading Saved GCN')
            checkpoint = torch.load(
                opt.model_name.replace('.load_gcn', '') + '/model.chkpt')
            chrome_model.load_state_dict(checkpoint['model'])
        else:
            # Initialize GCN output layer with window_model output layer
            logging.info('Loading Saved window_model')
            # checkpoint = torch.load(opt.saved_model)
            checkpoint = torch.load(
                opt.model_name.split('.finetune')[0] + '/model.chkpt')
            base_model = nn.DataParallel(base_model)
            base_model = base_model.cuda()
            base_model.load_state_dict(checkpoint['model'])
            chrome_model.out.weight.data = \
                base_model.module.model.classifier.weight.data
            chrome_model.out.bias.data = \
                base_model.module.model.classifier.bias.data
            chrome_model.batch_norm.weight.data = \
                base_model.module.model.batch_norm.weight.data
            chrome_model.batch_norm.bias.data = \
                base_model.module.model.batch_norm.bias.data

        optimizer = util_methods.get_optimizer(chrome_model, opt)

    scheduler = torch.torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.epochs, gamma=0.5)
    logging.info(optimizer)

    criterion = util_methods.get_criterion(opt)

    if torch.cuda.device_count() > 0 and opt.pretrain:
        logging.info(f'Using {torch.cuda.device_count()} GPUs!')
        base_model = nn.DataParallel(base_model)

    if torch.cuda.is_available() and opt.cuda:
        criterion = criterion.cuda()
        if opt.pretrain:
            base_model = base_model.cuda()
        if chrome_model is not None:
            chrome_model = chrome_model.cuda()
        if opt.gpu_id != -1:
            torch.cuda.set_device(opt.gpu_id)

    # print(opt.model_name)
    # logger = Logger(opt)

    try:
        run_model(base_model, chrome_model, data_file, criterion,
                  optimizer, scheduler, opt, logger=None)
    except KeyboardInterrupt:
        logging.info('-' * 89 + '\nManual Exit')
        exit()


if __name__ == '__main__':
    main(opt)
