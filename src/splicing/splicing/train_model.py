###############################################################################
# This file contains the code to train the SpliceAI model.
###############################################################################

import coloredlogs
import time
import logging
import argparse

import numpy as np
import h5py
from tqdm.auto import tqdm, trange
import wandb
import os
import yaml

import torch
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import torch.nn.functional as F

from splicing.models.splice_ai import SpliceAI, categorical_crossentropy_2d
from splicing.utils.utils import get_architecture, validate, get_data
from splicing.data_models.splice_dataset import SpliceDataset

coloredlogs.install(level=logging.INFO)


# ----------------------------------------------------------------
# Loading Config
# ---------------------------------------------------------------- 
with open("config.yaml", "r") as stream:
    try:
        repo_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

data_dir = os.path.join(
    repo_config['DATA_DIRECTORY'], 
    repo_config['DATA_PIPELINE']['output_dir']
)

CL_max = repo_config['DATA_PIPELINE']['context_length']

SL = repo_config['DATA_PIPELINE']['window_size']


# ----------------------------------------------------------------
# Command Line arguments
# ----------------------------------------------------------------
def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the input arguments to the file
    :param parser: parser of args
    :return: parsed arguments
    """

    parser.add_argument(
        '-cl', '--context_length', dest='context_length',
        type=int, default=400, help='The context length to use.')
    parser.add_argument(
        '-i', '--model_index', default=0, type=int, dest='model_index',
        help='The index of the model if training an ensemble.')
    parser.add_argument(
        '-nc', '--n_channels', type=int, default=32, dest='n_channels',
        help='Number of convolution channels to use.')
    parser.add_argument(
        '-w', '--class_weights', type=int, nargs=3, default=(1, 1, 1),
        dest='class_weights', help='Class weights to use.')

    return parser.parse_args()


def epoch_end(model, h5f, idxs, context_length, batch_size, class_weights,
              optimizer, start_time, model_index, epoch_n):
    print('----------------------------------------------------------')
    logging.info('\nValidation set metrics:')

    validate(model, h5f, idxs, context_length, batch_size, class_weights)

    logging.info(
        'Learning rate: %.5f' % (optimizer.param_groups[0]['lr']))
    logging.info(
        '--- %s seconds ---' % (time.time() - start_time))

    print('----------------------------------------------------------')

    torch.save(
        model.state_dict(),
        f'Models/SpliceAI{context_length}_e{epoch_n}_g{model_index}.h5')


def architecture_setup(cl):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    kernel_size, dilation_rate, batch_size = get_architecture(cl)

    context_length = 2 * np.sum(dilation_rate * (kernel_size - 1))
    assert context_length <= CL_max and context_length == cl

    return device, kernel_size, dilation_rate, batch_size, context_length


def train_model(model_index, cl, n_channels, class_weights):
    device, kernel_size, dilation_rate, batch_size, context_length = \
        architecture_setup(cl)

    logging.debug(f'Context nucleotides: {context_length} ')
    logging.debug(f'Sequence length (output): {SL} ')

    h5f = h5py.File(os.path.join(data_dir, 'dataset_train_all.h5'), 'r')

    num_idx = h5f.attrs['n_datasets']
    idx_all = np.random.permutation(num_idx)

    train_ratio = 0.9
    idx_train = idx_all[:int(train_ratio * num_idx)]
    idx_valid = idx_all[int(train_ratio * num_idx):]

    config = wandb.config
    config.n_channels = n_channels
    config.context_length = context_length
    config.kernel_size = kernel_size
    config.dilation_rate = dilation_rate
    config.batch_size = batch_size
    config.epochs = 10 * len(idx_train)
    config.context_length = context_length
    config.lr = 1e-3
    config.train_ratio = train_ratio
    config.class_weights = class_weights
    config.log_interval = 32

    wandb.init(project='dl-project', config=config)

    model = SpliceAI(
        config.n_channels, config.kernel_size, config.dilation_rate).to(device)
    summary(model, input_size=(4, context_length + SL), batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    start_time = time.time()
    for epoch_num in trange(config.epochs):
        dataloader = get_data(
            h5f, idx_train, config.context_length, config.batch_size)

        total_loss = 0
        size = len(dataloader.dataset)
        for batch, (X, y, loc, chr) in tqdm(
                enumerate(dataloader), total=size // config.batch_size,
                leave=False):

            y_hat, _, _ = model(X)
            loss = categorical_crossentropy_2d(
                y, y_hat, weights=config.class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch % config.log_interval == 0:
                y_hat = y_hat.detach().cpu().numpy()
                sums_true = y.sum(axis=(0, 2))
                sums_pred = y_hat.sum(axis=(0, 2))

                total = sums_true.sum()
                wandb.log({
                    'loss': loss.item() / config.batch_size,
                    # 'true inactive': sums_true[0] / total,
                    'true acceptors': sums_true[1] / total,
                    'true donors': sums_true[2] / total,
                    # 'predicted inactive': sums_pred[0] / sums_true[0],
                    'predicted acceptors': sums_pred[1] / sums_true[1],
                    'predicted donors': sums_pred[2] / sums_true[2],
                    # 'proportion of epoch done': batch / (size // batch_size),
                })

        logging.debug(f'Epoch loss: {total_loss / size:>12f}')

        if (epoch_num + 1) % len(idx_train) == 0:
            epoch_end(
                model, h5f, idx_valid,
                config.context_length, config.batch_size,
                config.class_weights, optimizer, start_time,
                model_index, (epoch_num + 1) // len(idx_train))
            start_time = time.time()

            if (epoch_num + 1) >= 6 * len(idx_train):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.5

    h5f.close()


if __name__ == '__main__':

    args = parse_args(
        argparse.ArgumentParser(description='Train a SpliceAI model.'))

    assert args.context_length in [80, 400, 2000, 10000]

    train_model(args.model_index, args.context_length,
                args.n_channels, args.class_weights)
