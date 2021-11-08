###############################################################################
# This file contains code to test the SpliceAI model.
###############################################################################

import time
import argparse

import h5py
import torch
import logging
import coloredlogs

from splicing.utils.utils import validate, get_architecture
from splicing.utils.constants import data_dir
from splicing.models.splice_ai import SpliceAI

coloredlogs.install(level=logging.INFO)


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
        '-m', '--model_fname', type=str, dest='model_fname',
        help='The filename of the saved model.')
    parser.add_argument(
        '-nc', '--n_channels', type=int, default=32, dest='n_channels',
        help='Number of convolution channels that the model uses.')

    return parser.parse_args()


def test_model(context_length, model_fname, n_channels):

    kernel_size, dilation_rate, _ = get_architecture(context_length)
    model = SpliceAI(n_channels, kernel_size, dilation_rate)
    model.load_state_dict(torch.load(f'Models/{model_fname}'))

    h5f = h5py.File(data_dir + 'dataset_test_0.h5', 'r')

    num_idx = len(h5f.keys()) // 2

    start_time = time.time()

    validate(model, h5f, list(range(num_idx)), context_length, batch_size=64,
             test=True)

    h5f.close()

    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")


if __name__ == '__main__':

    args = parse_args(
        argparse.ArgumentParser(description='Train a SpliceAI model.'))

    assert args.context_length in [80, 400, 2000, 10000]

    test_model(args.context_length, args.model_fname)
