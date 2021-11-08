###############################################################################
# This file contains code to test the SpliceAI model.
###############################################################################

import time
import argparse

import numpy as np
import h5py
import torch
import logging
import coloredlogs

from splicing.utils.utils import validate
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

    return parser.parse_args()


def test_model(context_length, model_fname):

    # model = SpliceAI(
    #     32,
    #     np.asarray([11, 11, 11, 11, 11, 11, 11, 11]),
    #     np.asarray([1, 1, 1, 1, 4, 4, 4, 4]))
    #
    # model.load_state_dict(torch.load(f'Models/{model_fname}'))

    model = torch.load(f'Models/{model_fname}')

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
