###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

import re
from math import ceil
import logging
from collections import Counter

import numpy as np
import wandb
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm
import os
import yaml


### LOADING CONFIG 
with open("config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

CL_max = config['DATA_PIPELINE']['context_length']
SL = config['DATA_PIPELINE']['window_size']

assert CL_max % 2 == 0

# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

# One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
# 2 is for donor and -1 is for padding.
OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])


CHR2IX = lambda chr: 23 if chr == 'X' else 24 if chr == 'Y' else int(chr)
IX2CHR = lambda ix: 'chrX' if ix == 23 else 'chrY' \
    if ix == 24 else f'chr{str(ix)}'


CHR2IX = lambda chr: 23 if chr == 'X' else 24 if chr == 'Y' else int(chr)


def ceil_div(x, y):
    return int(ceil(float(x) / y))


def create_datapoints(seq, strand, tx_start, tx_end, jn_start, jn_end, chrom):
    # This function first converts the sequence into an integer array, where
    # A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    # negative, then reverse complementing is done. The splice junctions 
    # are also converted into an array of integers, where 0, 1, 2, -1 
    # correspond to no splicing, acceptor, donor and missing information
    # respectively. It then calls reformat_data and one_hot_encode
    # and returns X, Y which can be used by Keras models.

    # replace the context window base pairs outside of the main sequence
    # with Ns
    #seq = 'N' * (CL_max // 2) + seq[CL_max // 2:-CL_max // 2] + 'N' * (
    #        CL_max // 2)
    # Context being provided on the RNA and not the DNA

    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')

    tx_start = int(tx_start)
    tx_end = int(tx_end)

    jn_start = list(
        map(lambda x: list(map(int, re.split(',', x)[:-1])), jn_start))
    jn_end = list(map(lambda x: list(map(int, re.split(',', x)[:-1])), jn_end))

    if strand == '+':

        X0 = np.asarray(list(map(int, list(seq))))
        Y0 = [-np.ones(tx_end - tx_start + 1) for t in range(1)]

        for t in range(1):

            if len(jn_start[t]) > 0:
                Y0[t] = np.zeros(tx_end - tx_start + 1)
                for c in jn_start[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][c - tx_start] = 2
                for c in jn_end[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][c - tx_start] = 1
                    # Ignoring junctions outside annotated tx start/end sites

    elif strand == '-':

        X0 = (5 - np.asarray(
            list(map(int, list(seq[::-1]))))) % 5  # Reverse complement
        Y0 = [-np.ones(tx_end - tx_start + 1) for t in range(1)] # WHY IS THIS + 1

        for t in range(1):

            if len(jn_start[t]) > 0:
                Y0[t] = np.zeros(tx_end - tx_start + 1)
                for c in jn_end[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][tx_end - c] = 2
                for c in jn_start[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][tx_end - c] = 1

    # take the big sequence of X's and labels for the gene, 
    # and split it into a dataset, breaking the big sequence
    # into chunks of length SL (plus context length for X's)
    # and storing each such chunk in separate row.
    # start location of each row stored in locs
    Xd, Yd, locs = reformat_data(X0, Y0, tx_start)


    X, Y = one_hot_encode(Xd, Yd)

    chroms = [CHR2IX(chrom[3:])] * len(locs)
    return X, Y, locs, chroms


def reformat_data(X0, Y0, tx_start):
    # This function converts X0, Y0 of the create_datapoints function into
    # blocks such that the data is broken down into data points where the
    # input is a sequence of length SL+CL_max corresponding to SL nucleotides
    # of interest and CL_max context nucleotides, the output is a sequence of
    # length SL corresponding to the splicing information of the nucleotides
    # of interest. The CL_max context nucleotides are such that they are
    # CL_max/2 on either side of the SL nucleotides of interest.

    #num_points = ceil_div(len(Y0[0]), SL)
    num_points = int(len(Y0[0]) // SL)
    locs = np.zeros(num_points)

    # create matrix of shape [# of sequences of length SL, SL + context window]
    Xd = np.zeros((num_points, SL + CL_max))
    Yd = [-np.ones((num_points, SL)) for t in range(1)]

    # add padding of size SL (e.g. 5000) to the end of the Xs and Ys
    #X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)
    #Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1)
    #      for t in range(1)]

    # populate rows with incremented sequence values
    for i in range(num_points):
        Xd[i] = X0[SL * i:CL_max + SL * (i + 1)]
        # for debugging purposes
        x_start = (tx_start - (CL_max // 2 + 1)) + SL * i
        x_end = (tx_start - (CL_max // 2 + 1)) + CL_max + SL * (i + 1)


    # populate labels for all nucleotides in each incremental sequence
    # and store the locations
    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL * i:SL * (i + 1)]
            locs[i] = tx_start + i * SL

    return Xd, Yd, locs


def clip_datapoints(X, Y, CL, N_GPUS):
    # This function is necessary to make sure of the following:
    # (i) Each time model.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store 
    # them as an array).

    rem = X.shape[0] % N_GPUS
    clip = (CL_max - CL) // 2

    if rem != 0 and clip != 0:
        return X[:-rem, clip:-clip], [Y[t][:-rem] for t in range(1)]
    elif rem == 0 and clip != 0:
        return X[:, clip:-clip], [Y[t] for t in range(1)]
    elif rem != 0 and clip == 0:
        return X[:-rem], [Y[t][:-rem] for t in range(1)]
    else:
        return X, [Y[t] for t in range(1)]


def one_hot_encode(Xd, Yd):
    return IN_MAP[Xd.astype('int8')], \
           [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]


def validate(model, h5f, idxs, context_length, batch_size,
             class_weights=(1, 1, 1), n_gpus=1, test=False):

    from splicing.models.splice_ai import categorical_crossentropy_2d
    from splicing.data_models.splice_dataset import SpliceDataset
    import torch
    from torch.utils.data import DataLoader

    y_true_1, y_true_2, y_pred_1, y_pred_2 = [], [], [], []

    total_loss, total_len = 0, 0
    for idx in tqdm(idxs):
        X = h5f['X' + str(idx)][:]
        Y = np.asarray((h5f['Y' + str(idx)][:]), np.float32)

        splice_dataset = SpliceDataset(X, Y, context_length, n_gpus)
        dataloader = DataLoader(splice_dataset, batch_size=batch_size)

        yp = np.zeros(shape=splice_dataset.Y.shape)
        m = 0
        partial_loss = 0
        with torch.no_grad():
            for batch, (X, y) in tqdm(enumerate(dataloader), leave=False):
                pred, _, _ = model(X)
                yp[m: m + len(X)] = pred.cpu().numpy()
                m += len(X)

                partial_loss += categorical_crossentropy_2d(
                    y, pred, weights=class_weights).item()
        total_loss += partial_loss
        total_len += len(dataloader.dataset)

        is_expr = splice_dataset.get_expr()

        y_true_1.extend(splice_dataset.get_true(1, is_expr))
        y_true_2.extend(splice_dataset.get_true(2, is_expr))
        y_pred_1.extend(yp[is_expr, 1, :].flatten())
        y_pred_2.extend(yp[is_expr, 2, :].flatten())

    logging.info(f'Total loss: {total_loss / total_len:>12f}')
    logging.info("\nAcceptor:")
    print_topl_statistics(
        np.asarray(y_true_1), np.asarray(y_pred_1), loss=total_loss,
        prediction_type='Acceptor', test=test)

    logging.info("\nDonor:")
    print_topl_statistics(
        np.asarray(y_true_2), np.asarray(y_pred_2), loss=total_loss,
        prediction_type='Donor', test=test)


def print_topl_statistics(y_true, y_pred, loss, prediction_type, log_wandb):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:
        idx_pred = argsorted_y_pred[-int(top_length * len(idx_true)):]

        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred))
                           / (float(min(len(idx_pred), len(idx_true))) + 1e-6)]
        threshold += [sorted_y_pred[-int(top_length * len(idx_true))]]

    auprc = average_precision_score(y_true, y_pred)

    no_positive_predictions = len(np.nonzero(y_pred > 0.5)[0])
    logging.info('Top-K Accuracy')
    logging.info('|0.5\t|1\t|2\t|4\t|')
    logging.info('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|'.format(
        topkl_accuracy[0], topkl_accuracy[1],
        topkl_accuracy[2], topkl_accuracy[3]))
    logging.info('Thresholds for K')
    logging.info('|0.5\t|1\t|2\t|4\t|')
    logging.info('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|'.format(
        threshold[0], threshold[1], threshold[2], threshold[3]))
    logging.info(f'AUPRC: {auprc:.6f}')
    logging.info(f'# True Splice Sites: {len(idx_true)} / {len(y_true)}')
    logging.info('# Predicted Splice Sites: '
                 f'{no_positive_predictions} / {len(y_pred)}')
    if log_wandb:
        wandb.log({
            f'full_valid/Test Loss: {prediction_type}': loss,
            f'full_valid/AUPRC: {prediction_type}': auprc,
            f'full_valid/Top-K Accuracy: {prediction_type}': topkl_accuracy[1],
            f'full_valid/Thresholds for K: {prediction_type}': threshold[1],
            f'full_valid/Proportion of True Splice Sites Predicted'
            f': {prediction_type}': no_positive_predictions / len(idx_true),
        })


def get_architecture(size, N_GPUS=1):
    if int(size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18 * N_GPUS
    elif int(size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18 * N_GPUS
    elif int(size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                         10, 10, 10, 10])
        BATCH_SIZE = 12 * N_GPUS
    elif int(size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                         10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6 * N_GPUS

    return W, AR, BATCH_SIZE


def get_data(h5f, available_chromosomes, context_length, batch_size,
             full=False, chromosome=None):
    from splicing.data_models.splice_dataset import SpliceDataset
    from torch.utils.data import DataLoader, ConcatDataset

    def get_dataset(dchromosome, dix):

        X = h5f[dchromosome + '_X' + str(dix)][:]
        y = np.asarray(h5f[dchromosome + '_Y' + str(dix)][:], dtype=np.float32)
        locs = np.asarray(h5f[dchromosome + '_Locations' + str(dix)][:],
                          dtype=np.float32)

        return SpliceDataset(X, y, locs, context_length)

    if not full:
        if chromosome is None:  # return a random chunk
            chromosome = IX2CHR(np.random.choice(available_chromosomes))
            n_chromosome_chunks = sum(
                [chromosome + '_X' == key[:len(chromosome + '_X')]
                 for key in h5f.keys()])
            ix = np.random.choice(n_chromosome_chunks)
            return DataLoader(
                get_dataset(chromosome, ix), batch_size=batch_size)
        else:  # return the full chromosome
            datasets = []
            n_chromosome_chunks = sum(
                [chromosome + '_X' == key[:len(chromosome + '_X')]
                 for key in h5f.keys()])
            for ix in range(n_chromosome_chunks):
                datasets.append(get_dataset(chromosome, ix))
            return DataLoader(ConcatDataset(datasets), batch_size=batch_size)

    else:  # test/valid
        datasets = []
        for chromosome in [IX2CHR(c) for c in available_chromosomes]:
            n_chromosome_chunks = sum(
                [chromosome + '_X' == key[:len(chromosome + '_X')]
                 for key in h5f.keys()])
            for ix in range(n_chromosome_chunks):
                datasets.append(get_dataset(chromosome, ix))
        return DataLoader(ConcatDataset(datasets), batch_size=batch_size)
