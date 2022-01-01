###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

import re
import logging
import os

from math import ceil
from collections import Counter
import numpy as np
import yaml
from splicing.utils.general_utils import CHR2IX, IX2CHR


### LOADING CONFIG 
with open("config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

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


def ceil_div(x, y):
    return int(ceil(float(x) / y))


def create_datapoints(seq, strand, tx_start, tx_end, jn_start, jn_end, chrom, \
    window_size, context_length):
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
    Xd, Yd, locs = reformat_data(X0, Y0, tx_start, window_size, \
         context_length)

    X, Y = one_hot_encode(Xd, Yd)

    chroms = [CHR2IX(chrom[3:])] * len(locs)
    return X, Y, locs, chroms


def reformat_data(X0, Y0, tx_start, window_size, context_length):
    # This function converts X0, Y0 of the create_datapoints function into
    # blocks such that the data is broken down into data points where the
    # input is a sequence of length SL+CL_max corresponding to SL nucleotides
    # of interest and CL_max context nucleotides, the output is a sequence of
    # length SL corresponding to the splicing information of the nucleotides
    # of interest. The CL_max context nucleotides are such that they are
    # CL_max/2 on either side of the SL nucleotides of interest.

    #num_points = ceil_div(len(Y0[0]), SL)
    num_points = int(len(Y0[0]) // window_size)
    locs = np.zeros(num_points)

    # create matrix of shape [# of sequences of length SL, SL + context window]
    Xd = np.zeros((num_points, window_size + context_length))
    Yd = [-np.ones((num_points, window_size)) for t in range(1)]

    # add padding of size SL (e.g. 5000) to the end of the Xs and Ys
    #X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)
    #Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1)
    #      for t in range(1)]

    # populate rows with incremented sequence values
    for i in range(num_points):
        Xd[i] = X0[window_size * i:context_length + window_size * (i + 1)]
        # for debugging purposes
        #x_start = (tx_start - (context_length // 2 + 1)) + window_size * i
        #x_end = (tx_start - (context_length // 2 + 1)) + \
        # context_length + window_size * (i + 1)


    # populate labels for all nucleotides in each incremental sequence
    # and store the locations
    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][window_size * i:window_size * (i + 1)]
            locs[i] = tx_start + i * window_size

    return Xd, Yd, locs


def clip_datapoints(X, Y, cl, cl_max, N_GPUS):
    # This function is necessary to make sure of the following:
    # (i) Each time model.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store 
    # them as an array).

    rem = X.shape[0] % N_GPUS
    clip = (cl_max - cl) // 2

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


def get_data(h5f, chromosome, context_length, batch_size, device='cuda'):
    from splicing.data_models.splice_dataset import SpliceDataset
    from torch.utils.data import DataLoader, ConcatDataset, Subset

    def get_dataset(dchromosome, dix):

        X = h5f[dchromosome + '_X' + str(dix)][:]
        y = np.asarray(h5f[dchromosome + '_Y' + str(dix)][:], dtype=np.float32)
        locs = np.asarray(h5f[dchromosome + '_Locations' + str(dix)][:],
                          dtype=np.float32)

        return SpliceDataset(X, y, locs, context_length, device=device)

    chromosome = IX2CHR(chromosome)
    datasets = []
    n_chromosome_chunks = sum(
        [chromosome + '_X' == key[:len(chromosome + '_X')]
            for key in h5f.keys()])
    for ix in range(n_chromosome_chunks):
        datasets.append(get_dataset(chromosome, ix))
    
    #for debugging:
    #subset_dataset = Subset(ConcatDataset(datasets), range(100))
    #return DataLoader(subset_dataset, batch_size=batch_size)

    return DataLoader(ConcatDataset(datasets), batch_size=batch_size)
    
