###############################################################################
"""
This parser takes as input the .h5 file produced by create_datafile.py and
outputs a .h5 file with datapoints of the form (X, Y), which can be understood
by Keras models.
"""
###############################################################################

import argparse

import h5py
import numpy as np
import time

from splicing.utils.utils import create_datapoints
from splicing.utils.constants import data_dir


start_time = time.time()

parser = argparse.ArgumentParser(
    description='Create the model-compatible datasets.')
parser.add_argument(
    '-g', '--group', dest='group', type=str,
    help='The chromosome group to process. One of ["train", "test", "all"].')
parser.add_argument(
    '-p', '--paralog', dest='paralog', type=str,
    help='Whether to include the genes with paralogs or not.')

args = parser.parse_args()

group = args.group
paralog = args.paralog

assert group in ['train', 'test', 'all']
assert paralog in ['0', '1', 'all']

h5f = h5py.File(data_dir + 'datafile_' + group + '_' + paralog + '.h5', 'r')

SEQ = h5f['SEQ'].asstr()[:]
STRAND = h5f['STRAND'].asstr()[:]
CHROM = h5f['CHROM'].asstr()[:]
TX_START = h5f['TX_START'].asstr()[:]
TX_END = h5f['TX_END'].asstr()[:]
JN_START = h5f['JN_START'].asstr()[:]
JN_END = h5f['JN_END'].asstr()[:]
h5f.close()

h5f2 = h5py.File(data_dir + 'dataset_' + group + '_' + paralog + '.h5', 'w')

CHUNK_SIZE = 100

for i in range(SEQ.shape[0] // CHUNK_SIZE):
    # Each dataset has CHUNK_SIZE genes

    if (i + 1) == SEQ.shape[0] // CHUNK_SIZE:
        NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0] % CHUNK_SIZE
    else:
        NEW_CHUNK_SIZE = CHUNK_SIZE

    X_batch = []
    Y_batch = [[] for t in range(1)]  # TODO: remove this [[]] ...
    locations_batch = []
    chromosomes_batch = []

    for j in range(NEW_CHUNK_SIZE):

        idx = i * CHUNK_SIZE + j

        X, Y, locations, chromosome = create_datapoints(
            SEQ[idx], STRAND[idx],
            TX_START[idx], TX_END[idx],
            JN_START[idx], JN_END[idx], CHROM[idx])

        X_batch.extend(X)
        locations_batch.extend(locations)
        chromosomes_batch.extend(chromosome)
        for t in range(1):
            Y_batch[t].extend(Y[t])

    X_batch = np.asarray(X_batch).astype('int8')
    for t in range(1):
        Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')

    # print(chromosomes_batch)

    h5f2.create_dataset('X' + str(i), data=X_batch)
    h5f2.create_dataset('Y' + str(i), data=Y_batch)
    h5f2.create_dataset('Locations' + str(i), data=locations_batch)
    h5f2.create_dataset('Chromosomes' + str(i), data=chromosomes_batch,
                        dtype=int)

h5f2.attrs['n_datasets'] = SEQ.shape[0] // CHUNK_SIZE

h5f2.close()

print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
