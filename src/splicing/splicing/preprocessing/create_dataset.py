###############################################################################
"""
This parser takes as input the .h5 file produced by create_datafile.py and
outputs a .h5 file with datapoints of the form (X, Y).
"""
###############################################################################

import argparse

import h5py
import numpy as np
import time
import os
import yaml

from splicing.utils.utils import create_datapoints

start_time = time.time()

###############################################################################
# Parsing Args
###############################################################################

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

###############################################################################
# Loading Config
###############################################################################

with open("config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

DATA_DIR = config['DATA_DIRECTORY']

INTERVAL = config['DATA_PIPELINE']['window_size']

# inputs
DATAFILE_PATH = os.path.join(
    DATA_DIR, 
    config['DATA_PIPELINE']['output_dir'],
    f'datafile_{group}_{paralog}_{INTERVAL}.h5'
)

# outputs
DATASET_PATH = os.path.join(
    DATA_DIR, 
    config['DATA_PIPELINE']['output_dir'],
    f'dataset_{group}_{paralog}_{INTERVAL}.h5'
)

###############################################################################
# Creating Dataset
###############################################################################
print(f"Reading from datafile {DATAFILE_PATH}")
h5f = h5py.File(DATAFILE_PATH, 'r')

SEQ = h5f['SEQ'].asstr()[:]
STRAND = h5f['STRAND'].asstr()[:]
JN_START = h5f['JN_START'].asstr()[:]
JN_END = h5f['JN_END'].asstr()[:]

TX_START = h5f['TX_START'].asstr()[:]
TX_END = h5f['TX_END'].asstr()[:]

h5f.close()


print(f"Outputting to dataset {DATASET_PATH}")
h5f2 = h5py.File(DATASET_PATH, 'w')

# CHUNK_SIZE: max number of genes to be stored in a single h5 dataset.
CHUNK_SIZE = 100

for i in range(SEQ.shape[0] // CHUNK_SIZE):
    # Each dataset has CHUNK_SIZE genes

    if (i + 1) == SEQ.shape[0] // CHUNK_SIZE:
        NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0] % CHUNK_SIZE
    else:
        NEW_CHUNK_SIZE = CHUNK_SIZE

    X_batch = []
    Y_batch = [[] for t in range(1)]
    locations_batch = []

    for j in range(NEW_CHUNK_SIZE): #for each gene

        idx = i * CHUNK_SIZE + j #idx of the gene data in datafile

        X, Y, locations = create_datapoints(
            SEQ[idx], STRAND[idx],
            TX_START[idx], TX_END[idx],
            JN_START[idx], JN_END[idx]
        )

        X_batch.extend(X)
        locations_batch.extend(locations)
        for t in range(1):
            Y_batch[t].extend(Y[t])

    X_batch = np.asarray(X_batch).astype('int8')
    for t in range(1):
        Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')

    h5f2.create_dataset('X' + str(i), data=X_batch)
    h5f2.create_dataset('Y' + str(i), data=Y_batch)
    h5f2.create_dataset('Locations' + str(i), data=locations_batch)

h5f2.close()

print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
