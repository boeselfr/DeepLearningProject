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
from tqdm import trange

from splicing.utils.spliceai_utils import create_datapoints

start_time = time.time()

###############################################################################
# Parsing Args
###############################################################################

parser = argparse.ArgumentParser(
    description='Create the model-compatible datasets.')

parser.add_argument(
    '-cl', '--context_length', dest='context_length',
    choices = [80, 400],
    type=int, default=400, help='The context length to use.')
parser.add_argument(
    '-ws', '--window_size', dest='window_size', 
    choices = [1000, 5000], type=int, default=5000, 
    help='Size of the pretrain batches and graph windows.')
parser.add_argument(
    '-g', '--group', dest='group', type=str,
    help='The chromosome group to process. One of ["train", "test", "all"].')
parser.add_argument(
    '-p', '--paralog', dest='paralog', type=str,
    help='Whether to include the genes with paralogs or not.')

args = parser.parse_args()

cl = args.context_length
ws = args.window_size
group = args.group
paralog = args.paralog

assert group in ['train', 'test', 'valid', 'all']
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

# CHUNK_SIZE: max number of genes to be stored in a single h5 dataset.
CHUNK_SIZE = config['DATA_PIPELINE']['dataset_chunk_size']

# inputs
DATAFILE_PATH = os.path.join(
    DATA_DIR,
    config['DATA_PIPELINE']['output_dir'],
    f'datafile_{group}_{paralog}_{ws}_{cl}.h5'
)

# outputs
DATASET_PATH = os.path.join(
    DATA_DIR,
    config['DATA_PIPELINE']['output_dir'],
    f'dataset_{group}_{paralog}_{ws}_{cl}.h5'
)

###############################################################################
# Creating Dataset
###############################################################################
print(f"Reading from datafile {DATAFILE_PATH}")
h5f = h5py.File(DATAFILE_PATH, 'r')

SEQ = h5f['SEQ'].asstr()[:]
STRAND = h5f['STRAND'].asstr()[:]
CHROM = h5f['CHROM'].asstr()[:]
TX_START = h5f['TX_START'].asstr()[:]
TX_END = h5f['TX_END'].asstr()[:]
JN_START = h5f['JN_START'].asstr()[:]
JN_END = h5f['JN_END'].asstr()[:]

h5f.close()

print(f"Outputting to dataset {DATASET_PATH}")
h5f2 = h5py.File(DATASET_PATH, 'w')

def count_nonzero_labels(Y):
    """ Determine the number of splice sites to decide which windows to keep"""
    counts = np.unique(Y, axis=0, return_counts=True)
    for i, val in enumerate(counts[0]):
        zero_count = 0
        if np.array_equal(val, np.array([1,0,0])):
            zero_count = counts[1][i]
    return len(Y) - zero_count

chroms = np.unique(CHROM)
overall_chunk_counter = 0

for chrom in chroms:
    print(f"Processing chromosome: {chrom}")

    chrom_dict = {}

    for idx in range(SEQ.shape[0]):    
        # loop though all genes        
        if CHROM[idx] == chrom: 
            X, Y, locations, _ = create_datapoints(
                SEQ[idx], STRAND[idx],
                TX_START[idx], TX_END[idx],
                JN_START[idx], JN_END[idx], 
                CHROM[idx], ws, cl
            )
            
            for i, loc in enumerate(locations):
                loc_dict = chrom_dict.get(loc, {})
                Xs = loc_dict.get("Xs", [])
                Xs.append(X[i])
                loc_dict["Xs"] = Xs
                Ys = loc_dict.get("Ys", [])
                Ys.append(Y[0][i])
                loc_dict["Ys"] = Ys
                chrom_dict[loc] = loc_dict

    #initialize first batch for this chromosome 
    chunk_counter = 0
    counter = 0
    X_batch = []
    Y_batch = [[] for t in range(1)]
    locations_batch = []
    total_keys = len(chrom_dict.keys())

    for loc, loc_dict in chrom_dict.items():
        counter += 1
        locations_batch.append(loc)
        Xs = loc_dict['Xs']
        Ys = loc_dict['Ys']
        assert len(Xs) == len(Ys)
        
        if len(Ys) > 1:
            stats_Ys = []
            for i, Y in enumerate(Ys):
                stats_Ys.append((i, Y, count_nonzero_labels(Y)))
            ordered_Ys = sorted(
                stats_Ys, 
                key = lambda x: x[2], 
                reverse = True
            )
            selected_index = ordered_Ys[0][0]
            
            # export corresponding data point
            X_batch.append(Xs[selected_index])
            Y_batch[0].append(Ys[selected_index])

        else:
            X_batch.append(Xs[0])
            Y_batch[0].append(Ys[0])
            
        if (counter == CHUNK_SIZE or 
            counter == total_keys - chunk_counter * CHUNK_SIZE):
            # export batch if it reaches chunk size
            # or last gene and non-empty
            print(f"Exporting batch {chrom} - {chunk_counter}")
            X_batch = np.asarray(X_batch).astype('int8')
            for t in range(1):
                Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')

            h5f2.create_dataset(
                chrom + '_X' + str(chunk_counter), 
                data=X_batch
            )
            h5f2.create_dataset(
                chrom + '_Y' + str(chunk_counter), 
                data=Y_batch
            )
            h5f2.create_dataset(
                chrom + '_Locations' + str(chunk_counter), 
                data=locations_batch
            )
            
            # update chrom chunk counter and overall counter
            chunk_counter += 1
            overall_chunk_counter += 1

            # reinitialize the batches
            counter = 0
            X_batch = []
            Y_batch = [[] for t in range(1)]  # TODO: remove this [[]] ...
            locations_batch = []


h5f2.attrs['n_datasets'] = overall_chunk_counter

h5f2.close()

print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
