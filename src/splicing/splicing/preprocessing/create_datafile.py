###############################################################################
"""
This parser takes as input the text files gtex_dataset.txt and
gtex_sequence.txt, and produces a .h5 file datafile_{}_{}.h5,
which will be later processed to create dataset_{}_{}.h5. The file
dataset_{}_{}.h5 will have datapoints of the form (X,Y).
"""
###############################################################################

import argparse

import numpy as np
import re
import time
import h5py
import csv
import os
import yaml

start_time = time.time()

###############################################################################
# Parsing Args
###############################################################################

parser = argparse.ArgumentParser(
    description='Create the data files from the gtex dataset '
                'and the DNA sequence.')
parser.add_argument(
    '-g', '--group', dest='group', type=str,
    help='The chromosome group to process. One of ["train", "test", "all"].')
parser.add_argument(
    '-p', '--paralog', dest='paralog', type=str,
    help='Whether to include the genes with paralogs or not.')

args = parser.parse_args()

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

# input
SPLICE_TABLE_PATH = os.path.join(
    DATA_DIR,
    config['RAW_DATA']['splice_table']
)

REF_GENOME_PATH = os.path.join(
    DATA_DIR,
    config['RAW_DATA']['genome']
)

CHROM_SIZE_FILE = os.path.join(
    config['DATA_DIRECTORY'],
    config['RAW_DATA']['chrom_sizes']
)

# data pipeline config
CL_MAX = config['DATA_PIPELINE']['context_length']

TRAIN_CHROMS = config['DATA_PIPELINE']['train_chroms']
VALID_CHROMS = config['DATA_PIPELINE']['valid_chroms']
TEST_CHROMS = config['DATA_PIPELINE']['test_chroms']
ALL_CHROMS = TRAIN_CHROMS + VALID_CHROMS + TEST_CHROMS

INTERVAL = config['DATA_PIPELINE']['window_size']

if group == 'train':
    CHROM_GROUP = TRAIN_CHROMS
elif group == 'valid':
    CHROM_GROUP = VALID_CHROMS
elif group == 'test':
    CHROM_GROUP = TEST_CHROMS
else:
    CHROM_GROUP = ALL_CHROMS

# output paths
GENE_WINDOWS_PATH = os.path.join(
    DATA_DIR,
    config['DATA_PIPELINE']['output_dir'],
    f'gene_windows_{INTERVAL}.bed'
)

GRAPH_WINDOWS_PATH = os.path.join(
    DATA_DIR,
    config['DATA_PIPELINE']['output_dir'],
    f'graph_windows_{INTERVAL}.bed'
)

SEQUENCE_FILE_PATH = os.path.join(
    DATA_DIR,
    config['DATA_PIPELINE']['output_dir'],
    f'gtex_sequence_{INTERVAL}.txt'
)

DATAFILE_PATH = os.path.join(
    DATA_DIR,
    config['DATA_PIPELINE']['output_dir'],
    f'datafile_{group}_{paralog}_{INTERVAL}.h5'
)

###############################################################################
# Utils
###############################################################################

# CL length adjustment after alignment modification
CL_R = int(CL_MAX / 2) # 200
CL_L = int(CL_R + 1) # 201


def apply_adjustment(start, end):
    # round gene start/end down/up adjusting to graph
    adj_start = (start // INTERVAL) * INTERVAL
    adj_end = ((end // INTERVAL) + 1) * INTERVAL

    assert (adj_end - adj_start) % INTERVAL == 0, \
        "gene window index adjustment error"

    # add in extra context window for edge nucleotides
    cl_start = adj_start - CL_L
    cl_end = adj_end + CL_R
    return adj_start, adj_end, cl_start, cl_end


###############################################################################
# Gene window adjustment
###############################################################################

# Collecting chromosome length
lengths = {}
with open(CHROM_SIZE_FILE) as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter='\t',
                                fieldnames=['chrom_name', 'length'])
    for csv_row in csv_reader:
        lengths[csv_row['chrom_name']] = int(csv_row['length']) - INTERVAL

gene_windows_file_w = open(GENE_WINDOWS_PATH, 'w')

# produce sequences for each gene and export to sequences bed file
chrom_bin_dict = {}
with open(SPLICE_TABLE_PATH, 'r') as fpr1:
    for line1 in fpr1:
        data1 = re.split('\n|\t', line1)[:-1]

        # TODO: decide if we want to limit this by CHROM_GROUP
        chrom = data1[2]
        startint = int(data1[4])
        endint = int(data1[5])

        adj_start, adj_end, cl_start, cl_end = apply_adjustment(
            startint, endint
        )

        if cl_end >= lengths[chrom]:
            print("CHROMOSOME INDEX OVER THE MAX")
            break

        gene_windows_file_w.write(
            chrom + '\t' + str(cl_start) + '\t' + str(cl_end) + '\n')

        chrom_bins = chrom_bin_dict.get(chrom, [])
        for i in range(int((adj_end - adj_start) / INTERVAL)):
            chrom_bins.append(
                (adj_start + i * INTERVAL, adj_start + (i + 1) * INTERVAL))
        chrom_bin_dict[chrom] = chrom_bins

gene_windows_file_w.close()

# Export bed file with unique list of graph windows for each chromosome
graph_windows_file_w = open(GRAPH_WINDOWS_PATH, 'w')

for chrom, windows in chrom_bin_dict.items():
    unique_windows = sorted(list(set(windows)))
    for start, end in unique_windows:
        graph_windows_file_w.write(
            chrom + '\t' + str(start) + '\t' + str(end) + '\n')

graph_windows_file_w.close()

###############################################################################
# Create sequences from gene index bed file
###############################################################################

sys_command = f'bedtools getfasta ' \
              f'-bed {GENE_WINDOWS_PATH} ' \
              f'-fi {REF_GENOME_PATH} ' \
              f'-fo {SEQUENCE_FILE_PATH} -tab'
print(f"Executing syscommand {sys_command}")
os.system(sys_command)

###############################################################################
# Create datafile from splice table and sequences
###############################################################################

NAME = []  # Gene symbol
PARALOG = []  # 0 if no paralogs exist, 1 otherwise
CHROM = []  # Chromosome number
STRAND = []  # Strand in which the gene lies (+ or -)
TX_START = []  # Position where transcription starts
TX_END = []  # Position where transcription ends
JN_START = []  # Positions where gtex exons end
JN_END = []  # Positions where gtex exons start
SEQ = []  # Nucleotide sequence

fpr2 = open(SEQUENCE_FILE_PATH, 'r')

with open(SPLICE_TABLE_PATH, 'r') as fpr1:
    for line1 in fpr1:

        line2 = fpr2.readline()

        data1 = re.split('\n|\t', line1)[:-1]
        data2 = re.split('\n|\t|:|-', line2)[:-1]

        # recomputing adjusted indices for genes from splicing data
        startint = int(data1[4])
        endint = int(data1[5])

        adj_start, adj_end, cl_start, cl_end = apply_adjustment(
            startint, endint
        )

        assert data1[2] == data2[0]
        assert cl_start == int(data2[1])
        assert cl_end == int(data2[2])
        assert adj_start == int(data2[1]) + CL_MAX // 2 + 1
        assert adj_end == int(data2[2]) - CL_MAX // 2

        if data1[2] not in CHROM_GROUP:
            continue

        if (paralog != data1[1]) and (paralog != 'all'):
            continue

        # example: adj_start 5000, adj_end 10000, cl_start 4799, cl_end 10200

        NAME.append(data1[0])
        PARALOG.append(int(data1[1]))
        CHROM.append(data1[2])
        STRAND.append(data1[3])
        TX_START.append(str(adj_start))
        TX_END.append(str(adj_end))
        JN_START.append(data1[6::2])
        JN_END.append(data1[7::2])
        SEQ.append(data2[3])

fpr1.close()
fpr2.close()

###############################################################################

print(f"Exporting datafile to: {DATAFILE_PATH}")

h5f = h5py.File(DATAFILE_PATH, 'w')

h5f.create_dataset('NAME', data=NAME)
h5f.create_dataset('PARALOG', data=PARALOG)
h5f.create_dataset('CHROM', data=CHROM)
h5f.create_dataset('STRAND', data=STRAND)
h5f.create_dataset('TX_START', data=TX_START)
h5f.create_dataset('TX_END', data=TX_END)
h5f.create_dataset('JN_START', data=JN_START)
h5f.create_dataset('JN_END', data=JN_END)
h5f.create_dataset('SEQ', data=SEQ)

h5f.close()

print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
