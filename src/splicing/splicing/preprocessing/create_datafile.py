###############################################################################
"""
This parser takes as input the text files gtex_dataset.txt and
gtex_sequence.txt, and produces a .h5 file datafile_{}_{}.h5,
which will be later processed to create dataset_{}_{}.h5. The file
dataset_{}_{}.h5 will have datapoints of the form (X,Y), and can be
understood by Keras models.
"""
###############################################################################

import argparse

import numpy as np
import re
import time
import h5py

from splicing.utils.constants import CL_max, data_dir, sequence, splice_table


start_time = time.time()

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

assert group in ['train', 'test', 'all']
assert paralog in ['0', '1', 'all']

if group == 'train':
    CHROM_GROUP = ['chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                   'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                   'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']
elif group == 'test':
    CHROM_GROUP = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
else:
    CHROM_GROUP = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9',
                   'chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21',
                   'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
                   'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY']

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

fpr2 = open(sequence, 'r')

with open(splice_table, 'r') as fpr1:
    for line1 in fpr1:

        line2 = fpr2.readline()

        data1 = re.split('\n|\t', line1)[:-1]
        data2 = re.split('\n|\t|:|-', line2)[:-1]

        assert data1[2] == data2[0]
        assert int(data1[4]) == int(data2[1]) + CL_max // 2 + 1
        assert int(data1[5]) == int(data2[2]) - CL_max // 2

        if data1[2] not in CHROM_GROUP:
            continue

        if (paralog != data1[1]) and (paralog != 'all'):
            continue

        NAME.append(data1[0])
        PARALOG.append(int(data1[1]))
        CHROM.append(data1[2])
        STRAND.append(data1[3])
        TX_START.append(data1[4])
        TX_END.append(data1[5])
        JN_START.append(data1[6::2])
        JN_END.append(data1[7::2])
        SEQ.append(data2[3])

fpr1.close()
fpr2.close()

###############################################################################

h5f = h5py.File(data_dir + 'datafile_' + group + '_' + paralog + '.h5', 'w')

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
