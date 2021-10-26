###############################################################################
# This file contains code to test the SpliceAI model.
###############################################################################

import sys
import time
import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.splicing.utils.utils import print_topl_statistics
from src.splicing.utils.constants import data_dir
from src.splicing.data_models.splice_dataset import SpliceDataset

# TODO

assert int(sys.argv[1]) in [80, 400, 2000, 10000]
CL = int(sys.argv[1])

###############################################################################
# Load model and test data
###############################################################################

BATCH_SIZE = 6
version = [1, 2, 3, 4, 5]

model = [[] for v in range(len(version))]

for v in range(len(version)):
    model[v] = torch.load(
        'Models/SpliceAI' + str(CL) + '_g' + str(version[v]) + '.h5')

h5f = h5py.File(data_dir + 'dataset' + '_' + 'test'
                + '_' + '0' + '.h5', 'r')

num_idx = len(h5f.keys()) // 2

###############################################################################
# Model testing
###############################################################################

start_time = time.time()

output_class_labels = ['Null', 'Acceptor', 'Donor']
# The three neurons per output correspond to no splicing, splice acceptor (AG)
# and splice donor (GT) respectively.

for output_class in [1, 2]:

    Y_true = [[] for t in range(1)]
    Y_pred = [[] for t in range(1)]

    for idx in range(num_idx):

        X = h5f['X' + str(idx)][:]
        Y = h5f['Y' + str(idx)][:]

        splice_dataset = SpliceDataset(X, Y, CL, N_GPUS=1)
        dataloader = DataLoader(splice_dataset, batch_size=BATCH_SIZE)

        Yps = [np.zeros(Yc[0].shape) for t in range(1)]

        for v in range(len(version)):

            Yp = model[v].predict(Xc, batch_size=BATCH_SIZE)

            Yps += Yp / len(version)
        # Ensemble averaging (mean of the ensemble predictions is used)

        is_expr = (Yc.sum(axis=(1, 2)) >= 1)

        Y_true.extend(Yc[is_expr, output_class, :].flatten())
        Y_pred.extend(Yps[is_expr, output_class, :].flatten())

    print("\n%s:" % (output_class_labels[output_class]))

    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    print_topl_statistics(Y_true, Y_pred)

h5f.close()

print("--- %s seconds ---" % (time.time() - start_time))
print("--------------------------------------------------------------")

###############################################################################
