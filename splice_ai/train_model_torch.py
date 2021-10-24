###############################################################################
# This file contains the code to train the SpliceAI model.
###############################################################################

import coloredlogs
import sys
import time
import logging

import numpy as np
import h5py
from tqdm import tqdm

# from multi_gpu import make_parallel
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
from splice_ai_torch import SpliceAI, categorical_crossentropy_2d
from utils import get_architecture, CL_max, SL, print_topl_statistics
from constants import data_dir
from splice_dataset import SpliceDataset

coloredlogs.install(level=logging.INFO)

# from splice_ai_torch import categorical_crossentropy_2d
# from splice_dataset import SpliceDataset


assert int(sys.argv[1]) in [80, 400, 2000, 10000]
CL = int(sys.argv[1])


def validate(model, h5f, idxs, CL, N_GPUS, BATCH_SIZE):

    Y_true_1 = []
    Y_true_2 = []
    Y_pred_1 = []
    Y_pred_2 = []

    for idx in tqdm(idxs):
        X = h5f['X' + str(idx)][:]
        Y = np.asarray((h5f['Y' + str(idx)][:]), np.float32)

        splice_dataset = SpliceDataset(X, Y, CL, N_GPUS)
        dataloader = DataLoader(splice_dataset, batch_size=BATCH_SIZE)

        Yp = np.zeros(shape=splice_dataset.Y.shape)
        m = 0
        loss = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                pred = model(X)
                preds = pred.cpu().numpy()
                Yp[m: m + len(preds)] = preds
                m += len(preds)

                loss += categorical_crossentropy_2d(pred, y).item()

        print(f"loss: {loss:>12f}")
        is_expr = splice_dataset.get_expr()

        Y_true_1.extend(splice_dataset.get_true(1, is_expr))
        Y_true_2.extend(splice_dataset.get_true(2, is_expr))
        Y_pred_1.extend(Yp[is_expr, :, 1].flatten())
        Y_pred_2.extend(Yp[is_expr, :, 2].flatten())

    print("\nAcceptor:")
    print_topl_statistics(np.asarray(Y_true_1), np.asarray(Y_pred_1))

    print("\nDonor:")
    print_topl_statistics(np.asarray(Y_true_2), np.asarray(Y_pred_2))


assert int(sys.argv[1]) in [80, 400, 2000, 10000]

###############################################################################
# Model
###############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

L = 32
N_GPUS = 1

# Hyper-parameters:
# L: Number of convolution kernels
# W: Convolution window size in each residual unit
# AR: Atrous rate in each residual unit
W, AR, BATCH_SIZE = get_architecture(sys.argv[1], N_GPUS)

CL = 2 * np.sum(AR * (W - 1))
assert CL <= CL_max and CL == int(sys.argv[1])

logging.debug(f' Context nucleotides: {CL} ')
logging.debug(f' Sequence length (output): {SL} ')

model = SpliceAI(L, W, AR).to(device)
summary(model, input_size=(4, CL + SL), batch_size=BATCH_SIZE)
optimizer = optim.Adam(model.parameters())

###############################################################################
# Training and validation
###############################################################################

h5f = h5py.File(data_dir + 'dataset_train_all.h5', 'r')

num_idx = len(h5f.keys()) // 2
idx_all = np.random.permutation(num_idx)
idx_train = idx_all[:int(0.9 * num_idx)]
idx_valid = idx_all[int(0.9 * num_idx):]

EPOCH_NUM = 10 * len(idx_train)

start_time = time.time()

for epoch_num in range(EPOCH_NUM):

    logging.info(f'On epoch number {epoch_num} / {EPOCH_NUM}.')

    idx = np.random.choice(idx_train)

    X = h5f['X' + str(idx)][:]
    Y = np.asarray(h5f['Y' + str(idx)][:], dtype=np.float32)

    splice_dataset = SpliceDataset(X, Y, CL, N_GPUS)

    dataloader = DataLoader(splice_dataset, batch_size=BATCH_SIZE)

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = categorical_crossentropy_2d(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # if loss > 0.4 and (batch > 0 or epoch_num > 0):
            #     y = y.cpu().numpy()
            #     is_expr = y.sum(axis=(1, 2)) >= 1
            #     preds = pred.detach().cpu().numpy()
            #
            #     print(y)
            #     print(preds)
            #
            #     print("\nAcceptor:")
            #     print_topl_statistics(
            #         y[is_expr, :, 1].flatten(),
            #         preds[is_expr, :, 1].flatten())
            #
            #     print("\nDonor:")
            #     print_topl_statistics(
            #         y[is_expr, :, 2].flatten(),
            #         preds[is_expr, :, 2].flatten())

    if (epoch_num + 1) % len(idx_train) == 0:
    # if (epoch_num + 1) % 4 == 0:
        # Printing metrics (see utils.py for details)

        print('--------------------------------------------------------------')
        logging.info('\nValidation set metrics:')

        validate(model, h5f, idx_valid, CL, N_GPUS, BATCH_SIZE)

        logging.info(
            'Learning rate: %.5f' % (optimizer.param_groups[0]['lr']))
        logging.info(
            '--- %s seconds ---' % (time.time() - start_time))
        start_time = time.time()

        print('--------------------------------------------------------------')

        # model.save(
        #     './Models/SpliceAI' + sys.argv[1] + '_g' + sys.argv[2] + '.h5')

        if (epoch_num + 1) >= 6 * len(idx_train):
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5

h5f.close()

###############################################################################
