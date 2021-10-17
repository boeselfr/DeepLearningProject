###############################################################################
# This file contains the code to train the SpliceAI model.
###############################################################################

import numpy as np
import sys
import time
import h5py
import keras.backend as kb
import tensorflow as tf
from spliceai import *
from utils import *
from multi_gpu import *
from constants import *
import logging, coloredlogs, tqdm

coloredlogs.install()

assert int(sys.argv[1]) in [80, 400, 2000, 10000]

###############################################################################
# Model
###############################################################################

L = 32
N_GPUS = 2

# Hyper-parameters:
# L: Number of convolution kernels
# W: Convolution window size in each residual unit
# AR: Atrous rate in each residual unit
W, AR, BATCH_SIZE = get_architecture(sys.argv[1], N_GPUS)

CL = 2 * np.sum(AR * (W - 1))
assert CL <= CL_max and CL == int(sys.argv[1])

logging.debug(f' Context nucleotides: {CL} ')
logging.debug(f' Sequence length (output): {SL} ')

model = SpliceAI(L, W, AR)
model.summary()
model_m = make_parallel(model, N_GPUS)
model_m.compile(loss=categorical_crossentropy_2d, optimizer='adam')

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

    logging.info(f'On epoch number {epoch_num}.')

    idx = np.random.choice(idx_train)

    X = h5f['X' + str(idx)][:]
    Y = tf.cast(h5f['Y' + str(idx)][:], tf.float32)

    Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
    model_m.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)

    if (epoch_num+1) % len(idx_train) == 0:
        # Printing metrics (see utils.py for details)

        print('--------------------------------------------------------------')
        logging.debug('\nValidation set metrics:')

        validate(model_m, h5f, idx_valid, CL, N_GPUS, BATCH_SIZE)
        # validate(model_m, h5f, idx_train, CL, N_GPUS, BATCH_SIZE)

        logging.debug('Learning rate: %.5f' % (kb.get_value(
            model_m.optimizer.lr)))
        logging.debug('--- %s seconds ---' % (time.time() - start_time))
        start_time = time.time()

        print('--------------------------------------------------------------')

        model.save(
            './Models/SpliceAI' + sys.argv[1] + '_g' + sys.argv[2] + '.h5')

        if (epoch_num + 1) >= 6 * len(idx_train):
            kb.set_value(
                model_m.optimizer.lr, 0.5 * kb.get_value(model_m.optimizer.lr))
            # Learning rate decay

h5f.close()

###############################################################################
