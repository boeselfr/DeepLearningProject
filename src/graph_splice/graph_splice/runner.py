# !import code; code.interact(local=vars())
from os import path
import time
import logging
from collections import Counter

import numpy as np
import h5py
from tqdm.auto import trange
import torch

from finetune import finetune
from pretrain import pretrain
from utils.evals import compute_metrics, SaveLogger
from utils.util_methods import directory_setup

from splicing.utils.utils import get_data, print_topl_statistics


def pass_end(model, elapsed, predictions, targets, loss,
             opt, pass_n, split):

    print('----------------------------------------------------------')
    logging.info('\nValidation set metrics:')

    is_expr = targets.sum(axis=(1, 2)) >= 1

    for ix, prediction_type in enumerate(['Acceptor', 'Donor']):
        targets_ix = targets[is_expr, ix + 1, :].flatten()
        predictions_ix = predictions[is_expr, ix + 1, :].flatten()

        total_len = len(targets)

        logging.info(f'Total loss: {loss / total_len:>12f}')
        logging.info("\nAcceptor:")
        print_topl_statistics(
            targets_ix, predictions_ix, loss=loss,
            prediction_type=prediction_type, test=split != 'train')

    logging.info(
        '--- %s seconds ---' % elapsed)

    print('----------------------------------------------------------')

    model_dir = path.join(opt.results_dir, 'Models')
    directory_setup(model_dir)
    torch.save(
        model.state_dict(),
        path.join(
            model_dir,
            f'SpliceAI{opt.context_length}_e{pass_n}_g{opt.model_index}.h5')
    )


def run_epoch(base_model, chrome_model, dataloader, criterion, optimizer,
              epoch, opt, split):
    start = time.time()
    if opt.pretrain or opt.save_feats:

        logging.info('Pretraining the base model.')

        predictions, targets, loss = pretrain(
            base_model, dataloader, criterion, optimizer, epoch, opt, split)

    elif not opt.save_feats:
        logging.info('Fine-tuning the graph-based model')
        predictions, targets, loss = finetune(
            chrome_model, dataloader, criterion, optimizer, epoch, opt, split)

    elapsed = ((time.time() - start) / 60)
    logging.info('\n({split}) elapse: {elapse:3.3f} min'.format(
        split=split, elapse=elapsed))
    logging.info('Total epoch loss: {loss:3.3f}'.format(loss=loss))

    return predictions, targets, loss, elapsed


def run_model(base_model, chrome_model, data_file,
              criterion, optimizer, scheduler, opt, logger):

    # if not opt.save_feats:
    #     save_logger = SaveLogger(opt.model_name)

    for epoch in trange(1, opt.epochs + 1):

        if scheduler and opt.lr_decay2 > 0:
            scheduler.step()

        train_loss, valid_loss = 0, 0
        if not opt.load_gcn and not opt.test_only:
            # TRAIN
            train_data = get_data(
                data_file, opt.idx_train, opt.context_length, opt.batch_size)
            train_predictions, train_targets, train_loss, elapsed = run_epoch(
                base_model, chrome_model, train_data,
                criterion, optimizer, epoch, opt, 'train')

            if epoch % opt.validation_interval == 0:

                # VALIDATE
                valid_data = get_data(
                    data_file, opt.idx_valid, opt.context_length,
                    opt.batch_size)
                valid_predictions, valid_targets, valid_loss, elapsed = \
                    run_epoch(base_model, chrome_model, valid_data,
                              criterion, optimizer, epoch, opt, 'valid')

            if epoch % len(opt.idx_train) == 0:
                # FULL VALIDATION
                valid_data = get_data(
                    data_file, opt.idx_valid, opt.context_length,
                    opt.batch_size, full=True)
                valid_predictions, valid_targets, valid_loss, elapsed = \
                    run_epoch(base_model, chrome_model, valid_data,
                              criterion, optimizer, epoch, opt, 'valid')
                pass_end(
                    base_model, elapsed, valid_predictions.numpy(),
                    valid_targets.numpy(), valid_loss,
                    opt, epoch // len(opt.idx_train), 'valid')

        # LOGGING
        # best_valid, best_test = logger.evaluate(
        #     train_metrics, valid_metrics, test_metrics=None,
        #     epoch=epoch, num_params=opt.total_num_parameters)

        # if not opt.save_feats:
        #     save_logger.save(
        #         epoch, opt, window_model, chrome_model,
        #         valid_loss, valid_metrics_sum, valid_metrics_sums,
        #         valid_preds, valid_targs)
        #     save_logger.log('valid.log', epoch, valid_loss, valid_metrics)
        #     save_logger.log('train.log', epoch, train_loss, train_metrics)

        # print('best loss epoch: ' + str(save_logger.best_loss_epoch))
        # print(opt.model_name)

    # TEST
    h5f = h5py.File(path.join(
        opt.splice_data_dir, 'dataset_test_0.h5'), 'r')
    test_data = get_data(
        h5f, list(range(len(h5f.keys()) // 2)),
        opt.context_length, opt.batch_size)
    test_preds, test_targs, test_loss, elapsed = run_epoch(
        base_model, chrome_model, test_data,
        criterion, optimizer, opt.epochs, opt, 'test')
    # if not opt.save_feats:
    #     save_logger.log('test.log', opt.epochs, test_loss, test_metrics)
