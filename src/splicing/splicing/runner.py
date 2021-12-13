# !import code; code.interact(local=vars())
import time
import logging

from tqdm.auto import trange

from splicing.finetune import finetune
from splicing.pretrain import pretrain

from splicing.utils.utils import print_topl_statistics
from splicing.utils.evals import SaveLogger


def pass_end(elapsed, predictions, targets, loss, opt):
    start_time = time.time()
    logging.info('\n---------------------------------------------------------')
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
            prediction_type=prediction_type, log_wandb=opt.wandb)

    logging.info('--- %s seconds ---' % (time.time() - start_time + elapsed))
    logging.info('\n---------------------------------------------------------')


def run_epoch(base_model, graph_model, full_model, datasets, criterion,
              optimizer, epoch, opt, split):
    start = time.time()
    if opt.pretrain or opt.save_feats:

        # logging.info('Pretraining the base model.')

        predictions, targets, loss = pretrain(
            base_model, datasets[split], criterion, optimizer,
            epoch, opt, split)

    elif opt.finetune:
        # logging.info('Fine-tuning the graph-based model')
        predictions, targets, loss = finetune(
            graph_model, full_model, datasets[split], criterion, optimizer,
            epoch, opt, split)

    elapsed = (time.time() - start) / 60
    # logging.info('\n({split}) elapse: {elapse:3.3f} min'.format(
    #     split=split, elapse=elapsed))
    # logging.info('Total epoch loss: {loss:3.3f}'.format(loss=loss))

    return predictions, targets, loss, elapsed


def run_model(base_model, graph_model, full_model, datasets,
              criterion, optimizer, scheduler, opt, logger):

    if not opt.save_feats:
        save_logger = SaveLogger(opt.model_name)

    for epoch in trange(1, opt.epochs + 1):

        print(f"Starting epoch {epoch}")

        if scheduler and (opt.pretrain or opt.lr_decay > 0):
            scheduler.step()

        train_loss, valid_loss = 0, 0
        if not opt.load_gcn and not opt.test_only:
            # TRAIN
            train_predictions, train_targets, train_loss, elapsed = run_epoch(
                base_model, graph_model, full_model, datasets,
                criterion, optimizer, epoch, opt, 'train')

            if epoch % opt.validation_interval == 0 and not opt.save_feats:

                # VALIDATE
                valid_predictions, valid_targets, valid_loss, elapsed = \
                    run_epoch(base_model, graph_model, full_model, datasets,
                              criterion, optimizer, epoch, opt, 'valid')

            if epoch % opt.full_validation_interval == 0 \
                    and not opt.save_feats:
                # FULL VALIDATION
                valid_predictions, valid_targets, valid_loss, elapsed = \
                    run_epoch(base_model, graph_model, full_model, datasets,
                              criterion, optimizer, epoch, opt, 'valid')
                pass_end(
                    elapsed, valid_predictions.numpy(), valid_targets.numpy(),
                    valid_loss, opt)

                if not opt.save_feats:
                    save_logger.save(
                        epoch, opt, base_model, graph_model, full_model,
                        valid_loss, valid_predictions, valid_targets)
                    # save_logger.log('valid.log', epoch, valid_loss)
                    # save_logger.log('train.log', epoch, train_loss)

        # LOGGING
        # best_valid, best_test = logger.evaluate(
        #     train_metrics, valid_metrics, test_metrics=None,
        #     epoch=epoch, num_params=opt.total_num_parameters)

        # print('best loss epoch: ' + str(save_logger.best_loss_epoch))
        # print(opt.model_name)

    # TEST
    if opt.save_feats:  # hacky
        for chromosome in range(len(opt.chromosomes['test'])):
            run_epoch(
                base_model, graph_model, full_model, datasets, criterion,
                optimizer, chromosome, opt, 'test')
        for chromosome in range(len(opt.chromosomes['valid'])):
            run_epoch(
                base_model, graph_model, full_model, datasets, criterion,
                optimizer, chromosome, opt, 'valid')
    else:
        test_predictions, test_targets, test_loss, elapsed = run_epoch(
            base_model, graph_model, full_model, datasets,
            criterion, optimizer, opt.epochs, opt, 'test')
        pass_end(
            elapsed, test_predictions.numpy(), test_targets.numpy(),
            test_loss, opt)
        # save_logger.log('test.log', opt.epochs, test_loss, test_metrics)
