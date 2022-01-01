# !import code; code.interact(local=vars())
import time
import logging

from tqdm.auto import trange

from splicing.finetune import finetune
from splicing.pretrain import pretrain

from splicing.utils.general_utils import print_topl_statistics, \
    shuffle_chromosomes, save_model


def pass_end(elapsed, predictions, targets, loss, opt, step, split):
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
            prediction_type=prediction_type, log_wandb=opt.wandb,
            step=step, split=split)

    logging.info('--- %s seconds ---' % (time.time() - start_time + elapsed))
    logging.info('\n---------------------------------------------------------')


def run_epoch(base_model, graph_model, full_model, datasets, criterion,
              optimizer, epoch, opt, split):
    start = time.time()
    if opt.pretrain or opt.save_feats or opt.test_baseline:

        # logging.info('Pretraining the base model.')

        predictions, targets, loss = pretrain(
            base_model, datasets[split], criterion, optimizer,
            epoch, opt, split)

    elif opt.finetune or opt.test_graph:
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
              criterion, optimizer, scheduler, opt):

    for epoch in trange(1, opt.epochs + 1):

        # print(f"Starting epoch {epoch}")

        if opt.finetune:
            datasets = shuffle_chromosomes(datasets)

        train_loss, valid_loss = 0, 0
        if not opt.load_gcn and not (opt.test_baseline or opt.test_graph):
            # TRAIN
            run_epoch(
                base_model, graph_model, full_model, datasets,
                criterion, optimizer, epoch, opt, 'train')

            #if epoch % opt.validation_interval == 0 and not opt.save_feats:

            #    # VALIDATE
            #    valid_predictions, valid_targets, valid_loss, elapsed = \
            #        run_epoch(base_model, graph_model, full_model, datasets,
            #                  criterion, optimizer, epoch, opt, 'valid')

            if not opt.save_feats and not (opt.test_baseline or opt.test_graph):
                
                # FULL VALIDATION
                valid_predictions, valid_targets, valid_loss, elapsed = \
                    run_epoch(base_model, graph_model, full_model, datasets,
                              criterion, optimizer, epoch, opt, 'valid')

                # FULL TEST
                if opt.test:
                    test_predictions, test_targets, test_loss, elapsed = \
                        run_epoch(base_model, graph_model, full_model, datasets,
                                criterion, optimizer, epoch, opt, 'test')

                pass_end(
                    elapsed, valid_predictions.numpy(), valid_targets.numpy(),
                    valid_loss, opt, split='full_valid', step=epoch)

                if opt.test:
                    pass_end(
                        elapsed, test_predictions.numpy(), test_targets.numpy(),
                        test_loss, opt, split='full_test', step=epoch)

                if opt.pretrain:
                    save_model(opt, epoch, base_model, model_type='base')
                else:
                    save_model(opt, epoch, graph_model, model_type='graph')
                    save_model(opt, epoch, full_model, model_type='full')

    if scheduler:
        if ((opt.pretrain and opt.cnn_sched in ["multisteplr", "steplr"]) or
            (opt.finetune and opt.ft_sched in ["multisteplr", "steplr"])):
            scheduler.step()
        elif ((opt.pretrain and opt.cnn_sched == "reducelr") or
            (opt.finetune and opt.ft_sched == "reducelr")):
            scheduler.step(valid_loss)

    # TEST
    if opt.save_feats:  # hacky
        run_epoch(
            base_model, graph_model, full_model, datasets, criterion,
            optimizer, chromosome, opt, 'test')
        run_epoch(
            base_model, graph_model, full_model, datasets, criterion,
            optimizer, chromosome, opt, 'valid')
    else:
        test_predictions, test_targets, test_loss, elapsed = run_epoch(
            base_model, graph_model, full_model, datasets,
            criterion, optimizer, opt.epochs, opt, 'test')
        pass_end(
            elapsed, test_predictions.numpy(), test_targets.numpy(),
            test_loss, opt, step=1, split='test')
