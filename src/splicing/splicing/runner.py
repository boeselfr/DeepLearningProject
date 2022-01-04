# !import code; code.interact(local=vars())
import time

from tqdm.auto import trange

from splicing.finetune import finetune
from splicing.pretrain import pretrain

from splicing.utils.general_utils import shuffle_chromosomes, save_model


def run_epoch(base_model, graph_model, full_model, datasets, criterion,
              optimizer, epoch, opt, split):
    start = time.time()
    if opt.pretrain or opt.save_feats or opt.test_baseline:

        # logging.info('Pretraining the base model.')

        combined_scores = pretrain(
            base_model, datasets[split], criterion, optimizer,
            epoch, opt, split)

    elif opt.finetune or opt.test_graph:
        # logging.info('Fine-tuning the graph-based model')
        combined_scores = finetune(
            graph_model, full_model, datasets[split], criterion, optimizer,
            epoch, opt, split)

    elapsed = (time.time() - start) / 60
    # logging.info('\n({split}) elapse: {elapse:3.3f} min'.format(
    #     split=split, elapse=elapsed))
    # logging.info('Total epoch loss: {loss:3.3f}'.format(loss=loss))

    return combined_scores, elapsed


def run_model(base_model, graph_model, full_model, datasets,
              criterion, optimizer, scheduler, opt):
    test_warmup = 3 if opt.finetune else 6
    for epoch in trange(1, opt.epochs + 1):

        if opt.finetune:
            # shuffle the sequence of the chromosomes to prevent overfitting
            datasets = shuffle_chromosomes(datasets)

        if not opt.load_gcn and not (opt.test_baseline or opt.test_graph):
            # TRAIN
            run_epoch(
                base_model, graph_model, full_model, datasets,
                criterion, optimizer, epoch, opt, 'train')

            if not opt.save_feats and not (opt.test_baseline or opt.test_graph):
                # FULL VALIDATION
                # run the validation on the complete validation dataset
                # after every pass over the entire genome
                valid_scores, elapsed = \
                    run_epoch(base_model, graph_model, full_model, datasets,
                              criterion, optimizer, epoch, opt, 'valid')

                # FULL TEST
                # run the model on the test dataset
                if epoch > test_warmup:
                    run_epoch(base_model, graph_model, full_model, datasets,
                              criterion, optimizer, epoch, opt, 'test')

                # save the model
                if opt.pretrain:
                    save_model(opt, epoch, base_model, model_type='base')
                else:
                    save_model(opt, epoch, graph_model, model_type='graph')
                    save_model(opt, epoch, full_model, model_type='full')

        if scheduler:
            # update the learning rate
            if ((opt.pretrain and opt.cnn_sched in ["multisteplr", "steplr"]) or
                (opt.finetune and opt.ft_sched in ["multisteplr", "steplr"])):
                scheduler.step()
            elif ((opt.pretrain and opt.cnn_sched == "reducelr") or
                (opt.finetune and opt.ft_sched == "reducelr")):
                scheduler.step(valid_scores["avg_loss"])

    if opt.save_feats:  # hacky
        # save the features of the validation and test datasets as well
        # since this is not done in the loop above
        run_epoch(
            base_model, graph_model, full_model, datasets, criterion,
            optimizer, 0, opt, 'test')
        run_epoch(
            base_model, graph_model, full_model, datasets, criterion,
            optimizer, 0, opt, 'valid')
