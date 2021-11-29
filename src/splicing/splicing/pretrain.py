import torch
from tqdm import tqdm
import wandb

from sklearn.metrics import average_precision_score

from splicing.utils import graph_utils
from splicing.utils.graph_utils import split2desc
from splicing.utils.utils import get_data, IX2CHR


def report_wandb(predictions, targets, loss, opt, split):

    # sums_true = y.sum(axis=(0, 2))
    # sums_pred = predictions.sum(axis=(0, 2))
    #
    # total = sums_true.sum()

    is_expr = targets.sum(axis=(1, 2)) >= 1
    auprcs = {}
    for ix, prediction_type in enumerate(['Acceptor', 'Donor']):
        targets_ix = targets[
                     is_expr, ix + 1, :].flatten().detach().cpu().numpy()
        predictions_ix = predictions[
                         is_expr, ix + 1, :].flatten().detach().cpu().numpy()
        auprcs[prediction_type] = average_precision_score(
            targets_ix, predictions_ix)

    wandb.log({
        f'{split}/loss': loss.item() / opt.batch_size,
        f'{split}/Acceptor AUPRC': auprcs['Acceptor'],
        f'{split}/Donor AUPRC': auprcs['Donor'],
        # f'{split}/true inactive': sums_true[0] / total,
        # f'{split}/true acceptors': sums_true[1] / total,
        # f'{split}/true donors': sums_true[2] / total,
        # f'{split}/predicted inactive': sums_pred[0] / sums_true[0],
        # f'{split}/predicted acceptors': sums_pred[1] / sums_true[1],
        # f'{split}/predicted donors': sums_pred[2] / sums_true[2],
        # f'{split}/proportion of epoch done': batch / (size // batch_size),
    })


def pretrain(base_model, data_file, criterion, optimizer, epoch, opt, split):
    if split == 'train':
        base_model.train()
    else:
        base_model.eval()

    if opt.save_feats:
        # go one by one to avoid large memory consumption
        dataloader = get_data(
            data_file, opt.chromosomes[split], opt.context_length,
            opt.batch_size, full=False,
            chromosome=IX2CHR(opt.chromosomes[split][epoch - 1]))
    else:
        load_full = (epoch > 0 and (epoch % opt.full_validation_interval == 0)
                     and split in ['valid', 'test'])
        dataloader = get_data(
            data_file, opt.chromosomes[split], opt.context_length,
            opt.batch_size, full=load_full)

    n_instances = len(dataloader.dataset)
    all_predictions = torch.zeros(n_instances, 3, opt.window_size).cpu()
    all_targets = torch.zeros(n_instances, 3, opt.window_size).cpu()
    all_x_f = torch.Tensor().cpu()
    all_locs, all_chroms = [], []

    total_loss = 0
    batch_size = opt.batch_size
    n_batches = n_instances // batch_size

    for batch, (X, y, loc) in enumerate(
            tqdm(dataloader, total=n_batches,
                 desc=split2desc[split], leave=False)):

        if opt.pretrain and split == 'train':
            optimizer.zero_grad()

        y_hat, x, _ = base_model(X)

        loss = criterion(y_hat, y)

        if opt.pretrain and split == 'train':
            loss.backward()
            optimizer.step()

        # Updates
        total_loss += loss.item()
        start_idx, end_idx = batch * batch_size, (batch + 1) * batch_size
        all_predictions[start_idx: end_idx, :] = y_hat.cpu().data
        all_targets[start_idx: end_idx, :] = y.cpu().data

        if opt.save_feats:
            all_x_f = torch.cat((all_x_f, x.detach().cpu()), 0)
            loc = list(loc.detach().cpu().numpy().astype(int))
            all_locs.extend(loc)

        if split == 'train' and batch % opt.log_interval == 0 \
                and opt.wandb:
            report_wandb(y_hat, y, loss, opt, split)

        if split == 'valid' and batch % opt.validation_interval == 0 \
                and opt.wandb:
            report_wandb(y_hat, y, loss, opt, split)

    if opt.save_feats:
        graph_utils.save_feats(
            opt.model_name, split, all_targets, all_locs, all_x_f,
            opt.chromosomes[split][epoch - 1], epoch)

    return all_predictions, all_targets, total_loss
