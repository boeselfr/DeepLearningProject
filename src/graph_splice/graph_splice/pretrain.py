import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from utils import util_methods
from utils.util_methods import split2description
from splicing.utils.constants import SL


def report_wandb_train(predictions, y, loss, opt):

    sums_true = y.sum(axis=(0, 2))
    sums_pred = predictions.sum(axis=(0, 2))

    total = sums_true.sum()
    wandb.log({
        'loss': loss.item() / opt.batch_size,
        # 'true inactive': sums_true[0] / total,
        'true acceptors': sums_true[1] / total,
        'true donors': sums_true[2] / total,
        # 'predicted inactive': sums_pred[0] / sums_true[0],
        'predicted acceptors': sums_pred[1] / sums_true[1],
        'predicted donors': sums_pred[2] / sums_true[2],
        # 'proportion of epoch done': batch / (size // batch_size),
    })


def pretrain(base_model, dataloader, criterion, optimizer, epoch, opt, split):
    if split == 'train':
        base_model.train()
    else:
        base_model.eval()

    total_loss = 0

    n_instances = len(dataloader.dataset)
    all_predictions = torch.zeros(n_instances, 3, SL).cpu()
    all_targets = torch.zeros(n_instances, 3, SL).cpu()
    all_x_f = torch.Tensor().cpu()
    all_locs = []

    batch_size = opt.batch_size
    n_batches = n_instances // batch_size

    pbar = tqdm(total=n_batches, mininterval=0.5,
                desc=split2description[split], leave=False)

    for batch, (X, y, loc) in enumerate(dataloader):
        pbar.update()

        if opt.pretrain and split == 'train':
            optimizer.zero_grad()

        y_hat, x, _ = base_model(X)
        predictions = F.softmax(y_hat, dim=1)
        loss = criterion(predictions, y)

        if opt.pretrain and split == 'train':
            loss.backward()
            optimizer.step()

        # Updates
        total_loss += loss.item()
        start_idx, end_idx = batch * batch_size, (batch + 1) * batch_size
        all_predictions[start_idx: end_idx, :] = predictions.cpu().data
        all_targets[start_idx: end_idx, :] = y.cpu().data

        if opt.save_feats:
            all_x_f = torch.cat((all_x_f, x.detach().cpu()), 0)
            for loc_i in loc:
                all_locs.append(loc_i)

        if split == 'train' and batch % opt.log_interval == 0:
            report_wandb_train(predictions, y, loss, opt)

    if opt.save_feats:
        util_methods.save_feats(
            opt.model_name, split, all_targets, all_locs, all_x_f)

    pbar.close()

    return all_predictions, all_targets, total_loss
