import torch
from tqdm import tqdm

from splicing.utils.general_utils import SPLIT2DESC, IX2CHR, save_feats
from splicing.utils.wandb_utils import report_wandb
from splicing.utils.spliceai_utils import get_data


def pretrain(base_model, data_file, criterion, optimizer, epoch, opt, split):
    if split == 'train':
        base_model.train()
    else:
        base_model.eval()

    all_preds = torch.Tensor().cpu()
    all_targets = torch.Tensor().cpu()
    all_x_f = torch.Tensor().cpu()
    all_locs = []

    total_loss = 0
    batch_count = 0
    batch_size = opt.batch_size

    for chromosome in opt.chromosomes[split]:
        dataloader = get_data(
            data_file, chromosome, opt.context_length,
            opt.batch_size
        )

        n_instances = len(dataloader.dataset)
        
        n_batches = n_instances // batch_size

        for batch, (X, y, loc) in enumerate(
                tqdm(dataloader, total=n_batches,
                    desc=SPLIT2DESC[split], leave=False)):

            if opt.pretrain and split == 'train':
                optimizer.zero_grad()

            y_hat, x, _ = base_model(X, save_feats=opt.save_feats)

            if opt.test_baseline:
                y = y.cpu()

            loss = criterion(y_hat, y)

            if opt.pretrain and split == 'train':
                loss.backward()
                optimizer.step()

            # Updates
            if split != 'train':
                total_loss += loss.item()
                all_preds = torch.cat((all_preds, y_hat.cpu().data), 0)
                all_targets = torch.cat((all_targets, y.cpu().data), 0)

            if opt.save_feats:
                all_x_f = torch.cat((all_x_f, x.detach().cpu()), 0)
                loc = list(loc.detach().cpu().numpy().astype(int))
                all_locs.extend(loc)

            if split == 'train' and batch_count % opt.log_interval == 0 \
                    and opt.wandb:
                report_wandb(y_hat, y, loss, opt, split)

            if split == 'valid' and batch_count % opt.validation_interval == 0 \
                    and opt.wandb:
                report_wandb(y_hat, y, loss, opt, split)

            batch_count+=1

        if opt.save_feats:
            save_feats(
                opt.model_name, split, all_targets, all_locs, all_x_f, 
                chromosome, epoch)
            all_targets = torch.Tensor().cpu()
            all_x_f = torch.Tensor().cpu()
            all_locs = []

    return all_preds, all_targets, total_loss
