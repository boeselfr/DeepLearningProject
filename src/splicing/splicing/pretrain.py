import torch
from tqdm import tqdm

from splicing.utils.general_utils import SPLIT2DESC, save_feats, \
    compute_scores, compute_average_scores
from splicing.utils.wandb_utils import report_wandb
from splicing.utils.spliceai_utils import get_data


def pretrain(base_model, data_file, criterion, optimizer, epoch, opt, split):
    if split == 'train':
        base_model.train()
    else:
        base_model.eval()

    all_locs = []

    total_loss = 0
    batch_count = 0
    batch_size = opt.batch_size

    scores = {}

    # loop over all the chromosomes
    for chromosome in opt.chromosomes[split]:

        # create the chromosome sequence dataset
        dataloader = get_data(
            data_file, chromosome, opt.context_length,
            opt.batch_size
        )

        n_instances = len(dataloader.dataset)
        
        n_batches = n_instances // batch_size

        # to store the predictions/nucleotide representations
        # of the entire chromosome
        all_preds = torch.zeros((n_instances, 3, opt.window_size)).cpu()
        all_targets = torch.zeros((n_instances, 3, opt.window_size)).cpu()
        all_x_f = torch.zeros((n_instances, opt.n_channels, opt.window_size)).cpu()
        count_ys = 0
        count_xs = 0

        if opt.pretrain:
            desc_prefix = "PRETRAIN"
        elif opt.save_feats:
            desc_prefix = "SAVE_FEATS"

        desc = f"{desc_prefix}: epoch {epoch}, split "\
            f"{SPLIT2DESC[split]}, chromosome {chromosome}"

        # loop over the chromosome in batches
        for batch, (X, y, loc) in enumerate(
                tqdm(dataloader, total=n_batches, desc=desc, leave=False)):

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
                all_preds[count_ys:count_ys + y_hat.shape[0]] = y_hat.cpu().data
                all_targets[count_ys:count_ys + y_hat.shape[0]] = y.cpu().data
                count_ys += y_hat.shape[0]

            # save the nucleotide representations
            if opt.save_feats:
                all_x_f[count_xs:count_xs + x.shape[0]] = x.detach().cpu()
                loc = list(loc.detach().cpu().numpy().astype(int))
                all_locs.extend(loc)
                count_xs += x.shape[0]

            if split == 'train' and batch_count % opt.log_interval == 0 \
                    and opt.wandb:
                report_wandb(y_hat, y, loss, opt, split)

            if split == 'valid' and batch_count % opt.validation_interval == 0 \
                    and opt.wandb:
                report_wandb(y_hat, y, loss, opt, split)

            batch_count += 1

        if opt.save_feats:
            save_feats(
                opt.model_name, split, all_targets, all_locs, all_x_f, 
                chromosome)
            all_targets = torch.Tensor().cpu()
            all_x_f = torch.Tensor().cpu()
            all_locs = []

        if split in ["valid", "test"] and not opt.save_feats:
            scores[chromosome] = compute_scores(
                all_preds.numpy(), 
                all_targets.numpy(),
                total_loss,
                opt.wandb,
                epoch,
                split,
                chromosome
            )

            all_preds = torch.Tensor().cpu()
            all_targets = torch.Tensor().cpu()
            all_x_f = torch.Tensor().cpu()
            all_locs = []
            total_loss = 0

    # combine the individual chromosome scores
    if split in ["valid", "test"] and not opt.save_feats:
        combined_scores = compute_average_scores(
            scores, opt.wandb, split
        )
    else:
        combined_scores = {}

    return combined_scores
