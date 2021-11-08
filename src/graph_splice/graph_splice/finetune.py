import os.path as path
import pickle

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import util_methods


# TODO
def finetune(chrome_model, dataloaders, criterion, optimizer,
             epoch, opt, split):
    if split == 'train':
        chrome_model.train()
    else:
        chrome_model.eval()

    all_preds = torch.Tensor().cpu()
    all_targets = torch.Tensor().cpu()

    total_loss = 0

    if opt.adj_type in ['hic', 'both']:
        graph_file = path.join(
            opt.graph_root,
            split + '_graphs_' + opt.hicsize + '_' + opt.hicnorm + 'norm.pkl')
        print(graph_file)
        split_adj_dict = pickle.load(open(graph_file, "rb"))
    else:
        split_adj_dict = None

    chrom_count = len(dataloaders)

    for dataloader in tqdm(dataloaders, mininterval=0.5,
                      desc='(' + split + ')', leave=False):
        x_f = dataloader['forward'].cuda()
        targets = dataloader['target'].cuda()
        x_f.requires_grad = True

        split_adj = util_methods.process_graph(
            opt.adj_type, split_adj_dict, x_f.size(0), chrom).cuda()

        if split == 'train':
            optimizer.zero_grad()

        _, pred, _, z = chrome_model(x_f, split_adj, None)

        loss = criterion(pred, targets)

        if split == 'train':
            loss.backward()
            optimizer.step()

        total_loss += loss.sum().item()
        all_preds = torch.cat((all_preds, F.sigmoid(pred).cpu().data), 0)
        all_targets = torch.cat((all_targets, targets.cpu().data), 0)

        # A Saliency or TF-TF Relationships
        # Compare to CNN Preds
        # cnn_pred_f = WindowModel.module.model.relu(x_f)
        # cnn_pred_f = WindowModel.module.model.batch_norm(cnn_pred_f.cuda())
        # cnn_pred_f = WindowModel.module.model.classifier(cnn_pred_f)
        # cnn_pred_r = WindowModel.module.model.relu(x_r)
        # cnn_pred_r = WindowModel.module.model.batch_norm(cnn_pred_r.cuda())
        # cnn_pred_r = WindowModel.module.model.classifier(cnn_pred_r)
        # cnn_pred = (cnn_pred_f+cnn_pred_r)/2
        # if chrom == 'chr8' and opt.A_saliency: stop()

    return all_preds, all_targets, total_loss
