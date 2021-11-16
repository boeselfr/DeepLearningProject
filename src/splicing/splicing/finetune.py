import os.path as path
import pickle

import torch
import torch.nn.functional as F
from tqdm import tqdm

from splicing.utils.graph_utils import process_graph, split2desc


# TODO
# data form: dict of chromosomes, each a dict of locations with the
#            corresponding (5000, 32) nucleotide features and (5000, 1) targets
def finetune(graph_model, data, criterion, optimizer, epoch, opt, split):
    if split == 'train':
        graph_model.train()
    else:
        graph_model.eval()

    all_preds = torch.Tensor().cpu()
    all_targets = torch.Tensor().cpu()

    total_loss = 0

    if opt.adj_type in ['hic', 'both']:
        graph_file = path.join(
            opt.graph_data_root,
            split + '_graphs_' + opt.hicsize + '_' + opt.hicnorm + 'norm.pkl')
        print(graph_file)
        split_adj_dict = pickle.load(open(graph_file, "rb"))
    else:
        split_adj_dict = None

    for chromosome in tqdm(data, mininterval=0.5, leave=False,
                           desc='(' + split2desc[split] + ')'):
        chromosome_data = data[chromosome]

        x = chromosome_data['x'][0]  # data of shape (1, 5000, 32)...
        y = chromosome_data['y'][0]
        x.requires_grad = True

        split_adj = process_graph(
            opt.adj_type, split_adj_dict, len(x), chromosome).cuda()

        if split == 'train':
            optimizer.zero_grad()

        # TODO
        _, pred, _, z = graph_model(x, split_adj, None)

        loss = criterion(pred, y)

        if split == 'train':
            loss.backward()
            optimizer.step()

        total_loss += loss.sum().item()
        all_preds = torch.cat((all_preds, F.sigmoid(pred).cpu().data), 0)
        all_targets = torch.cat((all_targets, y.cpu().data), 0)

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
