import os.path as path
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from splicing.utils.graph_utils import process_graph, split2desc, \
    build_node_representations
from splicing.data_models.splice_dataset import ChromosomeDataset


# TODO
# data form: dict of chromosomes, each a dict of locations with the
#            corresponding (5000, 32) nucleotide features and (5000, 1) targets
def finetune(graph_model, full_model, data, criterion, optimizer,
             epoch, opt, split):
    if split == 'train':
        graph_model.train()
    else:
        graph_model.eval()

    all_preds = torch.Tensor().cpu()
    all_targets = torch.Tensor().cpu()

    total_loss = 0

    # bin dict has information for all chromosomes
    # that's why we can load it here before:
    # we need it also for the constant case, thus always has to be loaded
    bin_dict_file = path.join(
        opt.graph_data_root,
        f'test_vail_train_bin_dict_{opt.hicsize}_{opt.hicnorm}norm.pkl')
    bin_dict = pickle.load(open(bin_dict_file, "rb"))

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

        xs = chromosome_data['x']
        ys = chromosome_data['y']

        x = build_node_representations(xs, mode=opt.node_representation)

        x.requires_grad = True

        edge_index = process_graph(
            opt.adj_type, split_adj_dict, len(x),
            chromosome, bin_dict, opt.window_size).cuda()

        graph = Data(x=x, edge_index=edge_index)

        # TODO... x is not of the correct form...
        node_representation = graph_model(graph)

        chromosome_dataset = ChromosomeDataset(xs, ys)
        dataloader = DataLoader(
            chromosome_dataset, batch_size=opt.graph_batch_size)

        for _x, _y in dataloader:

            if split == 'train':
                optimizer.zero_grad()

            _y_hat = full_model(_x, node_representation)

            loss = criterion(_y_hat, _y)

            if split == 'train':
                loss.backward()
                optimizer.step()

            total_loss += loss.sum().item()
            all_preds = torch.cat((all_preds, _y_hat.cpu().data), 0)
            all_targets = torch.cat((all_targets, _y.cpu().data), 0)

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
