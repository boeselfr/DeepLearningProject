import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


from splicing.utils.utils import clip_datapoints


class SpliceDataset(Dataset):

    def __init__(self, X, Y, locations, chromosomes,
                 context_length, n_gpus=1, device='cuda'):
        X, Y = clip_datapoints(X, Y, context_length, n_gpus)
        self.X = np.array([np.transpose(x) for x in X])
        self.Y = np.array([np.transpose(y) for y in Y[0]])
        self.locations = np.array(locations)
        self.chromosomes = np.array(chromosomes)

        self.device = device

    def get_true(self, index, expr):
        return self.Y[expr, index, :].flatten()

    def get_expr(self):
        return self.Y.sum(axis=(1, 2)) >= 1

    def __getitem__(self, index) -> T_co:
        x = torch.tensor(self.X[index], dtype=torch.float32).to(self.device)
        y = torch.tensor(self.Y[index], dtype=torch.float32).to(self.device)
        loc = torch.tensor(
            self.locations[index], dtype=torch.float32).to(self.device)
        chr = torch.tensor(
            self.chromosomes[index], dtype=torch.float32).to(self.device)
        return x, y, loc, chr

    def __len__(self):
        return len(self.X)
