import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


from utils import clip_datapoints
from itertools import product


class SpliceDataset(Dataset):

    def __init__(self, X, Y, CL, N_GPUS, device='cuda'):
        X, Y = clip_datapoints(X, Y, CL, N_GPUS)
        self.X = np.array([np.transpose(x) for x in X])
        self.Y = np.array([np.transpose(y) for y in Y[0]])

        # self.Y = np.zeros(shape=self.Y.shape)
        # for i, j in product(range(self.Y.shape[0]), range(self.Y.shape[2])):
        #     c = np.random.choice(3)
        #     self.Y[i, c, j] = 1

        self.device = device

    def get_true(self, index, expr):
        # return self.Y[expr, :, index].flatten()
        return self.Y[expr, index, :].flatten()

    def get_expr(self):
        return self.Y.sum(axis=(1, 2)) >= 1

    def __getitem__(self, index) -> T_co:
        x = torch.tensor(self.X[index], dtype=torch.float32).to(self.device)
        y = torch.tensor(self.Y[index], dtype=torch.float32).to(self.device)
        return x, y

    def __len__(self):
        return len(self.X)
