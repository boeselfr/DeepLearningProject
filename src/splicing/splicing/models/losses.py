import torch
from torch import nn


class CategoricalCrossEntropy2d(nn.Module):

    def __init__(self, weights):
        super(CategoricalCrossEntropy2d, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        return -torch.mean(
            self.weights[0] * targets[:, 0, :] * torch.log(
                predictions[:, 0, :] + 1e-10)
            + self.weights[1] * targets[:, 1, :] * torch.log(
                predictions[:, 1, :] + 1e-10)
            + self.weights[2] * targets[:, 2, :] * torch.log(
                predictions[:, 2, :] + 1e-10))
