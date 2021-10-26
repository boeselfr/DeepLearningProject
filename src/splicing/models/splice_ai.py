import numpy as np

import torch
from torch import nn


class SpliceAI(nn.Module):
    def __init__(self, L, W, AR, device='cuda'):
        super(SpliceAI, self).__init__()
        self.L, self.W, self.AR = L, W, AR
        assert len(self.W) == len(self.AR)

        self.CL = 2 * np.sum(self.AR * (self.W - 1))

        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=L,
            kernel_size=1).to(device)
        self.skip = nn.Conv1d(
            in_channels=L,
            out_channels=L,
            kernel_size=1).to(device)

        self.residual_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        self.n_blocks = len(W)
        for i in range(self.n_blocks):
            self.residual_blocks.append(
                nn.Sequential(
                    nn.BatchNorm1d(L),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=L,
                        out_channels=L,
                        kernel_size=W[i],
                        dilation=AR[i],
                        padding='same'),
                    nn.BatchNorm1d(L),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=L,
                        out_channels=L,
                        kernel_size=W[i],
                        dilation=AR[i],
                        padding='same')).to(device)
            )

            if ((i + 1) % 4 == 0) or ((i + 1) == len(W)):
                self.skip_connections.append(
                    nn.Conv1d(
                        in_channels=L,
                        out_channels=L,
                        kernel_size=1).to(device)
                )

        self.out = nn.Conv1d(
            in_channels=L,
            out_channels=3,
            kernel_size=1).to(device)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, input):

        conv = self.conv(input)
        skip = self.skip(conv)

        for i in range(self.n_blocks):
            tmp_conv = self.residual_blocks[i](conv)
            conv = torch.add(conv, tmp_conv)
            if ((i + 1) % 4 == 0) or ((i + 1) == self.n_blocks):
                dense = self.skip_connections[i // 4](conv)
                skip = torch.add(skip, dense)

        skip = skip[:, :, self.CL // 2: -self.CL // 2]

        conv = self.out(skip)
        out = self.out_act(conv)
        return out


def categorical_crossentropy_2d(y_true, y_pred, weights=(1, 1, 1)):
    return -torch.mean(
        weights[0] * y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + 1e-10)
        + weights[1] * y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + 1e-10)
        + weights[2] * y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + 1e-10))
