import numpy as np

import torch
from torch import nn


class SpliceAI(nn.Module):
    def __init__(self, n_channels, kernel_size, dilation_rate, device='cuda'):
        super(SpliceAI, self).__init__()

        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        assert len(self.kernel_size) == len(self.dilation_rate)

        self.context_length = 2 * np.sum(
            self.dilation_rate * (self.kernel_size - 1))

        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=self.n_channels,
            kernel_size=1).to(device)
        self.skip = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=self.n_channels,
            kernel_size=1).to(device)

        self.residual_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        self.n_blocks = len(self.kernel_size)
        for i in range(self.n_blocks):
            self.residual_blocks.append(
                nn.Sequential(
                    nn.BatchNorm1d(self.n_channels),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=self.n_channels,
                        out_channels=self.n_channels,
                        kernel_size=self.kernel_size[i],
                        dilation=self.dilation_rate[i],
                        padding='same'),
                    nn.BatchNorm1d(self.n_channels),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=self.n_channels,
                        out_channels=self.n_channels,
                        kernel_size=self.kernel_size[i],
                        dilation=self.dilation_rate[i],
                        padding='same')).to(device)
            )

            if ((i + 1) % 4 == 0) or ((i + 1) == len(kernel_size)):
                self.skip_connections.append(
                    nn.Conv1d(
                        in_channels=self.n_channels,
                        out_channels=self.n_channels,
                        kernel_size=1).to(device)
                )

        self.out = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=3,
            kernel_size=1).to(device)
        # self.out_act = nn.Softmax(dim=1)

    def forward(self, input):

        conv = self.conv(input)
        skip = self.skip(conv)

        for i in range(self.n_blocks):
            tmp_conv = self.residual_blocks[i](conv)
            conv = torch.add(conv, tmp_conv)
            if ((i + 1) % 4 == 0) or ((i + 1) == self.n_blocks):
                dense = self.skip_connections[i // 4](conv)
                skip = torch.add(skip, dense)

        skip = skip[:, :, self.context_length // 2: -self.context_length // 2]

        conv = self.out(skip)
        # out = self.out_act(conv)
        out = conv
        return out


def categorical_crossentropy_2d(y_true, y_pred, weights=(1, 1, 1)):
    return -torch.mean(
        weights[0] * y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + 1e-10)
        + weights[1] * y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + 1e-10)
        + weights[2] * y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + 1e-10))
