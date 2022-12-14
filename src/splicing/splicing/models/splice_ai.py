import logging

import numpy as np

import torch
from torch import nn


class SpliceAI(nn.Module):
    """ the base CNN local model based on SpliceAI"""
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

        # residual blocks
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
        self.out_act = nn.Softmax(dim=1)

    def forward(self, input, save_feats):

        conv = self.conv(input)
        skip = self.skip(conv)

        for i in range(self.n_blocks):
            tmp_conv = self.residual_blocks[i](conv)
            conv = torch.add(conv, tmp_conv)
            if ((i + 1) % 4 == 0) or ((i + 1) == self.n_blocks):
                dense = self.skip_connections[i // 4](conv)
                skip = torch.add(skip, dense)

        # discard the padded predictions outside of the window
        x = skip[:, :, self.context_length // 2: -self.context_length // 2]

        pred = self.out(x)
        pred = self.out_act(pred)
        return (pred, x, None) if save_feats else (pred, None, None)


# ensemble of models for evaluation of trained models
class SpliceAIEnsemble(nn.Module):
    def __init__(self, models, window_size):
        super(SpliceAIEnsemble, self).__init__()
        self.models = models
        self.window_size = window_size

    def forward(self, input):

        predictions = torch.zeros(
            size=(len(self.models), input.shape[0], 3, self.window_size)
        )

        for ii, model in enumerate(self.models):
            predictions[ii, :, :, :] = model(input)[0].cpu()

        combined_predictions = torch.mean(predictions, dim=0)

        return combined_predictions, None, None
