#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Linear Deep Neural Network """

import torch
from torch import nn
import pytorch_lightning as pl


class LinearModel(pl.LightningModule):

    def __init__(self, n_in, n_out, n_layers, n_per_layers, activation):
        super().__init__()
        assert n_layers >= 2

        self.layers = nn.Sequential(
            nn.Linear(n_in, n_per_layers),
            activation,
            *[nn.Sequential(nn.Linear(n_per_layers, n_per_layers), activation) for i in range(n_layers-2)],
            nn.Linear(n_per_layers, n_out),
            #nn.Softmax(),
        )
        self.activation = activation

    def init_weights(self, sig_w=1, sig_b=1):
        for name, param in self.named_parameters():
            if name[-4:] == 'bias':
                if sig_b ==0:
                    torch.nn.init.constant_(param,0)
                else:
                    torch.nn.init.normal_(param, 0., sig_b)
            else:
                if sig_w ==0:
                    torch.nn.init.constant_(param,0)
                else:
                    torch.nn.init.normal_(param, 0., sig_w/(param.shape[1]**(1/2)))

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-4)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss(reduction='mean')(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        pred = y_hat.data.max(1, keepdim=True)[1]
        acc = pred.eq(y.data.view_as(pred)).sum()/y.size(0)
        loss = 1 - acc
        self.log('val_loss', loss)
        return loss