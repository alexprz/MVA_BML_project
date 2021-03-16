"""Reproduce figure 5 of the paper."""
import numpy as np
import argparse
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fig5B')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--nlayers', type=int, default=10, help='Number of layers')
    parser.add_argument('--nplayers', type=int, default=10, help='Number of neurons per layer')
    parser.add_argument('--rs', type=int, default=0, help='Random state')
    parser.add_argument('--sigw', type=float, default=np.sqrt(2), help='Sigma weights')
    parser.add_argument('--sigb', type=float, default=0, help='Sigma bias')
    parser.add_argument('--name', type=str, default='Exp', help='experiment name')
    parser.add_argument('--ns', type=float, default='0', help='negative slope of Leaky ReLU')
    args = parser.parse_args()

    dataset = MNIST('MNIST/', train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.bs)
    val_loader = DataLoader(mnist_val, batch_size=args.bs)
    
    if args.ns==0:
        activation=nn.ReLU()
    else:
        activation=nn.LeakyReLU(negative_slope=args.ns)
    
    model = LinearModel(n_in=28*28, n_out=10, n_layers=args.nlayers,
                        n_per_layers=args.nplayers, activation=activation)
    model.init_weights(sig_w=args.sigw, sig_b=args.sigb)

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/',
                                          filename='{epoch:02d}-{val_loss:.3f}',
                                          monitor='val_loss')

    logger = TensorBoardLogger("tb_logs", name="{}_{}_{}_{}_{}_{}".format(args.name,args.nlayers,args.nplayers,args.ns,args.sigw,args.sigb))

    trainer = pl.Trainer(max_epochs=args.epochs,
                         checkpoint_callback=checkpoint_callback,
                         logger=logger)
    
    trainer.fit(model, train_loader, val_loader)
