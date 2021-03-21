"""Compute and plot the loss evolution as a function of sigma_b on the EOC."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from eoc import get_eoc_by_name
from model import LinearModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fig5B')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--nlayers', type=int, default=10, help='Number of layers')
    parser.add_argument('--nplayers', type=int, default=10, help='Number of neurons per layer')
    parser.add_argument('--rs', type=int, default=0, help='Random state')
    parser.add_argument('--sigb_max', type=float, default=0.1, help='maximum Sigma bias')
    parser.add_argument('--nsigb', type=int, default=21, help='number of points on the curve (number of different sigma_b considere')
    parser.add_argument('--name', type=str, default='Exp', help='experiment name')
    parser.add_argument('--ns', type=float, default='0', help='negative slope of Leaky ReLU')
    parser.add_argument('--act', type=str, default='elu', help='Which activation to use')
    parser.add_argument('--n', type=int, default=500000, help='Number of samples to drawn for the computation of eoc')
    args = parser.parse_args()

    np.random.seed(args.rs)
    torch.manual_seed(args.rs)

    # Select available device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # Download MNIST dataset from PyTorch
    dataset = MNIST('MNIST/', train=True, download=True,
                    transform=transforms.ToTensor())

    # Create a train set and a validation set
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # Create associated data loaders
    train_loader = DataLoader(mnist_train, batch_size=args.bs)
    val_loader = DataLoader(mnist_val, batch_size=args.bs)

    # Select an activation function given its name
    if args.act == 'relu':
        activation = nn.ReLU()

    elif args.act == 'lrelu':
        activation = nn.LeakyReLU(negative_slope=args.ns)

    elif args.act == 'elu':
        activation = nn.ELU()

    else:
        raise ValueError(f'Unknown activation {args.act}')

    # Create fully connected linear model with dim 784 input and dim 10 output
    model = LinearModel(n_in=28*28, n_out=10, n_layers=args.nlayers,
                        n_per_layers=args.nplayers, activation=activation)

    # List of sigma on the abscisse
    sigma_list = np.linspace(0, args.sigb_max, args.nsigb)
    loss_list = []

    for sigb in sigma_list:
        # Get sigma_w on the EOC and then train for 10 epochs
        sigw = get_eoc_by_name(args.act, sigb, args.n)[1]

        # Customly init the model weights (chaotic, ordered, EOC)
        model.init_weights(sig_w=sigw, sig_b=sigb)

        # Create a Tensorboard logger to monitor metrics (losses on train and val)
        logger = TensorBoardLogger("tb_logs/{}/".format(args.name), name=f'{args.act}_{args.nlayers}_{args.nplayers}_{sigb}')

        # Create checkpoint to automatically save trained models
        checkpoint_callback = ModelCheckpoint(dirpath=f'checkpoints/{args.name}/',
                                              filename='{sigb:.3f}-{epoch:02d}-{val_loss:.3f}',
                                              monitor='val_loss')

        # Create a PyTorch Lightning trainer to train the model
        trainer = pl.Trainer(max_epochs=args.epochs,
                             checkpoint_callback=checkpoint_callback,
                             logger=logger)

        # Train the model using PyTorch Lightning
        trainer.fit(model, train_loader, val_loader)

        # Compute the loss on the validatoin set after the training
        with torch.no_grad():
            loss = torch.zeros(1).to(device)

            for x, y in val_loader:
                loss += nn.CrossEntropyLoss(reduction='sum')(model(x.view(x.size(0), -1).float()), y)

            loss /= 5000
            print(f'Loss for sigma_b={sigb}: {loss.item():.2f}')
            loss_list.append(loss.cpu().numpy())

    # Plot and save the figure
    plt.figure(figsize=(5, 5))
    plt.plot(sigma_list, loss_list)
    plt.xlabel('sigma b')
    plt.ylabel('Validation loss after 10 epochs')
    plt.grid(which='major')
    plt.savefig(f'figs/{args.name}_{args.act}_{args.nlayers}_{args.nplayers}')
    plt.tight_layout()
    plt.show()
