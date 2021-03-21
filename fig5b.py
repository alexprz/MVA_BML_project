"""Reproduce figure 5 of the article with custom activation function."""
import numpy as np
import argparse
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from pytorch_lightning.loggers import TensorBoardLogger
from model import LinearModel


if __name__ == '__main__':
    # Read command line arguments given to the script
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
    parser.add_argument('--act', type=str, default='relu', help='Which activation to use')
    args = parser.parse_args()

    # Download MNIST dataset from PyTorch
    dataset = MNIST('MNIST/', train=True, download=True, transform=transforms.ToTensor())

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

    # Customly init the model weights (chaotic, ordered, EOC)
    model.init_weights(sig_w=args.sigw, sig_b=args.sigb, seed=args.rs)

    # Create checkpoint to automatically save trained models
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/',
                                          filename='{epoch:02d}-{val_loss:.3f}',
                                          monitor='val_loss')

    # Create a Tensorboard logger to monitor metrics (losses on train and val)
    logger = TensorBoardLogger('tb_logs', name=f'{args.name}_{args.nlayers}_{args.nplayers}_{args.ns}_{args.sigw}_{args.sigb}')

    # Create a PyTorch Lightning trainer to train the model
    trainer = pl.Trainer(max_epochs=args.epochs,
                         checkpoint_callback=checkpoint_callback,
                         logger=logger)

    # Train the model using PyTorch Lightning
    trainer.fit(model, train_loader, val_loader)

    # The curves are available in tensorboard by running:
    # tensorboard --logdir tb_logs
