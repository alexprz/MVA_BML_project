"""Reproduce figure 1 of the paper."""

import numpy as np
import argparse
import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from matplotlib import cm

from model import LinearModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fig5B')
    parser.add_argument('--nlayers', type=int, default=10, help='Number of layers')
    parser.add_argument('--nplayers', type=int, default=10, help='Number of neurons per layer')
    parser.add_argument('--rs', type=int, default=0, help='Random state')
    parser.add_argument('--sigw', type=float, default=np.sqrt(2), help='Sigma weights')
    parser.add_argument('--sigb', type=float, default=0, help='Sigma bias')
    parser.add_argument('--name', type=str, default='pred', help='experiment name')
    parser.add_argument('--ns', type=float, default=0, help='negative slope of Leaky ReLU')
    args = parser.parse_args()
    
    if args.ns==0:
        activation=nn.ReLU()
    else:
        activation=nn.LeakyReLU(negative_slope=args.ns)
        
    with torch.no_grad():
    
        model = LinearModel(n_in=2, n_out=1, n_layers=args.nlayers,
                            n_per_layers=args.nplayers, activation=activation)
        model.init_weights(sig_w=args.sigw, sig_b=args.sigb)
        model = model.cuda()
        model.eval()
        
        #plot
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
        # Make data.
        X = torch.arange(-5, 5, 0.2).cuda()
        Y = torch.arange(-5, 5, 0.2).cuda()
        X, Y = torch.meshgrid(X, Y)
        grid = torch.stack((X,Y),axis=2)
        grid = grid.view(-1,2)
    
        pred = model(grid)
        pred=pred.view(X.shape)
        print(pred)

        # Plot the surface.
        surf = ax.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), pred.cpu().numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        os.makedirs('figs/', exist_ok=True)
        plt.title(args.name)
        plt.savefig('figs/{}.jpg'.format(args.name))#, bbox_inches='tight')
        plt.show()
    
