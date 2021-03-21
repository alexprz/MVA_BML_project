"""Compute EOC (taken from the GitHub of the article).

https://github.com/soufiane001/impact_of_init_and_activation/blob/master/eoc_curves.py
"""
import argparse
import numpy as np


# Defining some activation functions and their derivatives
def relu(x):
    """Implement ReLU activation function."""
    return x*(x > 0)


def relu_dash(x):
    """Implement ReLU derivative."""
    return (x > 0).astype(int)


def sigmoid(x):
    """Implement signmoid activation function."""
    return 1 / (1 + np.exp(-x))


def ssoftplus(x):
    """Implement softplus activation function."""
    return np.log(1 + np.exp(x)) - np.log(2)


def tanh(x):
    """Implement tanh activation function."""
    return np.tanh(x)


def tanh_dash(x):
    """Implement tanh derivative."""
    return 4/(np.exp(x) + np.exp(-x))**2


def tanh_2dash(x):
    """Implement tanh derivative."""
    return 8*(-np.exp(x) + np.exp(-x))/(np.exp(-x) + np.exp(x))**3


def elu(x):
    """Implement ELU activation function."""
    return x*(x > 0) + (np.exp(x)-1) * (x <= 0)


def elu_dash(x):
    """Implement ELU derivative."""
    return (x > 0) + np.exp(x) * (x <= 0)


def get_eoc(act, act_dash, sigma_bs, N):
    """Compute sigma_w on the EOC given sigma_b.

    Parameters:
    -----------
        act : callable
            Activation function
        act : callable
            Activation derivative
        sigma_bs : float or float array of shape (n,)
            Standard deviation of the bias
        N : int
            Number of samples to draw to do the computation

    Returns:
    --------
        np.array of shape (n, 3)
            Contains (sigma_b, sigma_w, q) for each of the input sigma_b

    """
    # Simulate gaussian variables for mean calculations
    z1 = np.random.randn(N)

    eoc = []
    if not isinstance(sigma_bs, list):
        sigma_bs = [sigma_bs]

    for sigma in sigma_bs:
        q = 0
        for _ in range(200):
            q = sigma**2 + np.mean(act(np.sqrt(q)*z1)**2)/np.mean(act_dash(np.sqrt(q)*z1)**2)
        eoc.append([sigma, 1/np.sqrt(np.mean(act_dash(np.sqrt(q)*z1)**2)), q])

    return np.squeeze(np.array(eoc))


def get_eoc_by_name(act_name, sigma_bs, N):
    """Compute same values as get_eoc but by specifying activation name."""
    if act_name == 'relu':
        return get_eoc(relu, relu_dash, sigma_bs, N)

    if act_name == 'elu':
        return get_eoc(elu, elu_dash, sigma_bs, N)

    raise ValueError(f'Unknown activation function {act_name}')


if __name__ == '__main__':
    # An example of computation
    parser = argparse.ArgumentParser(description='fig1')
    parser.add_argument('--sigb', type=float, default=0, help='Sigma bias')
    parser.add_argument('--act', type=str, default='elu', help='Which activation to use')
    parser.add_argument('--rs', type=int, default=0, help='Random state')
    parser.add_argument('--n', type=int, default=500000, help='Number of samples to drawn for the computation')
    args = parser.parse_args()

    np.random.seed(args.rs)

    eoc = get_eoc_by_name(args.act, args.sigb, args.n)

    print(f'EOC for sigma_b={args.sigb} and act="{args.act}"" is:')
    print(f'\tsigma_w={eoc[1]:.3f}')
