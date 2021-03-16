"""Compute EOC (taken from the GitHub of the paper).
https://github.com/soufiane001/impact_of_init_and_activation/blob/master/eoc_curves.py
"""
import argparse
import numpy as np

# Defining activation function and their derivatives
def relu(x):
    return x*(x>0)

def relu_dash(x):
    return (x > 0).astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ssoftplus(x):
    return np.log(1 + e^x) - np.log(2)

def tanh(x):
    return np.tanh(x)

def tanh_dash(x):
    return 4 /(np.exp(x) + np.exp(-x))**2

def tanh_2dash(x):
    return  8*(-np.exp(x) + np.exp(-x))/(np.exp(-x) + np.exp(x))**3

def swish(x):
    return x * sigmoid(x)

def swish_dash(x):
    return (sigmoid(x) + x * sigmoid_dash(x))

def elu(x):
    return x *(x>0) + (np.exp(x)-1) * (x<=0)

def elu_dash(x):
    return (x>0) + np.exp(x) * (x<=0)


# We define a function get_eoc that returns triplets (\sigma_b, \sigma_w, q) on the EOC
def get_eoc(act, act_dash, sigma_bs, N):
    # simulate gaussian variables for mean calculations
    z1 = np.random.randn(N)

    eoc = []
    if not isinstance(sigma_bs, list):
        sigma_bs = [sigma_bs]

    for sigma in sigma_bs:
        q = 0
        for i in range(200):
            q = sigma**2 + np.mean(act(np.sqrt(q)*z1)**2)/np.mean(act_dash(np.sqrt(q)*z1)**2)
        eoc.append([sigma, 1/np.sqrt(np.mean(act_dash(np.sqrt(q)*z1)**2)), q])

    return np.squeeze(np.array(eoc))


def get_eoc_by_name(act_name, sigma_bs, N):
    if act_name == 'relu':
        return get_eoc(relu, relu_dash, sigma_bs, N)
    elif act_name == 'elu':
        return get_eoc(elu, elu_dash, sigma_bs, N)
    else:
        raise ValueError(f'Unknown activation function {act_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fig1')
    parser.add_argument('--sigb', type=float, default=0, help='Sigma bias')
    parser.add_argument('--act', type=str, default='elu', help='Which activation to use')
    parser.add_argument('--rs', type=int, default=0, help='Random state')
    parser.add_argument('--n', type=int, default=500000, help='Number of samples to drawn for the computation')
    args = parser.parse_args()

    np.random.seed(args.rs)

    eoc = get_eoc_by_name(args.act, args.sigb, args.n)

    print(f'EOC for sigma_b={args.sigb} and act {args.act} is:\n\tsigma_w={eoc[1]:.3f}')
