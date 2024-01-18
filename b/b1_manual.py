import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.optimize as opt
import scipy.spatial
import torch


def g(x: torch.Tensor | float) -> torch.Tensor:
    return -torch.sin(6 * math.pi * x)**2 + 6 * x**2 - 5 * x**4 + 3 / 2


def periodic_kernel(x, x_, v, p, l):
    dists = torch.cdist(x.view(-1, 1), x_.view(-1, 1))
    s = torch.sin(math.pi * dists / p) / l
    return v * torch.exp(-2 * s**2)


def linear_kernel(x, x_, a, b):
    return b + a * torch.outer(x, x_)


l = 30
n = 30
xs = torch.arange(l) / (l - 1)
ys = g(xs) + torch.normal(0.0, np.sqrt(0.01).item(), (l,))

train_indices = np.random.choice(l, n, replace=False)
train_mask = torch.zeros(l, dtype=bool)
train_mask[train_indices] = True

xs_train = xs[train_mask]
ys_train = ys[train_mask]

xs_test = xs[~train_mask]
ys_test = ys[~train_mask]


def log_likelihood(x, y: torch.Tensor, kernel, noise):
    n = y.numel()

    prod = lambda p: p.sign * p.logabsdet
    return -(
        y.reshape(1, -1) @ torch.linalg.inv(noise * torch.eye(n) + kernel(x, x)) @ y.reshape(-1, 1)
        + prod(torch.linalg.slogdet(noise * torch.eye(n) + kernel(x, x)))
        + n * np.log(2 * math.pi)
    ) / 2


def posterior(x_predict, x, y, kernel, noise):
    n = y.numel()
    mu = kernel(x, x_predict).T @ torch.linalg.inv(kernel(x, x) + noise * torch.eye(n)) @ y
    Sigma = kernel(x_predict, x_predict) - kernel(x, x_predict).T @ torch.linalg.inv(kernel(x, x) + noise * torch.eye(n)) @ kernel(x, x_predict)

    return mu, Sigma


def main():
    v = torch.tensor(1., requires_grad=True)
    p = torch.tensor(1/6., requires_grad=False)
    l = torch.tensor(1., requires_grad=True)
    b = torch.tensor(0., requires_grad=True)
    a = torch.tensor(0., requires_grad=True)
    n = torch.tensor(0.2, requires_grad=True)

    k1 = lambda x, x_: linear_kernel(x, x_, a, b)
    k2 = lambda x, x_: periodic_kernel(x, x_, v, p, l)

    kernel = lambda x, x_: k1(x, x_) + k2(x, x_)
    optimiser = torch.optim.Adam((v, l, b, a), lr=0.005)

    for i in range(15_000):
        optimiser.zero_grad()
        loss = -log_likelihood(xs_train, ys_train, kernel, n)
        loss.backward()
        optimiser.step()

        if (i + 1) % 1000 == 0:
            print(i, loss.item())

    print('v', v)
    print('p', p)
    print('l', l)
    print('b', b)
    print('a', a)
    print('n', n)

    # plt.plot(xs_train, ys_train, "kx")
    # xs = torch.linspace(0, 1, 100)
    # with torch.no_grad():
    #     mean, cov = posterior(xs, xs_train, ys_train, kernel, n)
    # sd = cov.diag().sqrt()

    # plt.plot(xs, mean, 'r', lw=2)
    # plt.fill_between(
    #     xs,
    #     mean - 1.96 * sd,
    #     mean + 1.96 * sd,
    #     color="C0",
    #     alpha=0.3
    # )
    # plt.plot(xs, g(xs), color="green")

    # plt.savefig('pred.png', dpi=600)


    # k1 = lambda x, x_: linear_kernel(x, x_, b, a)
    # k2 = lambda x, x_, p: periodic_kernel(x, x_, torch.exp(p), torch.exp(l))

    # variances = torch.logspace(-3, 0, 100)
    # periods = torch.linspace(0.1, 5, 100)

    # lls = torch.empty(100, 100)
    # for i, v in enumerate(variances):
    #     for j, p in enumerate(periods):
    #         kernel = lambda x, x_: k1(x, x_) + k2(x, x_, p)
    #         loss = log_likelihood(xs_train, ys_train, kernel, v).item()
    #         lls[i, j] = loss

    # plt.figure()
    # plt.contourf(*np.meshgrid(variances, periods), lls)
    # plt.colorbar()
    # plt.xscale('log')
    # plt.yscale('log')
    # # plt.contourf(lls)
    # plt.savefig('contour.png', dpi=300)





if __name__ == '__main__':
    main()
