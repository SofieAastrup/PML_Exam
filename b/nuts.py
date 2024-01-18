import math
import sys
import pickle

import matplotlib.pyplot as plt
import torch

import pyro
import torch
import pyro.distributions as pdist
import torch.distributions as tdist
import arviz
import numpy as np
import matplotlib.pyplot as plt


torch.multiprocessing.set_sharing_strategy("file_system")


def g(x: torch.Tensor | float) -> torch.Tensor:
    return -torch.sin(6 * math.pi * x)**2 + 6 * x**2 - 5 * x**4 + 3 / 2


def periodic_kernel(x, x_, v, p, l):
    dists = torch.cdist(x.view(-1, 1), x_.view(-1, 1))
    s = torch.sin(math.pi * dists / p) / l
    return v * torch.exp(-2 * s**2)


def linear_kernel(x, x_, a, b):
    return b + a * torch.outer(x, x_)


l = 30
n = 20
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


v = torch.tensor(5., requires_grad=True)
p = torch.tensor(16.3, requires_grad=True)
l = torch.tensor(0.2, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
a = torch.tensor(1.3, requires_grad=True)
n = torch.tensor(0.175, requires_grad=True)

k1 = lambda x, x_: linear_kernel(x, x_, a, b)
k2 = lambda x, x_: periodic_kernel(x, x_, v, p, l)

kernel = lambda x, x_: k1(x, x_) + k2(x, x_)

print('v', v)
print('p', p)
print('l', l)
print('b', b)
print('a', a)
print('n', n)


def model():
    period = pyro.sample("period", pdist.LogNormal(0.1, 10)) + 1
    noise = pyro.sample("noise", pdist.LogNormal(0.01, 10))

    k1 = lambda x, x_: linear_kernel(x, x_, a, b)
    k2 = lambda x, x_: periodic_kernel(x, x_, v, period, l)

    kernel = lambda x, x_: k1(x, x_) + k2(x, x_)

    pyro.factor(
        "log_prob",
        log_likelihood(xs_train, ys_train, kernel, noise) + pdist.LogNormal(0.1, 10).log_prob(period) + pdist.LogNormal(0.01, 10).log_prob(noise)
    )

nuts_kernel = pyro.infer.NUTS(model, jit_compile=False)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=1000, num_chains=8, warmup_steps=10_000)
mcmc.run()

data = arviz.from_pyro(mcmc)
print(arviz.summary(data))
arviz.plot_trace(data)
plt.savefig('arviz.png', dpi=600)

samples = mcmc.get_samples()

print(type(samples))
with open('nuts_data.pickle', 'wb') as f:
    pickle.dump(samples, f)
