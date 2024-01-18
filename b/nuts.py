import math
import sys
import pickle

import matplotlib.pyplot as plt
import torch

import pyro
import pyro.infer
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




def log_likelihood(x, y: torch.Tensor, kernel, noise):
    n = y.numel()

    prod = lambda p: p.sign * p.logabsdet
    return -(
        y.reshape(1, -1) @ torch.linalg.inv(noise * torch.eye(n) + kernel(x, x)) @ y.reshape(-1, 1)
        + prod(torch.linalg.slogdet(noise * torch.eye(n) + kernel(x, x)))
        + n * np.log(2 * math.pi)
    ) / 2


def log_likelihood_prior(x, y, kernel, period, noise):
    ll = log_likelihood(x, y, kernel, noise)
    return ll + pdist.LogNormal(0.1, 10).log_prob(period) + pdist.LogNormal(-3.0, 1.5).log_prob(noise)


def posterior_predictive(x_predict, x, y, kernel, noise):
    n = y.numel()
    mu = kernel(x, x_predict).T @ torch.linalg.inv(kernel(x, x) + noise * torch.eye(n)) @ y.reshape(-1, 1)
    Sigma = kernel(x_predict, x_predict) - kernel(x, x_predict).T @ torch.linalg.inv(kernel(x, x) + noise * torch.eye(n)) @ kernel(x, x_predict)

    return mu, Sigma


a = torch.tensor(1.3, requires_grad=False)
b = torch.tensor(0.5, requires_grad=False)
v = torch.tensor(5., requires_grad=False)
l = torch.tensor(0.2, requires_grad=False)


def model():
    period = 10**pyro.sample("period", pdist.Uniform(0, 2))
    noise = 10**pyro.sample("noise", pdist.Uniform(-2, 1))
    # period = pyro.sample("period", pdist.LogNormal(0.1, 10))
    # noise = pyro.sample("noise", pdist.LogNormal(-3, 1.5))

    k1 = lambda x, x_: linear_kernel(x, x_, a, b)
    k2 = lambda x, x_: periodic_kernel(x, x_, v, period, l)

    kernel = lambda x, x_: k1(x, x_) + k2(x, x_)

    pyro.factor(
        "log_prob",
        log_likelihood_prior(xs_train, ys_train, kernel, period, noise)
    )


#Grid search

def compute_estimate(xs_train, ys_train, xs_test, ys_test):
    max_ll, optimal_parameters = -np.inf, None
    for period in torch.logspace(0, 2, 100):
        for noise in torch.logspace(-2, 1, 100):
            kernel = lambda x, x_: linear_kernel(x, x_, a, b) + periodic_kernel(x, x_, v, period, l)
            ll = log_likelihood_prior(xs_train, ys_train, kernel, period, noise)

            if ll > max_ll:
                max_ll = ll
                optimal_parameters = (period, noise)


    print(f'{ll=}')
    period, noise = optimal_parameters
    print(f'{period=}, {noise=}')

    kernel = lambda x, x_: linear_kernel(x, x_, a, b) + periodic_kernel(x, x_, v, period, l)
    mu, sigma = posterior_predictive(xs_test, xs_train, ys_train, kernel, noise)
    print(f'{mu.shape=}')
    print(f'{sigma.shape=}')

    grid_log_prob = torch.distributions.Normal(mu.flatten(), torch.sqrt(torch.diag(sigma))).log_prob(ys_test).sum()
    print(f'{grid_log_prob=}')


    ###NUTS sampling
    nuts_kernel = pyro.infer.NUTS(model, jit_compile=False)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=500, num_chains=5, warmup_steps=2000, mp_context='fork')
    mcmc.run()

    data = arviz.from_pyro(mcmc)
    print(arviz.summary(data))
    arviz.plot_trace(data)
    plt.savefig('arviz.png', dpi=600)

    samples = mcmc.get_samples()

    plt.figure()
    plt.scatter(10**samples['period'], 10**samples['noise'])
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('scatter.png', dpi=600)


    log_probs = []
    for period, noise in zip(samples['period'], samples['noise']):
        kernel = lambda x, x_: linear_kernel(x, x_, a, b) + periodic_kernel(x, x_, v, 10**period, l)
        mu, sigma = posterior_predictive(xs_test, xs_train, ys_train, kernel, 10**noise)

        log_prob = torch.distributions.Normal(mu, torch.sqrt(torch.diag(sigma))).log_prob(ys_test).sum()
        log_probs.append(log_prob)

    nuts_log_prob = sum(log_probs) / len(log_probs)
    print(f'{nuts_log_prob=}')
    return (grid_log_prob, nuts_log_prob)

glps = []
nlps = []
for _ in range(1):
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

    grid_log_prob, nuts_log_prob = compute_estimate(xs_train, ys_train, xs_test, ys_test)

    pyro.clear_param_store()
    
    glps.append(grid_log_prob)
    nlps.append(nuts_log_prob)

print(np.mean(np.array(glps)))
print(np.mean(np.array(nlps)))



