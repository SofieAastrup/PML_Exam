import math
import os
import os.path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.datasets
import torchvision.transforms
import torchvision.utils
from torch.nn import functional as F


class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder_1 = torch.nn.Linear(784, 400)
        self.encoder_2a = torch.nn.Linear(400, 2)
        self.encoder_2b = torch.nn.Linear(400, 2)

        self.decoder_1 = torch.nn.Linear(2, 400)
        self.decoder_2 = torch.nn.Linear(400, 784)

        self.activation = torch.nn.ReLU()
        self.bce = torch.nn.BCELoss(reduction="none")

        self._image_dir = None

    def encode(self, x):
        h = self.activation(self.encoder_1(x))
        return self.encoder_2a(h), self.encoder_2b(h)

    def decode(self, x):
        h = self.activation(self.decoder_1(x))
        return torch.sigmoid(self.decoder_2(h))

    def likelihood(self, x, parameters):
        ...

    def mean(self, x):
        ...

    def images(self, x):
        return self.mean(self.decode(x))

    def sample_likelihood(self, x, n: int = 1000, importance: bool = True):
        if importance:
            sum = torch.empty(0, device=self.device)
            mu, logvar = self.encode(x.view(-1, 784))
            q = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
            p = torch.distributions.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            for i in range(math.ceil(n / 100_000)):
                k = min(n - i * 100_000, 100_000)
                samples = q.sample((k,)).squeeze()
                decoded = self.decode(samples)
                ll = self.likelihood(x.view(-1, 784), decoded).sum(dim=1)
                sum = torch.concat(
                    (
                        sum,
                        p.log_prob(samples).sum(dim=1)
                        - q.log_prob(samples).sum(dim=1)
                        + ll,
                    )
                )
            return torch.logsumexp(sum, dim=0) - torch.log(torch.tensor(n))
        else:
            # sum = 0
            sum = torch.empty(0, device=self.device)
            d = torch.distributions.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            for i in range(math.ceil(n / 100_000)):
                k = min(n - i * 100_000, 100_000)
                samples = d.sample((k, 2))
                decoded = self.decode(samples)
                q = self.likelihood(x.view(-1, 784), decoded)
                # sum += torch.sum(q.sum(dim=1))
                sum = torch.concat((sum, q.sum(dim=1)))
            # return sum / n
            return torch.logsumexp(sum, dim=0) - torch.log(torch.tensor(n))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        distribution = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))

        z = distribution.rsample()

        return self.decode(z), mu, logvar

    def loss(self, x, y, mu, logvar, prefix: str = "train"):
        ...

    def training_step(self, batch, _):
        x, _ = batch
        y, mu, logvar = self(x)

        loss = self.loss(x, y, mu, logvar)
        return loss

    @property
    def image_dir(self):
        if self._image_dir is None:
            self._image_dir = os.path.join(self.logger.log_dir, "figures")
            os.makedirs(self._image_dir, exist_ok=True)

        return self._image_dir

    def validation_step(self, batch, batch_index):
        x, l = batch
        y, mu, logvar = self(x)

        self.loss(x, y, mu, logvar, prefix="val")
        y = self.mean(y)

        z, *_ = self.encode(x.view(-1, 784))
        scatter = plt.scatter(
            z[:, 1].cpu(), -z[:, 0].cpu(), c=l.cpu(), cmap="tab10", s=2, alpha=0.7
        )
        plt.legend(*scatter.legend_elements(), loc="upper right")

        if batch_index == 0:
            n = min(x.shape[0], 8)
            comparison = torch.cat((x[:n], y.view(-1, 1, 28, 28)[:n]))
            torchvision.utils.save_image(
                comparison.cpu(),
                os.path.join(
                    self.image_dir, f"reconstruction_{self.trainer.current_epoch}.png"
                ),
                nrow=n,
            )

            k = 12
            eps = 1e-5
            us = torch.linspace(eps, 1 - eps, k, device=self.device)
            xs = torch.distributions.normal.Normal(0, 1).icdf(us)
            samples = torch.dstack(torch.meshgrid(xs, xs, indexing="ij")).reshape(-1, 2)
            images = self.images(samples)
            torchvision.utils.save_image(
                images.view(k**2, 1, 28, 28),
                os.path.join(
                    self.image_dir, f"investigation_{self.trainer.current_epoch}.png"
                ),
                nrow=k,
            )

    def on_validation_epoch_end(self):
        plt.savefig(
            os.path.join(self.image_dir, f"encoding_{self.trainer.current_epoch}.png"),
            dpi=600,
        )
        plt.close()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class MLP(pl.LightningModule):
    def __init__(self, in_dim, out_dim, latent_dim):
        super().__init__()
        latent_dim = 2
        # network
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

        # mean and variance layers
        self.fc31 = nn.Linear(out_dim, latent_dim)
        self.fc32 = nn.Linear(out_dim, latent_dim)

        self.fc4 = nn.Linear(latent_dim, out_dim)
        self.fc5 = nn.Linear(out_dim, out_dim)
        self.fc6 = nn.Linear(out_dim, in_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mean = self.fc31(h2)
        variance = F.softplus(self.fc32(h2)) + 1e-8
        return h2, mean, variance

    def reparameterize(self, mu, var):
        eps = torch.randn_like(var).to(self.device)
        z = mu + eps * (var**0.5)
        return z

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)  # remove sigmoid?


def log_bernoulli_with_logits(x, logits):
    bce = nn.BCEWithLogitsLoss(reduction="none")

    return -bce(input=logits, target=x.view(-1, 784)).sum(-1)


def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl


class LVAE(VAE):
    def __init__(self):
        super().__init__()
        in_dim = 784
        layer1_dim = 500
        layer2_dim = 250
        latent1_dim = 4
        latent2_dim = 2
        self.MLP1 = MLP(in_dim, layer1_dim, latent1_dim)
        self.MLP2 = MLP(layer1_dim, layer2_dim, latent2_dim)
        self.MLP3 = MLP(latent2_dim, layer1_dim, latent1_dim)

    def encode(self, x):
        l1, mu_up, var_up = self.MLP1.encode(x)

        (
            _,
            mu_q1,
            var_q1,
        ) = self.MLP2.encode(l1)

        z1 = self.MLP2.reparameterize(mu_q1, var_q1)

        _, mu_down, var_down = self.MLP3.encode(z1)

        var_q0 = 1 / (var_up ** (-1) + var_down ** (-1))
        mu_q0 = (var_up ** (-1) * mu_up + var_down ** (-1) * mu_down) / (
            var_up ** (-1) + var_down ** (-1)
        )

        return z1, mu_q0, var_q0, mu_q1, var_q1

    def decode(self, z1):
        _, mu_p0, var_p0 = self.MLP3.encode(z1)

        z0 = self.MLP3.reparameterize(mu_p0, var_p0)

        decoded = self.MLP1.decode(z0)

        return decoded, mu_p0, var_p0

    def training_step(self, batch, _):
        x, _ = batch
        y, mu, logvar = self(x)

        loss, _ = self.loss(x)
        self.log("loss", loss)
        return loss

    def forward(self, x):
        z1, _, _, _, _ = self.encode(x.view(-1, 784))
        return self.decode(z1)

    def negative_elbo_bound(self, x, beta):
        z_given_x, qmu0, qvar0, qmu1, qvar1 = self.encode(x.view(-1, 784))
        decoded_bernoulli_logits, pmu0, pvar0 = self.decode(z_given_x)

        rec = log_bernoulli_with_logits(x, decoded_bernoulli_logits)
        rec = -torch.mean(rec)

        pm, pv = torch.zeros(qmu1.shape, device=self.device), torch.ones(
            qvar1.shape, device=self.device
        )

        kl1 = kl_normal(qmu1, qvar1, pm, pv)
        kl2 = kl_normal(qmu0, qvar0, pmu0, pvar0)
        kl = beta * torch.mean(kl1 + kl2)

        nelbo = rec + kl

        return nelbo, rec, kl, decoded_bernoulli_logits

    def mean(self, x):
        if isinstance(x, tuple):
            y, *_ = x
            return y
        else:
            return x

    def loss(self, x, *args, prefix: str = "train"):
        beta = 0.1
        nelbo, _, _, decoded_bernoulli_logits = self.negative_elbo_bound(x, beta)
        loss = nelbo

        return loss, decoded_bernoulli_logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class VAEBernoulli(VAE):
    def likelihood(self, x, parameters):
        return self.bce(parameters, x.view(-1, 784))

    def mean(self, x):
        return x

    def loss(self, x, y, mu, logvar, prefix: str = "train"):
        bce = self.likelihood(x.view(-1, 784), y).sum()
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        loss = bce + kld

        self.log(f"{prefix}_bce", bce)
        self.log(f"{prefix}_kld", kld)
        self.log(f"{prefix}_loss", loss)

        return loss


class VAEContinuousBernoulli(VAE):
    def __init__(self, use_mean: bool = True):
        super().__init__()

        self.use_mean = use_mean

    def likelihood(self, x, parameters):
        cb = torch.distributions.ContinuousBernoulli(parameters)
        return cb.log_prob(x)

    def mean(self, x):
        if self.use_mean:
            cb = torch.distributions.ContinuousBernoulli(x)
            return cb.mean
        else:
            return x

    def loss(self, x, y, mu, logvar, prefix: str = "train"):
        ll = self.likelihood(x.view(-1, 784), y).sum()
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        loss = -ll + kld

        self.log(f"{prefix}_ll", ll)
        self.log(f"{prefix}_kld", kld)
        self.log(f"{prefix}_loss", loss)

        return loss


class VAEBeta(VAE):
    def __init__(self):
        super().__init__()

        self.decoder_2b = torch.nn.Linear(
            self.decoder_2.in_features, self.decoder_2.out_features
        )
        self.positive = torch.nn.Sigmoid()

    def decode(self, x):
        h = self.activation(self.decoder_1(x))
        alpha = self.positive(self.decoder_2(h))
        beta = self.positive(self.decoder_2b(h))
        return alpha, beta

    def likelihood(self, x, parameters):
        alpha, beta = parameters
        b = torch.distributions.Beta(alpha, beta)
        return b.log_prob(x).sum()

    def mean(self, x):
        alpha, beta = x
        b = torch.distributions.Beta(alpha, beta)
        return b.mean

    def loss(self, x, y, mu, logvar, prefix: str = "train"):
        ll = self.likelihood(torch.clip(x.view(-1, 784), 1e-2, 1 - 1e-2), y)
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        loss = -ll + kld

        self.log(f"{prefix}_ll", ll)
        self.log(f"{prefix}_kld", kld)
        self.log(f"{prefix}_loss", loss)

        return loss


class Diffusion(pl.LightningModule):
    def __init__(self, T: int) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 64, kernel_size=7, padding=3),
            torch.nn.ELU(),
            torch.nn.Conv2d(64, 64, kernel_size=9, padding=4),
            torch.nn.ELU(),
            torch.nn.Conv2d(64, 32, kernel_size=7, padding=3),
            torch.nn.ELU(),
            torch.nn.Conv2d(32, 16, kernel_size=5, padding=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(16, 8, kernel_size=3, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(8, 4, kernel_size=3, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(4, 1, kernel_size=1, padding=0),
        )

        self.time = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 784),
            torch.nn.ELU(),
        )

        self.T = T
        # self.betas = 0.02 * torch.ones(self.T)
        self.betas = torch.linspace(0.005, 0.02, T)
        # self.betas = torch.linspace(1e-4, 2e-2, T)

        self.activation = torch.nn.ReLU()

        self.omegas = 2 * math.pi * torch.tensor([1, 2, 4, 8]).reshape(1, -1) / self.T

        self._image_dir = None

    @property
    def image_dir(self):
        if self._image_dir is None:
            self._image_dir = os.path.join(self.logger.log_dir, "figures")
            os.makedirs(self._image_dir, exist_ok=True)

        return self._image_dir

    def forward(self, xs, ts):
        ts_embedding = torch.cos(
            self.omegas.to(self.device) * ts.reshape(-1, 1).to(self.device)
        )

        zs = xs + self.time(ts_embedding).reshape(-1, 1, 28, 28)
        zs = self.net(zs)

        return zs

    def training_step(self, batch, _):
        xs, _ = batch
        ts = torch.from_numpy(np.random.choice(self.T, xs.shape[0])).to(self.device)
        epsilons = torch.normal(0, 1, xs.shape, device=self.device)

        alpha_bar_ts = (
            torch.tensor(
                [torch.prod(1 - self.betas[:t]) for t in ts], device=self.device
            )
            .reshape(-1, 1, 1, 1)
            .repeat(1, 1, 28, 28)
        )

        xts = torch.sqrt(alpha_bar_ts) * xs + torch.sqrt(1 - alpha_bar_ts) * epsilons

        error = ((epsilons - self(xts, ts)) ** 2).sum(axis=-1).sum(-1).sum(-1).mean()
        self.log("error", error)

        return error

    def mu(self, xt, t):
        factor = 1 / torch.sqrt(1 - self.betas[t])
        alpha_bar_t = torch.prod(1 - self.betas[:t])
        return factor * (
            xt
            - self.betas[t] / torch.sqrt(1 - alpha_bar_t) * self(xt, torch.tensor([t]))
        )

    def generate_samples(self, n):
        with torch.no_grad():
            xs = torch.normal(
                0, 1, size=(n, 1, 28, 28), dtype=torch.float32, device=self.device
            )

            for t in range(self.T, 1, -1):
                xs = torch.normal(
                    self.mu(xs, t - 1), torch.sqrt(self.betas[-1]).to(self.device)
                )

        return torch.clip(xs, 0, 1)

    def validation_step(self, batch, batch_index):
        if self.trainer.current_epoch == 0:
            xs, _ = batch
            xs = xs[0].unsqueeze(0)
            ts = torch.tensor([0, 1, 3, 5, 10, 20, 50, 75, 100]).to(self.device)
            epsilons = torch.normal(0, 1, xs.shape, device=self.device)

            alpha_bar_ts = (
                torch.tensor(
                    [torch.prod(1 - self.betas[:t]) for t in ts], device=self.device
                )
                .reshape(-1, 1, 1, 1)
                .repeat(1, 1, 28, 28)
            )

            xts = (
                torch.sqrt(alpha_bar_ts) * xs + torch.sqrt(1 - alpha_bar_ts) * epsilons
            )
            torchvision.utils.save_image(
                xts.cpu(),
                os.path.join(self.image_dir, f"noises_{batch_index}.png"),
                nrow=3,
            )

        if self.trainer.current_epoch % 10 == 0 and batch_index == 0:
            samples = self.generate_samples(16)
            torchvision.utils.save_image(
                samples.cpu(),
                os.path.join(
                    self.image_dir, f"samples_{self.trainer.current_epoch}.png"
                ),
                nrow=4,
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
