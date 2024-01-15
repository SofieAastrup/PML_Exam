import os
import os.path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.distributions
import torch.nn
import torch.optim
import torchvision
import torchvision.datasets
import torchvision.transforms
import torchvision.utils


class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder_1 = torch.nn.Linear(784, 400)
        self.encoder_2a = torch.nn.Linear(400, 2)
        self.encoder_2b = torch.nn.Linear(400, 2)

        self.decoder_1 = torch.nn.Linear(2, 400)
        self.decoder_2 = torch.nn.Linear(400, 784)

        self.activation = torch.nn.ReLU()
        self.bce = torch.nn.BCELoss(reduction="sum")

        self._image_dir = None

    def encode(self, x):
        h = self.activation(self.encoder_1(x))
        return self.encoder_2a(h), self.encoder_2b(h)

    def decode(self, x):
        h = self.activation(self.decoder_1(x))
        return torch.sigmoid(self.decoder_2(h))

    def mean(self, x):
        ...

    def images(self, x):
        return self.mean(self.decode(x))

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

        z, _ = self.encode(x.view(-1, 784))
        scatter = plt.scatter(z[:, 1].cpu(), -z[:, 0].cpu(), c=l.cpu(), cmap="tab10", s=2, alpha=0.7)
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


class VAEBernoulli(VAE):
    def mean(self, x):
        return x

    def loss(self, x, y, mu, logvar, prefix: str = "train"):
        bce = self.bce(y, x.view(-1, 784))
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

    def mean(self, x):
        if self.use_mean:
            cb = torch.distributions.ContinuousBernoulli(x)
            return cb.mean
        else:
            return x

    def loss(self, x, y, mu, logvar, prefix: str = "train"):
        bce = self.bce(y, x.view(-1, 784))
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        cb = torch.distributions.ContinuousBernoulli(y)
        loss = bce + kld + cb._cont_bern_log_norm().sum()

        self.log(f"{prefix}_bce", bce)
        self.log(f"{prefix}_kld", kld)
        self.log(f"{prefix}_cb_norm", cb._cont_bern_log_norm().sum())
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
        alpha = self.softplus(self.decoder_2(h))
        beta = self.softplus(self.decoder_2b(h))
        return alpha, beta

    def mean(self, x):
        alpha, beta = x
        b = torch.distributions.Beta(alpha, beta)
        return b.mean

    def loss(self, x, y, mu, logvar, prefix: str = "train"):
        alpha, beta = y
        b = torch.distributions.Beta(alpha, beta)
        ll = b.log_prob(torch.clip(x.view(-1, 784), 1e-2, 1 - 1e-2)).sum()
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        loss = -ll + kld

        self.log(f"{prefix}_ll", ll)
        self.log(f"{prefix}_kld", kld)
        self.log(f"{prefix}_loss", loss)

        return loss
