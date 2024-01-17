import argparse
import datetime
import os
import sys

import pytorch_lightning as pl
import selector
import torch.utils.data
import torchvision
from pytorch_lightning.loggers.csv_logs import CSVLogger

import models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--accelerator', type=str)
    args, _ = parser.parse_known_args()

    model: models.VAE = selector.add_options_from_module(
        parser, "model", models, pl.LightningModule
    ).func.load_from_checkpoint(args.path).to(args.accelerator)
    # print(model)

    dataset=torchvision.datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    # test_data_loader = selector.add_arguments(
    #     parser, "test_data", torch.utils.data.DataLoader
    # )(
    #     ,
    #     shuffle=False,
    # )

    k = 14
    print(dataset[k][1])
    x = dataset[k][0].to(model.device).reshape(-1, 784)
    x = 1 - x
    # x = torch.zeros(1, 784, device=model.device)
    with torch.no_grad():
        print(model.sample_likelihood(x, n = args.n, importance=False))
        print(model.sample_likelihood(x, n = args.n, importance=True))

        y, mu, logvar = model(x)
        print(model.likelihood(x, y).sum())
