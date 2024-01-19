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


def f(x):
    return (x - 0.5) * 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--flip', type=bool, default=False)
    args, _ = parser.parse_known_args()

    train_dataset = torchvision.datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    if args.flip:
        for i, (image, label) in enumerate(train_dataset):
            if label == 1 or label == 7:
                train_dataset.data[i] = ((1 - image) * 255).type(torch.uint8)

    train_data_loader = selector.add_arguments(
        parser, "train_data", torch.utils.data.DataLoader
    )(
        train_dataset,
        shuffle=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    if args.flip:
        for i, (image, label) in enumerate(test_dataset):
            if label == 1 or label == 7:
                test_dataset.data[i] = ((1 - image) * 255).type(torch.uint8)

    test_data_loader = selector.add_arguments(
        parser, "test_data", torch.utils.data.DataLoader
    )(
        test_dataset,
        shuffle=False,
    )

    model = selector.add_options_from_module(
        parser, "model", models, pl.LightningModule
    )()

    logger = CSVLogger("logs", model.__class__.__name__)
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args=parser,
        logger=logger,
        log_every_n_steps=1,
    )

    os.makedirs(logger.log_dir, exist_ok=True)
    with open(os.path.join(logger.log_dir, "arguments"), "x") as f:
        f.write(" ".join(sys.argv))

    print(datetime.datetime.now())
    trainer.fit(model, train_data_loader, test_data_loader)
    print(datetime.datetime.now())
