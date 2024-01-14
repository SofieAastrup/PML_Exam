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
    parser = pl.Trainer.add_argparse_args(parser)

    train_data_loader = selector.add_arguments(
        parser, "train_data", torch.utils.data.DataLoader
    )(
        dataset=torchvision.datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        shuffle=True,
    )
    test_data_loader = selector.add_arguments(
        parser, "test_data", torch.utils.data.DataLoader
    )(
        dataset=torchvision.datasets.MNIST(
            "data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
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
