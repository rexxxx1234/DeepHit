import os
from argparse import ArgumentParser
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.deephit import DeepHit as Model
from torchvision.transforms import Compose
from math import floor, pi
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset
from copy import copy

# from .transforms import *
# from .dataset import RadcureDataset

seed_everything(42)


def main(hparams):
    slurm_id = os.environ.get("SLURM_JOBID")
    if slurm_id is None:
        version = None
    else:
        version = str(slurm_id)
    logger = TensorBoardLogger(hparams.logdir, name=hparams.exp_name, version=version)
    checkpoint_path = os.path.join(
        logger.experiment.get_logdir(),
        "checkpoints",
        "{epoch:02d}-{val_loss:.2e}-{ci:.2f}",
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, save_top_k=5, monitor="ci", mode="max"
    )
    model = Model(hparams)
    trainer = Trainer.from_argparse_args(hparams, deterministic=True, benchmark=False)
    trainer.logger = logger
    trainer.checkpoint_callback = checkpoint_callback
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory where training logs will be saved.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Directory where training logs will be saved.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="METABRIC",
        choices=["METABRIC", "SYNTHETIC"],
        help="Directory where training logs will be saved.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes to use for data loading.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="simple_cnn",
        help="Experiment name for logging purposes.",
    )
    parser.add_argument(
        "--model_type", type=str, default="resnet", help="Which model to use."
    )
    parser.add_argument(
        "--model_depth",
        type=int,
        default=50,
        help="Number of depth of the ResNet model.",
    )

    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    main(hparams)
