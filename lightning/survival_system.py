import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from datasets.survial_dataset import SurvivalDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
# from dataloaders.dali import (
#     ExternalInputIterator,
#     ExternalSourcePipeline,
#     LightningDaliDataloader,
# )
# from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

from models.pconv_unet import PConvUNet
from models.vgg16_extractor import VGG16Extractor

from loss.loss_compute import LossCompute

from argparse import ArgumentParser
from utils import download_dataset, import_dataset_METABRIC, import_dataset_SYNTHETIC


class SurvivalSystem(pl.LightningModule):
    def __init__(self, hparams):
        """Initialize the module.
        Parameters
        ----------
        hparams
            `Namespace` object containing the model hyperparameters.
            Should usually be generated automatically by `argparse`.
        """
        super(SurvivalSystem, self).__init__()

        self.hparams = hparams

        # self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=5)
        # self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)
        # self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3)
        # self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3)
        # self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # self.global_pool = nn.AdaptiveAvgPool3d(1)
        # self.leaky_relu = nn.LeakyReLU(0.1)
        # self.fc = nn.Linear(512, 1)

        # self.conv_bn1 = nn.BatchNorm3d(self.conv1.out_channels)
        # self.conv_bn2 = nn.BatchNorm3d(self.conv2.out_channels)
        # self.conv_bn3 = nn.BatchNorm3d(self.conv3.out_channels)
        # self.conv_bn4 = nn.BatchNorm3d(self.conv4.out_channels)

        # self.apply(self.init_params)
        # INPUT DIMENSIONS
        # self.x_dim              = input_dims['x_dim']
        # self.num_Event          = input_dims['num_Event']
        # self.num_Category       = input_dims['num_Category']

        # # NETWORK HYPER-PARMETERS
        # self.h_dim_shared       = network_settings['h_dim_shared']
        # self.h_dim_CS           = network_settings['h_dim_CS']
        # self.num_layers_shared  = network_settings['num_layers_shared']
        # self.num_layers_CS      = network_settings['num_layers_CS']

        # self.active_fn          = network_settings['active_fn']
        # self.drop_rate          = network_settings['drop_rate']

        # layers = []

        # #shared FCN
        # for layer in range(self.num_layers_shared):
        #     if self.num_layers_shared == 1:
        #         layers.extend([nn.Linear(self.x_dim, self.h_dim_shared), self.active_fn])
        #     else:
        #         if layer == 0:
        #             layers.extend([nn.Linear(self.x_dim, self.h_dim_shared),
        #                            self.active_fn, nn.Dropout(p=self.drop_rate)])
        #         if layer >= 0 and layer != (self.num_layers_shared - 1):
        #             layers.extend([nn.Linear(self.h_dim_shared, self.h_dim_shared),
        #                            self.active_fn, nn.Dropout(p=self.drop_rate)])
        #         else:  # layer == num_layers-1 (the last layer)
        #             layers.extend([nn.Linear(self.h_dim_shared, self.h_dim_shared), self.active_fn])

        # self.shared = nn.Sequential(*layers)
        # self.shared.apply(init_weights)

        # layers = []

        # #CS FCN
        # for layer in range(self.num_layers_CS):
        #     if self.num_layers_CS == 1:
        #         layers.extend([nn.Linear(self.x_dim+self.h_dim_shared, self.h_dim_CS), self.active_fn])
        #     else:
        #         if layer == 0:
        #             layers.extend([nn.Linear(self.x_dim+self.h_dim_shared,
        #                            self.h_dim_CS), self.active_fn, nn.Dropout(p=self.drop_rate)])
        #         if layer >= 0 and layer != (self.num_layers_CS - 1):
        #             layers.extend([nn.Linear(self.h_dim_CS, self.h_dim_CS),
        #                            self.active_fn, nn.Dropout(p=self.drop_rate)])
        #         else:  # layer == num_layers-1 (the last layer)
        #             layers.extend([nn.Linear(self.h_dim_CS, self.h_dim_CS), self.active_fn])

        # self.CS = nn.Sequential(*layers)
        # self.CS.apply(init_weights)

        # layers = []

        # #Outputs Layer
        # layers.extend([nn.Dropout(p=self.drop_rate), nn.Linear(self.num_Event*self.h_dim_CS,
        #                 self.num_Event*self.num_Category), nn.Softmax(dim=-1)])

        # self.out = nn.Sequential(*layers)
        # self.out.apply(init_weights)

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. 'Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification',
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight)
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def prepare_data(self):
        download_dataset(self.hparams.data_name, self.hparams.data_root)
        if self.hparams.data_name == "METABRIC":
            alldata = import_dataset_METABRIC(self.hparams.data_root)
        elif self.hparams.data_name == "SYNTHETIC":
            alldata = import_dataset_SYNTHETIC(self.hparams.data_root)
        full_dataset = SurvivalDataset(alldata)

        # make sure the validation set is balanced
        full_indices = range(len(full_dataset))
        full_targets = full_dataset.label
        train_indices, test_indices = train_test_split(full_indices,
                                                       test_size=0.2,
                                                       stratify=full_targets)
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.2, stratify=full_targets[train_indices])
        train_dataset, val_dataset, test_dataset = (Subset(
            full_dataset, train_indices), Subset(
                full_dataset, val_indices), Subset(full_dataset, test_indices))
        # val_dataset.dataset = copy(full_dataset)
        # self.pos_weight = torch.tensor(len(full_targets) / full_targets.sum())

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def setup(self, step):
        # step is either 'fit' or 'test' 90% of the time not relevant
        _, num_event, num_category = self.train_dataset.dataset.mask1.shape  # dim of mask1: [subj, Num_Event, Num_Category]
        x_dim = self.train_dataset.dataset.data.shape[-1]

        # num_layers_shared = self.hparams.num_layers_shared
        # num_layers_CS = self.hparams.num_layers_CS  # case-specific
        # hidden_dim_shared = self.hparams.hidden_dim_shared
        # hidden_dim_CS = self.hparams.hidden_dim_CS
        # out_dim = self.hparams.out_dim


        activation = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh()
        }[self.hparams.activation]

        def create_FC(n_layer, in_dim, hidden_dim, out_dim):
            layers = nn.ModuleList()
            for i in range(1, n_layer):
                layers.extend([nn.Linear(in_dim, hidden_dim), activation, nn.Dropout(self.hparams.dropout)])
            # final
            if n_layer == 1:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            layers.apply(self.init_params)
            return nn.Sequential(*layers)

        self.shared_FC = create_FC(self.hparams.num_layers_shared, x_dim, self.hparams.hidden_dim_shared, self.hparams.hidden_dim_CS)
        self.case_specific_nets = nn.ModuleList([create_FC(self.hparams.num_layers_CS, self.hparams.hidden_dim_CS, self.hparams.hidden_dim_CS, self.hparams.out_dim) for i in range(num_event)])

        #Outputs Layer
        self.out_layer = nn.Sequential(nn.Dropout(self.hparams.dropout), nn.Linear(num_event * self.hparams.out_dim, num_event*num_category))
        self.out_layer.apply(self.init_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass on a batch of examples.
        Parameters
        ----------
        x
            A batch of examples.
        Returns
        -------
        torch.Tensor
            The predicted logits.
        """
        shared_out = self.shared_FC(x)
        # do not use residual connection due to https://github.com/chl8856/DeepHit/issues/1
        case_out = [case_net(shared_out) for case_net in self.case_specific_nets]
        out = self.out_layer(torch.cat(case_out, dim=1))
        return out

    def train_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def configure_optimizers(self):
        """This method is called automatically by pytorch-lightning."""
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": MultiStepLR(optimizer, milestones=[60, 160, 360]),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Run a single training step on a batch of samples.
        This method is called automatically by pytorch-lightning.
        """
        data, time, label, mask1, mask2 = batch
        output = self.forward(data)
        output = output.squeeze(1)
        # loss = 0.1 * self._dml_loss(embedding, y)
        loss = F.binary_cross_entropy_with_logits(output,
                                                  y.float(),
                                                  pos_weight=self.pos_weight)
        logs = {"training/loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        """Run a single validation step on a batch of samples.
        This method is called automatically by pytorch-lightning.
        """
        data, time, label, mask1, mask2 = batch
        # output, embedding = self.forward(x)
        output = self.forward(data)
        import ipdb; ipdb.set_trace(context=11)
        loss = F.binary_cross_entropy_with_logits(output,
                                                  y.float(),
                                                  pos_weight=self.pos_weight)
        pred_prob = torch.sigmoid(output)
        return {"loss": loss, "pred_prob": pred_prob, "y": y}

    def validation_epoch_end(self, outputs):
        """Compute performance metrics on the validation dataset.
        This method is called automatically by pytorch-lightning.
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob = torch.cat([x["pred_prob"]
                               for x in outputs]).detach().cpu().numpy()
        y = torch.cat([x["y"] for x in outputs]).detach().cpu().numpy()
        try:
            roc_auc = roc_auc_score(y, pred_prob)
        except ValueError:
            roc_auc = float("nan")
        avg_prec = average_precision_score(y, pred_prob)
        # log loss and metrics to Tensorboard
        log = {
            "validation/loss": loss,
            "validation/roc_auc": roc_auc,
            "validation/average_precision": avg_prec,
        }
        return {"val_loss": loss, "roc_auc": roc_auc, "log": log}

    def test_step(self, batch, batch_idx):
        """Run a single test step on a batch of samples.
        This method is called automatically by pytorch-lightning.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """Compute performance metrics on the test dataset.
        This method is called automatically by pytorch-lightning.
        """
        pred_prob = torch.cat([x["pred_prob"]
                               for x in outputs]).detach().cpu().numpy()
        y = torch.cat([x["y"] for x in outputs]).detach().cpu().numpy()
        try:
            roc_auc = roc_auc_score(y, pred_prob)
        except ValueError:
            roc_auc = float("nan")
        avg_prec = average_precision_score(y, pred_prob)
        ids = self.test_dataset.clinical_data["Study ID"]
        pd.Series(pred_prob, index=ids,
                  name="binary").to_csv(self.hparams.pred_save_path)
        return {"roc_auc": roc_auc, "average_precision": avg_prec}

    # def add_model_specific_args(parent_parser, root_dir):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        """Add model-specific hyperparameters to the parent parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
        parser.add_argument("--lr", type=float, default=3e-4, help="The initial learning rate.")
        parser.add_argument("--weight_decay", type=float, default=1e-5, help="The amount of weight decay to use.")
        parser.add_argument("--activation", type=str, default="relu", help="The batch size.")
        parser.add_argument("--dropout", type=float, default=0.5, help="The batch size.")
        parser.add_argument("--num_layers_shared", type=int, default=2, help="The batch size.")
        parser.add_argument("--num_layers_CS", type=int, default=2, help="The batch size.")
        parser.add_argument("--hidden_dim_shared", type=int, default=128, help="The batch size.")
        parser.add_argument("--hidden_dim_CS", type=int, default=128, help="The batch size.")
        parser.add_argument("--out_dim", type=int, default=128, help="The batch size.")
        return parser
