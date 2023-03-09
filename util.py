"""
AI4ER GTC - Sea Ice Classification
Classes for loading the data for input to the models
and visualising the data
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
from torch.utils.data import Dataset
from torchvision import transforms
from pytorch_lightning import Callback
import pandas as pd


class SeaIceDataset(Dataset):
    """
    An implementation of a PyTorch dataset for loading sea ice SAR/chart image pairs.
    Inspired by https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
    """

    def __init__(self, sar_path: str, sar_files: list[str],
                 chart_path: str, chart_files: list[str],
                 transform: transforms.Compose = transforms.Compose([]),
                 class_categories: dict = None,
                 sar_band3: str = "angle"):
        """
        Constructs a SeaIceDataset.
        :param sar_path: Base folder path of SAR images
        :param sar_files: List of filenames of SAR images
        :param chart_path: Base folder path of charts
        :param chart_files: List of filenames of charts
        :param transform: Callable transformation to apply to images upon loading
        :param class_categories: Mapping from SIGRID codes to target classes
        :param sar_band3: "angle" to use SAR incidence angle or "ratio" to use HH/HV as third band
        """
        self.sar_path = sar_path
        self.sar_files = sar_files
        self.chart_path = chart_path
        self.chart_files = chart_files
        self.transform = transform
        self.class_categories = class_categories
        self.sar_band3 = sar_band3

        # read in precomputed mean and std deviation for HH, HV, incidence angle, and ratio
        metrics_df = pd.read_csv("metrics.csv", delimiter=",")
        self.hh_mean = metrics_df.iloc[0]["hh_mean"]
        self.hh_std = metrics_df.iloc[0]["hh_std"]
        self.hv_mean = metrics_df.iloc[0]["hv_mean"]
        self.hv_std = metrics_df.iloc[0]["hv_std"]
        self.angle_mean = metrics_df.iloc[0]["angle_mean"]
        self.angle_std = metrics_df.iloc[0]["angle_std"]
        self.ratio_mean = metrics_df.iloc[0]["hh_hv_mean"]
        self.ratio_std = metrics_df.iloc[0]["hh_hv_std"]

        # handle sar_band3
        if self.sar_band3 == "angle":
            self.band3_mean = self.angle_mean
            self.band3_std = self.angle_std
        else:
            self.sar_path = f"{self.sar_path}_band3"
            self.band3_mean = self.ratio_mean
            self.band3_std = self.ratio_std

    def __len__(self):
        """
        Implements the len(SeaIceDataset) magic method. Required to implement by Dataset superclass.
        When training/testing, this method tells our training loop how much longer we have to go in our Dataset.
        :return: Length of SeaIceDataset
        """
        return len(self.sar_files)

    def __getitem__(self, i: int):
        """
        Implements the SeaIceDataset[i] magic method. Required to implement by Dataset superclass.
        When training/testing, this method is used to actually fetch data and apply transformations.
        :param i: Index of which image pair to fetch
        :return: Dictionary with SAR and chart pair
        """

        # load data from files
        sar_name = f"{self.sar_path}/{self.sar_files[i]}"
        chart_name = f"{self.chart_path}/{self.chart_files[i]}"
        sar = rxr.open_rasterio(sar_name, masked=True).values  # take all bands for shape of l x w x 3
        chart = rxr.open_rasterio(chart_name, masked=True).values  # take array of shape l x w

        # recategorize classes 
        if self.class_categories is not None:
            for key, value in self.class_categories.items():
                chart[np.isin(chart, value)] = key

        # apply transforms
        sample = {"sar": sar, "chart": chart, "sar_name": sar_name, "chart_name": chart_name}
        if self.transform is not None:
            # Convert the data to tensors
            sar = torch.from_numpy(sar)
            chart = torch.from_numpy(chart)

            # normalise the sar data with mean and std deviation for each channel
            sar_transform = transforms.Compose([transforms.Normalize(mean=[self.hh_mean, self.hv_mean, self.band3_mean],
                                                                     std=[self.hh_std, self.hv_std, self.band3_std])])
            sar = sar_transform(sar)
            sample = {"sar": self.transform(sar), "chart": self.transform(chart).squeeze(0).long(),
                      "sar_name": sar_name, "chart_name": chart_name}

        return sample

    def visualise(self, i):
        """
        Allows us to visualise a particular SAR/chart pair.
        :param i: Index of which image pair to visualise
        :return: None
        """
        sample = self[i]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sample["sar"])
        ax[1].imshow(sample["chart"])
        plt.tight_layout()
        plt.show()


class Visualise(Callback):
    """
    Callback to visualise input/output samples and predictions.
    """

    def __init__(self, val_dataloader, n_to_show=5):
        """
        Construct callback object.
        :param val_dataloader: Validation dataloader to use when visualising outputs
        :param n_to_show: How many images to show
        """
        self.val_dataloader = val_dataloader
        self.n_to_show = n_to_show

    def on_validation_epoch_start(self, trainer, pl_module):
        """
        Callback to run on valiation epoch start.
        :param trainer: PyTorch Lightining Trainer class instance
        :param pl_module: PyTorch Lightning Module class instance
        """
        for batch in self.val_dataloader:
            x, y = batch["sar"].to(pl_module.device), batch["chart"].squeeze().long().to(pl_module.device)
            keep = y.sum(dim=[1, 2]) > 0  # keep only images with both classes
            x, y = x[keep], y[keep]
            x, y = x[:self.n_to_show], y[:self.n_to_show]  # keep only the first few images if there are more
            y_hat = pl_module(x)
            y_hat_pred = y_hat.argmax(dim=1)
            fig, ax = plt.subplots(self.n_to_show, 5, figsize=(15, self.n_to_show * 3))
            for i in range(self.n_to_show):
                a = x[i].detach().cpu().numpy().transpose(1, 2, 0)
                ax[i, 0].imshow(a[:, :, 0])
                ax[i, 0].set_title("SAR Band 1")
                ax[i, 1].imshow(a[:, :, 1])
                ax[i, 1].set_title("SAR Band 2")
                ax[i, 2].imshow(a[:, :, 2])
                ax[i, 2].set_title("SAR Band 3")
                ax[i, 3].imshow(y_hat_pred[i].detach().cpu().numpy())
                ax[i, 3].set_title("Prediction")
                ax[i, 4].imshow(y[i].detach().cpu().numpy())
                ax[i, 4].set_title("Truth")
            plt.tight_layout()
            wandb_logger = trainer.logger.experiment
            wandb_logger.log({"val_image": fig,
                              "val_image_x": x,
                              "val_image_y": y,
                              "val_image_y_hat": y_hat,
                              "val_image_y_hat_pred": y_hat_pred})
            plt.savefig(f"{wandb_logger.dir}/{trainer.global_step}.png")
            plt.close(fig)
            break  # only visualise from first batch
