import os
import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class SeaIceDataset(Dataset):
    """
    An implementation of a PyTorch dataset for loading sea ice SAR/chart image pairs.
    Inspired by https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
    """

    def __init__(self, sar_path: str, chart_path: str, transform: transforms = None):
        """
        Constructs a SeaIceDataset.
        :param sar_path: Path to the source folder of SAR images
        :param chart_path: Path to the source folder of corresponding chart labels
        :param transform: Callable transformation to apply to images upon loading
        """
        self.sar_path = sar_path
        self.sar_files = os.listdir(self.sar_path)
        self.chart_path = chart_path
        self.chart_files = os.listdir(self.chart_path)
        self.transform = transform

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
        When training/testing, this method is used to actually fetch data.
        :param i: Index of which image pair to fetch
        :return: Dictionary with SAR and chart pair
        """
        sar_name = os.path.join(self.sar_path, self.sar_files[i])
        chart_name = os.path.join(self.chart_path, self.chart_files[i])
        sar = io.imread(sar_name).copy()  # take all bands for shape of 256 x 256 x 3
        chart = io.imread(chart_name).copy()[:, :, 0]  # take red band only for shape of 256 x 256 x 1
        chart[chart < 80] = 0  # binarise to water
        chart[chart >= 80] = 255  # binarise to ice
        sample = {"sar": sar, "chart": chart}
        if self.transform:
            sample = {"sar": self.transform(sar), "chart": self.transform(chart).squeeze(0).long()}
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