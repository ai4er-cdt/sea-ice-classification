import matplotlib.pyplot as plt
import rioxarray
from torch.utils.data import Dataset
from torchvision import transforms


class SeaIceDataset(Dataset):
    """
    An implementation of a PyTorch dataset for loading sea ice SAR/chart image pairs.
    Inspired by https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
    """

    def __init__(self,
                 sar_path: str,
                 sar_files: list[str],
                 chart_path: str,
                 chart_files: list[str],
                 transform: transforms = None):
        """
        Constructs a SeaIceDataset.
        :param sar_path: Base folder path of SAR images
        :param sar_files: List of filenames of SAR images
        :param chart_path: Base folder path of charts
        :param chart_files: List of filenames of charts
        :param transform: Callable transformation to apply to images upon loading
        """
        self.sar_path = sar_path
        self.sar_files = sar_files
        self.chart_path = chart_path
        self.chart_files = chart_files
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
        sar_name = f"{self.sar_path}/{self.sar_files[i]}"
        chart_name = f"{self.chart_path}/{self.chart_files[i]}"
        sar = rioxarray.open_rasterio(sar_name, masked=True).values  # take all bands for shape of 256 x 256 x 3
        chart = rioxarray.open_rasterio(chart_name, masked=True).values  # take binary array of shape 256 x 256
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
