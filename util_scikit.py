"""
AI4ER GTC - Sea Ice Classification
Classes for loading the data for input to the models
"""
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
from numpy import ndarray
from xarray.core.dataarray import DataArray

def define_band3(sar: DataArray, sar_band3: str = 'angle') -> DataArray:
    
    """
    Defines the type of band in the third channel of the DataArray.
    The possible options are ratio and angle.

        Parameters:
            sar (xarray.core.dataarray.DataArray): SAR image
            sar_band3 (str): Name of 3rd band to return in the DataArray
        Returns:
            sar (xarray.core.dataarray.DataArray): DataArray with the 3rd band as specified
    """

    if sar_band3 == "ratio":

        band1 = sar.sel(band=1)
        band2 = sar.sel(band=2)
        band3 = sar.sel(band=3)
        band3.values = (band1.values / (band2.values + 0.0001))
        sar.loc[dict(band=3)] = band3

    return sar
    

def normalize_sar(sar: DataArray, sar_band3: str = 'angle') -> DataArray:
    
    """
    Normalises a SAR image with the mean and standard deviation of the whole
    training dataset. Returns a normalised DataArray

        Parameters:
            sar (xarray.core.dataarray.DataArray): SAR image
            sar_band3 (str): Name of 3rd band to use for metrics
        Returns:
            sar (xarray.core.dataarray.DataArray): DataArray with all the bands normalised
    """

    metrics_df = pd.read_csv("metrics.csv", delimiter=",")
    hh_mean = metrics_df.iloc[0]["hh_mean"]
    hh_std = metrics_df.iloc[0]["hh_std"]
    hv_mean = metrics_df.iloc[0]["hv_mean"]
    hv_std = metrics_df.iloc[0]["hv_std"]
    angle_mean = metrics_df.iloc[0]["angle_mean"]
    angle_std = metrics_df.iloc[0]["angle_std"]
    ratio_mean = metrics_df.iloc[0]["hh_hv_mean"]
    ratio_std = metrics_df.iloc[0]["hh_hv_std"]

    if sar_band3 == "angle":
        band3_mean = angle_mean
        band3_std = angle_std
    elif sar_band3 == "ratio":
        band3_mean = ratio_mean
        band3_std = ratio_std

    sar[0] = (sar[0] - hh_mean) / hh_std
    sar[1] = (sar[1] - hv_mean) / hv_std
    sar[2] = (sar[2] - band3_mean) / band3_std

    return sar


def recategorize_chart(chart: DataArray, class_categories: dict) -> DataArray:
    
    """
    Assigns new categories to an ice chart image and returns the corresponding DataArray.

        Parameters:
            chart (xarray.core.dataarray.DataArray): Ice chart image
            class_categories (dict): Dictionary of new class labels to be used
        Returns:
            chart (xarray.core.dataarray.DataArray): DataArray with the new class labels
    """
    
    if class_categories is not None:
        for key, value in class_categories.items():
            chart[np.isin(chart, value)] = key

    return chart


def load_sar(file_path: str, sar_band3: str, parse_coordinates: bool=True) -> ndarray:
    
    """
    Wrapper of the loading and processing functions for SAR images.
    Returns an ndarray.

        Parameters:
            file_path (str): Path to raster file
            parse_coordinates (bool): Parses the coordinates of the file, if any
            sar_band3 (str): Name of 3rd band to return in the DataArray
        Returns:
            sar (numpy.ndarray): DataArray with the new class labels
    """
    
    sar = rxr.open_rasterio(file_path, parse_coordinates=parse_coordinates)
    band3_sar = define_band3(sar, sar_band3)
    normalized_raster = normalize_sar(band3_sar, sar_band3)
    
    return normalized_raster.values


def load_chart(file_path: str, class_categories: dict, parse_coordinates: bool=True, masked: bool=True, flip_vertically: bool=False) -> ndarray:
    
    """
    Wrapper of the loading and processing functions for ice chart images.
    Returns an ndarray.

        Parameters:
            file_path (str): Path to raster file
            class_categories (dict): Dictionary of new class labels to be used
            parse_coordinates (bool): Parses the coordinates of the file, if any
            masked (bool): Reads raster as a mask
            flip_vertically (bool): Whether to flip the resulting DataArray vertically
        Returns:
            sar (numpy.ndarray): DataArray with the new class labels
    """
    
    chart = rxr.open_rasterio(file_path, parse_coordinates=parse_coordinates, masked=masked)
    if flip_vertically:
        chart.reindex(y=chart.y[::-1])
    new_raster = recategorize_chart(chart.values, class_categories)
    
    return new_raster

    
def crop_image(raster: DataArray, height_size: int, width_size: int) -> ndarray:
    
    """
    Crops a DataArray image according to input parameters.
    Returns an ndarray.

        Parameters:
            raster (xarray.core.dataarray.DataArray): Ice chart image
            height_size (int): Size in the y-axis for the resulting ndarray
            width_size (int): Size in the y-axis for the resulting ndarray
        Returns:
            raster (numpy.ndarray): ndarray with the specified size
    """
    
    _, height, width = raster.shape
    y_pad = (height-height_size) // 2
    x_pad = (width-width_size) // 2
    raster = raster[:, y_pad:y_pad+height_size, x_pad:x_pad+width_size]
    
    return raster
